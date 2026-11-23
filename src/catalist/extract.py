"""Ingest: turn a team's meetings into cards with the LLM, meeting by meeting, window by window.

LLM: reads a window + the team's memory, proposes cards.
Code: validates every card (D22, D23), assigns ids, maps evidence to turn_ids,
decides what is current (D20), skips unchanged meetings (D26).
"""

import hashlib
import time
from functools import partial
from pathlib import Path

from pydantic import ValidationError
from sqlalchemy import Engine, delete, select, update
from sqlalchemy.orm import Session

from catalist.cards import Card
from catalist.llm import call_with_retry
from catalist.parser import parse_meeting
from catalist.store import CardRow, MeetingRow, RejectedCardRow, store_meeting
from catalist.windows import label, windows

# Cards as plain dicts so each card is validated on its own (D23).
SCHEMA = {
    "title": "Cards",
    "description": "All cards extracted from this meeting window.",
    "type": "object",
    "properties": {"cards": {"type": "array", "items": Card.model_json_schema()}},
    "required": ["cards"],
}

PROMPT = """You keep the memory of a product-design team that designs a TV remote control over four meetings.
Read one window of a meeting transcript and extract a card for every:
- proposal: someone suggests an option (who = the proposer).
- decision: the team settles on something (who = who decided or agreed). If it changes, reverses or drops an earlier decision listed in TEAM MEMORY, set supersedes = that card id.
- action: a task someone takes on or is given (who = the owner).
- number: a price, cost, total, budget, limit or percentage (value + unit; who = who said it).
- disagreement: two or more people take different positions (who = all of them; positions = who thinks what, and why).
If a card reports back on an open action in TEAM MEMORY, set answers = that action's card id.

Rules:
- topic: short snake_case subject, e.g. case_material, energy_source, lcd_screen, selling_price. If a topic in TEAM MEMORY fits, reuse its name exactly.
- who: speaker letters only (A, B, C, D).
- evidence: the 1-3 turn numbers [#n] where it is said most clearly. Only numbers shown in this window.
- summary: one plain sentence with the concrete content (what, numbers, reasons).
- Be thorough: a typical window holds 10-25 cards. Do not skip proposals, numbers (prices, costs, totals) or decisions, even small ones.
- Do not repeat a card already in TEAM MEMORY unless something changed. Small talk, jokes and meeting logistics are not cards.

TEAM MEMORY (earlier cards that are still current; ids in brackets):
{registry}

MEETING {meeting_id}, window {n} of {total}:
{turns}"""

ALWAYS_SHOWN = {"decision", "action"}


def check_card(card: Card, window: set[int], known: dict[str, str]) -> str | None:
    """Checks that need outside information (D22). known = card_id -> type, same team."""
    if outside := [i for i in card.evidence if i not in window]:
        return f"evidence {outside} is not in this window"
    if card.supersedes and known.get(card.supersedes) != "decision":
        return f"supersedes {card.supersedes!r} is not a known decision"
    if card.answers and known.get(card.answers) != "action":
        return f"answers {card.answers!r} is not a known action"
    return None


def live_cards(cards: list[CardRow]) -> list[CardRow]:
    """Current = not superseded and not answered by a later card (D20, computed, never stored)."""
    replaced = {c.supersedes for c in cards} | {c.answers for c in cards}
    return [c for c in cards if c.card_id not in replaced]


def registry_text(cards: list[CardRow]) -> str:
    by_topic: dict[str, list[CardRow]] = {}
    for card in live_cards(cards):
        by_topic.setdefault(card.topic, []).append(card)
    lines = []
    for topic, group in by_topic.items():
        # shortcut: last 3 proposals/numbers/disagreements per topic; past ~100 topics, show only relevant ones
        recent = [c for c in group if c.type not in ALWAYS_SHOWN][-3:]
        lines.append(f"{topic}:")
        for c in group:
            if c.type in ALWAYS_SHOWN or c in recent:
                value = f" = {c.value:g} {c.unit}" if c.value is not None else ""
                lines.append(
                    f"  [{c.card_id}] {c.type} ({','.join(c.who)}){value}: {c.summary}"
                )
    return "\n".join(lines) or "(none yet)"


def ingest_team(engine: Engine, llm, paths: list[Path]) -> None:
    meetings = sorted(
        ((parse_meeting(p), hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths),
        key=lambda m: m[0].order,
    )
    redo = (
        False  # once a meeting changes, every later meeting of the team is redone (D26)
    )
    for meeting, digest in meetings:
        with Session(engine) as session:
            stored = session.get(MeetingRow, meeting.meeting_id)
            redo = redo or stored is None or stored.content_hash != digest
            if not redo:
                print(f"{meeting.meeting_id}: unchanged, skipped")
                continue
            earlier = list(
                session.scalars(
                    select(CardRow).where(
                        CardRow.team == meeting.team, CardRow.order < meeting.order
                    )
                )
            )
        store_meeting(engine, meeting)  # hash stays empty until the cards are stored
        start = time.time()
        kept, rejected = [], []
        parts = windows(meeting.turns)
        for n, part in enumerate(parts):
            prompt = PROMPT.format(
                registry=registry_text(earlier + kept),
                meeting_id=meeting.meeting_id,
                n=n + 1,
                total=len(parts),
                turns="\n".join(label(t) for t in part),
            )
            runnable = llm.with_structured_output(
                SCHEMA, method="json_schema", include_raw=True
            )
            out = call_with_retry(partial(runnable.invoke, prompt))
            found = len((out["parsed"] or {}).get("cards", []))
            if found <= 2 and len(part) > 100:  # Gemma sometimes stops after one card
                retry = f"{prompt}\n\nA first pass found only {found} card(s) in these {len(part)} turns. That is far too few: read the whole window again and list ALL cards."
                second = call_with_retry(partial(runnable.invoke, retry))
                if len((second["parsed"] or {}).get("cards", [])) > found:
                    out = second
            if out["parsing_error"] or not out["parsed"]:
                raw = {"content": str(out["raw"].content)[:20000]}
                rejected.append(
                    RejectedCardRow(
                        meeting_id=meeting.meeting_id,
                        window=n,
                        reason=f"window unparseable: {out['parsing_error']}",
                        raw=raw,
                    )
                )
                continue
            known = {c.card_id: c.type for c in earlier + kept}
            indices = {t.index for t in part}
            for item in out["parsed"].get("cards", []):
                try:
                    card = Card.model_validate(item)
                    error = check_card(card, indices, known)
                except ValidationError as e:
                    error = str(e.errors()[0]["msg"])
                if error:
                    rejected.append(
                        RejectedCardRow(
                            meeting_id=meeting.meeting_id,
                            window=n,
                            reason=error,
                            raw=item,
                        )
                    )
                    continue
                card_id = f"{meeting.meeting_id}-{len(kept) + 1}"
                known[card_id] = card.type
                kept.append(
                    CardRow(
                        card_id=card_id,
                        meeting_id=meeting.meeting_id,
                        team=meeting.team,
                        order=meeting.order,
                        evidence=[meeting.turns[i].turn_id for i in card.evidence],
                        **card.model_dump(exclude={"evidence"}),
                    )
                )
        # One transaction: old cards out, new cards in, hash set -> a crash leaves the hash empty = redo
        with Session(engine) as session, session.begin():
            for table in (CardRow, RejectedCardRow):
                session.execute(
                    delete(table).where(table.meeting_id == meeting.meeting_id)
                )
            session.add_all(kept + rejected)
            session.execute(
                update(MeetingRow)
                .where(MeetingRow.meeting_id == meeting.meeting_id)
                .values(content_hash=digest)
            )
        print(
            f"{meeting.meeting_id}: {len(parts)} windows, kept {len(kept)}, "
            f"rejected {len(rejected)}, {time.time() - start:.0f}s"
        )
