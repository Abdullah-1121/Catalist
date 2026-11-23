"""Query-time agent: an explicit LangGraph loop (D9).

decide (LLM: search / read / answer)  ->  act (code runs the tool)  ->  decide ...
                                      ->  finish (code turns turn_ids into verified quotes, D2)
The team filter is code, not LLM: every tool only sees the named team's data.
"""

import re
from functools import partial
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from sqlalchemy import Engine, func, select
from sqlalchemy.orm import Session

from catalist.extract import live_cards
from catalist.llm import call_with_retry
from catalist.quotes import best_quote
from catalist.store import CardRow, MeetingRow, TurnRow

TURN_ID = re.compile(
    r"[A-Z]{2}\d{4}[a-z]#\d+"
)  # the LLM sometimes wraps ids: "[TS3012b#112 A]"
CARD_ID = re.compile(r"[A-Z]{2}\d{4}[a-z]-\d+")
MAX_STEPS = 4  # tool calls per question; free tier = ~1 call per minute per model

STEP_SCHEMA = {
    "title": "Step",
    "type": "object",
    "properties": {
        "thought": {
            "type": "string",
            "description": "What you know and what is missing",
        },
        "action": {"type": "string", "enum": ["search", "read", "answer"]},
        "query": {
            "type": "string",
            "description": "search: keywords, e.g. 'lcd screen cost'",
        },
        "turn_id": {
            "type": "string",
            "description": "read: a turn id like TS3006b#112",
        },
        "answer": {"type": "string", "description": "answer: 1-3 plain sentences"},
        "card_ids": {
            "type": "array",
            "items": {"type": "string"},
            "description": "answer: ids of the TEAM MEMORY cards the answer relies on",
        },
        "citations": {
            "type": "array",
            "description": "answer: 1-3 turns that prove it, each with words copied exactly",
            "items": {
                "type": "object",
                "properties": {
                    "turn_id": {"type": "string"},
                    "quote": {"type": "string"},
                },
                "required": ["turn_id", "quote"],
            },
        },
    },
    "required": ["thought", "action"],
}

PROMPT = """You answer questions about the meetings of team {team}, a product-design team designing a TV remote control (meetings a, b, c, d in time order).
Speakers are letters (A-D). Roles (Project Manager, Industrial Designer, User Interface Designer, Marketing Expert) are often stated by the speakers themselves: give the role with the letter when the transcript makes it clear.

TEAM MEMORY: cards extracted from the meetings, in time order. CURRENT = not replaced later; "replaced by X" = a later card changed it.
{memory}

Memory can miss changes made late in a meeting. If the question asks how something ended, what was final, or whether something was kept, search the team's LAST meeting (d) for that topic before you answer.

QUESTION: {question}

STEPS SO FAR:
{steps}

Choose the next action:
- search: keyword search over this team's transcript turns (to find details or the exact words someone used).
- read: read one turn and its neighbours in full (needed before you quote it).
- answer: 1-3 sentences that answer every part of the question; card_ids of the memory cards you used; and 1-3 citations (turn_id + at least 6 words copied exactly from that turn). Prefer turns you have read, from at least two different meetings when the answer spans meetings.
{force}"""


class State(TypedDict):
    team: str
    question: str
    steps: list[str]
    result: dict


def overlap(a: str, b: str) -> float:
    words = lambda t: {w for w in re.findall(r"[a-z0-9]+", t.lower()) if len(w) > 3}
    x, y = words(a), words(b)
    return len(x & y) / (len(x | y) or 1)


def memory_text(cards: list[CardRow]) -> str:
    live = {c.card_id for c in live_cards(cards)}
    replaced_by = {c.supersedes: c.card_id for c in cards if c.supersedes}
    replaced_by |= {c.answers: c.card_id for c in cards if c.answers}
    lines = []
    for c in sorted(cards, key=lambda c: (c.topic, c.order)):
        value = f" = {c.value:g} {c.unit}" if c.value is not None else ""
        status = (
            "CURRENT" if c.card_id in live else f"replaced by {replaced_by[c.card_id]}"
        )
        lines.append(
            f"{c.topic} | [{c.card_id}] {c.meeting_id} {c.type} ({','.join(c.who)}){value}: "
            f"{c.summary} {c.reason} {c.positions} | {status} | evidence {' '.join(c.evidence)}"
        )
    return "\n".join(lines) or "(no cards)"


def search(session: Session, team: str, query: str, limit: int = 8) -> str:
    # shortcut: full-text search instead of embeddings (deadline); words OR-ed so one miss doesn't zero the result
    terms = " or ".join(query.split())
    vector = func.to_tsvector("english", TurnRow.text)
    tsq = func.websearch_to_tsquery("english", terms)
    rows = session.scalars(
        select(TurnRow)
        .join(MeetingRow)
        .where(MeetingRow.team == team, vector.bool_op("@@")(tsq))
        .order_by(func.ts_rank(vector, tsq).desc())
        .limit(limit)
    )
    return (
        "\n".join(f"[{t.turn_id} {t.speaker}] {t.text[:300]}" for t in rows)
        or "no hits"
    )


def read(session: Session, team: str, turn_id: str) -> str:
    turn = session.get(TurnRow, turn_id)
    if turn is None or not turn_id.startswith(team):
        return f"no turn {turn_id} in team {team}"
    rows = session.scalars(
        select(TurnRow)
        .where(
            TurnRow.meeting_id == turn.meeting_id,
            TurnRow.index.between(turn.index - 2, turn.index + 2),
        )
        .order_by(TurnRow.index)
    )
    return "\n".join(f"[{t.turn_id} {t.speaker}] {t.text}" for t in rows)


def build(engine: Engine, llm):
    decide_llm = llm.with_structured_output(STEP_SCHEMA, method="json_schema")
    # Last step: the schema itself forces an answer (forcing only the action left answers empty)
    final_schema = {
        **STEP_SCHEMA,
        "properties": {
            **STEP_SCHEMA["properties"],
            "action": {"type": "string", "enum": ["answer"]},
        },
        "required": ["thought", "action", "answer", "card_ids", "citations"],
    }
    final_llm = llm.with_structured_output(final_schema, method="json_schema")

    def decide(state: State) -> dict:
        with Session(engine) as session:
            cards = list(
                session.scalars(select(CardRow).where(CardRow.team == state["team"]))
            )
        last = len(state["steps"]) >= MAX_STEPS
        prompt = PROMPT.format(
            team=state["team"],
            memory=memory_text(cards),
            question=state["question"],
            steps="\n\n".join(state["steps"]) or "(none)",
            force="You have used all your steps: you MUST answer now." if last else "",
        )
        model = final_llm if last else decide_llm
        step = call_with_retry(partial(model.invoke, prompt)) or {"action": "answer"}
        if step.get("action") == "answer" and not step.get("answer") and not last:
            step = call_with_retry(partial(final_llm.invoke, prompt)) or step
        if last:
            step["action"] = "answer"
        return {"result": step}

    def act(state: State) -> dict:
        step = state["result"]
        with Session(engine) as session:
            if step["action"] == "search":
                seen = search(session, state["team"], step.get("query", ""))
            else:
                seen = read(session, state["team"], step.get("turn_id", ""))
        note = f"{step['action']} {step.get('query') or step.get('turn_id')}:\n{seen}"
        print(
            f"  {state['question'][:40]}... -> {step['action']} {step.get('query') or step.get('turn_id')}",
            flush=True,
        )
        return {"steps": state["steps"] + [note]}

    def finish(state: State) -> dict:
        step, team = state["result"], state["team"]
        answer = step.get("answer", "")
        with Session(engine) as session:
            cards = {
                c.card_id: c
                for c in session.scalars(select(CardRow).where(CardRow.team == team))
            }
            used = [
                cards[i]
                for raw in step.get("card_ids", [])
                for i in CARD_ID.findall(str(raw))
                if i in cards
            ]
            closest = sorted(cards.values(), key=lambda c: -overlap(answer, c.summary))
            # Tiers, first one that yields a quote wins (D2: quotes always made by code):
            # 1. turns the LLM cited  2. evidence of cards it used  3. cards closest to the answer
            tiers = [
                [
                    (tid, c.get("quote", ""))
                    for c in step.get("citations", [])
                    for tid in TURN_ID.findall(str(c.get("turn_id", "")))
                ],
                [(c.evidence[0], c.summary) for c in used],
                [(c.evidence[0], c.summary) for c in closest[:3]] if answer else [],
            ]
            citations = []
            for tier in tiers:
                for turn_id, claim in tier:
                    turn = session.get(TurnRow, turn_id)
                    if turn is None or not turn.turn_id.startswith(team):
                        continue  # invented or other-team turn: dropped
                    quote = best_quote(claim or answer, turn.text)
                    if quote and len(citations) < 3:
                        citations.append(
                            {"meeting_id": turn.meeting_id, "quote": quote}
                        )
                if citations:
                    break
        return {"result": {"answer": answer, "citations": citations}}

    graph = StateGraph(State)
    graph.add_node("decide", decide)
    graph.add_node("act", act)
    graph.add_node("finish", finish)
    graph.add_edge(START, "decide")
    graph.add_conditional_edges(
        "decide", lambda s: "finish" if s["result"]["action"] == "answer" else "act"
    )
    graph.add_edge("act", "decide")
    graph.add_edge("finish", END)
    return graph.compile()


def ask(agent, team: str, question: str) -> dict:
    return agent.invoke(
        {"team": team, "question": question, "steps": [], "result": {}}
    )["result"]


BRIEF_SCHEMA = {
    "title": "Brief",
    "type": "object",
    "properties": {
        "sections": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "heading": {
                        "type": "string",
                        "enum": [
                            "Decisions",
                            "What changed",
                            "Open actions",
                            "Open disagreements",
                        ],
                    },
                    "points": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "text": {
                                    "type": "string",
                                    "description": "One sentence; owners by letter",
                                },
                                "card_ids": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                            },
                            "required": ["text", "card_ids"],
                        },
                    },
                },
                "required": ["heading", "points"],
            },
        }
    },
    "required": ["sections"],
}

BRIEF_PROMPT = """Write the brief team {team} reads before its final meeting, from its memory below (meetings before the final one only).
Sections: Decisions (what is decided now, CURRENT cards), What changed (decisions that replaced earlier ones, numbers that moved),
Open actions (tasks with owner letter, not yet reported back), Open disagreements (still unresolved).
If a later decision on the same topic contradicts an earlier one (even if it is not marked "replaced by"), list only the later one under Decisions and describe the change under What changed.
An action is open only if no later card shows it was done, reported back or made moot.
Every point must list the card_ids it comes from. Do not invent anything that is not in the memory.

MEMORY:
{memory}"""


def brief(engine: Engine, llm, team: str, before_order: int) -> str:
    """LLM groups and words the points; code attaches a verified quote for every point."""
    with Session(engine) as session:
        cards = list(
            session.scalars(
                select(CardRow).where(
                    CardRow.team == team, CardRow.order < before_order
                )
            )
        )
        by_id = {c.card_id: c for c in cards}
        out = call_with_retry(
            partial(
                llm.with_structured_output(BRIEF_SCHEMA, method="json_schema").invoke,
                BRIEF_PROMPT.format(team=team, memory=memory_text(cards)),
            )
        )
        lines = [f"Brief for team {team}, before its final meeting"]
        for section in out["sections"]:
            lines.append(f"\n{section['heading'].upper()}")
            for point in section["points"]:
                cites = []
                for card_id in point["card_ids"]:
                    card = by_id.get(card_id)
                    turn = card and session.get(TurnRow, card.evidence[0])
                    quote = turn and best_quote(card.summary, turn.text)
                    if quote:
                        cites.append(f'{turn.meeting_id}: "{quote}"')
                if cites:  # a point without a verifiable source is dropped, not printed uncited
                    lines.append(f"- {point['text']} [{'; '.join(cites[:2])}]")
        return "\n".join(lines)


def valid_citations(engine: Engine, citations: list[dict]) -> int:
    """How many citations pass the official rule: >= 6 words, exact, inside one turn of that meeting."""
    from catalist.quotes import is_exact

    with Session(engine) as session:
        return sum(
            any(
                is_exact(c["quote"], t.text)
                for t in session.scalars(
                    select(TurnRow).where(TurnRow.meeting_id == c["meeting_id"])
                )
            )
            for c in citations
        )
