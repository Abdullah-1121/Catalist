# Write-up: meeting memory agent

## Architecture

```
transcripts ─► parser (code) ─► Postgres: meetings, turns (turn_id = TS3012b#112)
                    │
                    ▼  per team, meetings in order a→d; teams independent
       windows ≤ ~10k tokens, cut between turns (code)
                    │
                    ▼  prompt = team registry (current cards) + window turns
       Gemma 4 (open weights) ─► cards: proposal | decision | action | number | disagreement
                    │
                    ▼  code: schema rules, evidence inside window, supersedes→decision,
       cards table (append-only)   answers→action, same team; rejects stored with reason

question ─► LangGraph loop: decide (LLM) ─► search / read turns (code, team-filtered) ─► … ─► answer
                                                                     │
                       code turns cited turns into exact ≥6-word quotes ◄┘ (never trusts LLM text)
```

**Memory over time.** Cards are an append-only event log. When a team changes its mind, the extractor writes a *new* decision with `supersedes = <old card id>`. "Current" is computed (not superseded / not answered by a later card), never stored, so it can't go stale and the history stays available ("they reversed in meeting c"). The LLM can only link to card ids it was shown in the team registry, and code verifies the link (target exists, same team, right type). Implicit back-references ("let's go with the other one") are resolved at ingest time, while that meeting's context and the team's current state are both in the prompt.

**Scale (10,000 meetings, growing daily).** Nothing needs the whole corpus in one context. Ingest cost is per window (bounded), the registry is per team (one line per topic, so it grows with topics, not meetings), and every query is filtered to one team in SQL before the LLM sees anything. Teams are independent, so ingest parallelises across teams; within a team, meetings run in order. New meetings are incremental: content hash → unchanged files are skipped; a changed file re-extracts that meeting and the later meetings of its team only. Every LLM call is cached in Postgres (sha256 key), so re-runs are free and reproducible.

## What the LLM does vs what code does

| LLM (judgement) | Code (exact, cheap, testable) |
|---|---|
| Spot proposals/decisions/actions/numbers/disagreements in messy speech | Parse files, turn ids, ordering, windows, token estimates |
| Name topics, reuse registry names; link a decision to the one it replaces | Validate every card; check links point at real cards of the right type |
| Decide the next agent step; write 1–3 sentence answers | Team filtering, full-text search, reading turns, step limit |
| Group the brief's points | Every quote: copied from the turn by code, verified ≥6 words exact; points without a verifiable source are dropped |

Quotes are the clearest case: the LLM paraphrases and "cleans up" disfluent speech, and the scoring is word-for-word. So the LLM only *points* at turns (or cards). Code copies the words, falling back to the run of real words closest to the claim.

## Tools, and where they broke

- **Model: Gemma 4 (26B MoE), open weights, through Google AI Studio's free OpenAI-compatible endpoint.** The constraint was open-weight models only and no budget. The local Intel Mac is too slow for inference. In production the same weights run self-hosted (vLLM), which is a config change (`LLM_BASE_URL`).
  - The free tier is the bottleneck: 16k input tokens per minute per model. 503s came in multi-minute bursts this morning; the 31B model was unusable (every team failed on 503s). 429s carry no `Retry-After` header, only "retry in Ns" in the message text, so we parse it. The SDK's default timeout (600 s) let a call hang → 120 s timeout plus our own retry (logged, 30 s backoff on 5xx).
  - Gemma always "thinks" and the thinking can't be switched off. With `json_schema` structured output the thinking does not leak into the content, and calls were ~2× faster (20 s vs ~50 s).
- **LangChain `with_structured_output(method="json_schema")`** worked first try. We pass a plain dict schema and validate each card ourselves with pydantic. A single wrapper model would reject a whole window for one bad card.
- **LangChain `SQLAlchemyCache`** broke: it uses the whole prompt as primary key, and Postgres can't index a 22 kB key → own cache with a sha256 key.
- **LangGraph** for the agent loop: an explicit `StateGraph` (decide → act → decide … → finish), so the loop and its step limit are visible. The LLM picks actions as JSON rather than via function calling, because function calling was untested on this endpoint and `json_schema` was proven.
- **Postgres** (with pgvector available): one store for turns, cards, rejects and cache. Search is Postgres full-text (`websearch_to_tsquery`, words OR-ed). **Embeddings were planned** (Cloudflare bge-m3, tested working) **and cut for the deadline**. Keyword search misses paraphrases; at 10k meetings we'd add a pgvector HNSW index on turns and cards.
- **Rules of thumb vs measurement:** "4 chars per token" is wrong for this data. Messy speech measured 3.13 chars/token, so ÷4 would undercount by 28% and overflow the rate limit. We estimate with ÷3 (verified: estimated 9,999 tokens of turns → 9,256 real tokens for the whole prompt).

## Extraction quality: what we measured

- Hand-check on TS3012a/b: decisions and numbers correct (25 € selling price, 12.50 € production cost; LCD dropped for cost and battery; no speech recognition). Errors found: noisy evidence turns, one one-person "disagreement" (→ new rule: a disagreement needs 2+ speakers), messy topic names (→ registry + clean-up in code).
- Under-extraction: TS3012c gave **2 cards from 653 turns**. One prompt line ("a typical window holds 10–25 cards; don't skip proposals, numbers, decisions") → **25 cards**.
- Rejected cards are stored with the reason and counted per meeting, never silently dropped. A dropped "decision that supersedes" would leave the old decision looking current, which is worse than a missing card because it looks correct.

## Evaluation

RESULTS_PLACEHOLDER

## Cost and time to ingest 1,000 meetings

COST_PLACEHOLDER

## Three questions it got wrong, and why

WRONG_PLACEHOLDER

## Known limits / next steps

- Embedding search (pgvector) alongside keyword search; a reranker for long teams.
- Show only the registry topics relevant to the window once a team has hundreds of topics (shortcut today: the whole registry, capped per topic).
- Speaker roles are found by the agent at query time (people introduce themselves); a per-meeting role table would make it cheaper.
- If several proposals are open when a window is cut, a later decision may link to the wrong one (no window overlap, by choice: overlap costs tokens and duplicates cards).
