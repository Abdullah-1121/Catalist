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

Own set: 10 questions on TS3012 (`eval/questions.json`), each with the expected answer and the exact turns that prove it, checked by hand. `catalist eval` runs the agent, auto-checks expected facts (keywords) and verifies every citation against the transcript; the final verdict is by hand (`eval/results.json`).

| Run | Correct | Half | Wrong | Valid citations |
|---|---|---|---|---|
| 1 (first agent version) | 8 | 0 | 2 (empty answers) | 13/13, but 3 answers had none |
| 2 (after fixes) | 7 | 2 | 1 | **19/19**, every answer cited |

Score run 2: **8/10** (half = 0.5). Fixes between runs came from the eval itself: code-made citations with 3 fallback tiers (an answer is never uncited), and a final step whose schema *requires* an answer (the LLM sometimes chose "answer" with no text).
Hand-checked official answers so far: Q01, Q13, Q21, Q22, Q24 all correct, each with valid citations (the evidence turns are listed in `eval/official_checks.md`).

## Cost and time to ingest 1,000 meetings

Measured (free tier, Gemma 4 26B/31B, 2026-11-23): median **111 s per meeting**, mean 185 s, worst 843 s, almost all of it waiting on rate limits (16k input tokens/min per model) and 503/429 retries; ~1.5 windows per meeting; 9% of cards rejected by the checks.

Tokens per meeting (from the transcripts, measured 3.13 chars/token): ~9.5k of turns, plus ~3k of instructions + registry per window (×1.5), plus re-asks on lazy windows (~1 in 3 windows) → **~18k input + ~2.5k output tokens per meeting**.

| 1,000 meetings | Time | Cost |
|---|---|---|
| Free tier, one model (16k TPM) | 18M tokens / 16k per min ≈ **19 h** (measured pace with retries: ~30 h) | $0 |
| Paid hosted open-weight 26–31B (≈ $0.10–0.30 per M input, $0.20–0.60 per M output; assumption, prices vary by provider) | minutes to ~1 h: teams run in parallel, limited only by provider rate limits | **≈ $2.50–7** |
| Self-hosted (vLLM, same weights) | hours on one GPU; scales with GPUs | GPU time only |

Ingest is incremental: daily new meetings only cost their own windows (unchanged meetings are skipped by content hash).

## Three questions it got wrong, and why

WRONG_PLACEHOLDER

## Known limits / next steps

- Embedding search (pgvector) alongside keyword search; a reranker for long teams.
- Show only the registry topics relevant to the window once a team has hundreds of topics (shortcut today: the whole registry, capped per topic).
- Speaker roles are found by the agent at query time (people introduce themselves); a per-meeting role table would make it cheaper.
- If several proposals are open when a window is cut, a later decision may link to the wrong one (no window overlap, by choice: overlap costs tokens and duplicates cards).
