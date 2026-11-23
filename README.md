# Catalist meeting memory

An agent that turns meeting transcripts into queryable memory and briefs a team before its next meeting.

- **Ingest** (LLM + code): each meeting is cut into ~10k-token windows; an open-weight LLM (Gemma 4 via Google AI Studio) extracts *cards*: proposal, decision, action, number, disagreement. Code validates every card, links it to exact transcript turns, and stores it in Postgres. Cards are append-only: a change is a new decision that `supersedes` the old one, so "current" is computed, never overwritten.
- **Query** (agent): an explicit LangGraph loop. The LLM reads the team's cards, then decides to `search` the transcript, `read` turns, or `answer`. Code filters by team, and code (not the LLM) produces the citations: every quote is checked to be at least 6 words copied exactly from one turn.
- **Brief**: the cards of the team's earlier meetings, grouped by the LLM into decisions / what changed / open actions / open disagreements. Code attaches a verified quote to every point.

See `docs/WRITEUP.md` for the architecture, the decisions and where things broke.

## Run it

Requirements: Docker, [uv](https://docs.astral.sh/uv/), a Google AI Studio API key (free tier works).

```bash
cp .env.example .env               # fill in LLM_API_KEY (and check the other values)
docker compose up -d --wait        # Postgres 18 + pgvector on localhost:5432
uv sync

# 1. Ingest: parse, store turns, extract cards (one team at a time, meetings in order)
uv run --env-file .env catalist ingest TS3012 ES2002          # any team ids

# 2. Ask a new question about one team
uv run --env-file .env catalist ask TS3012 "Which features did the team drop, and why?"

# 3. Produce the deliverable: answers.json (all questions) + the TS3012 brief
uv run --env-file .env catalist answer          # or: catalist answer Q01 Q07
uv run --env-file .env catalist brief
```

Re-running is safe and cheap. A meeting whose file has not changed is skipped (content hash), and every LLM call is cached in Postgres, so a repeat call costs nothing. If a meeting changes, it and every later meeting of that team are re-extracted.

`.env` settings: `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL` (any OpenAI-compatible endpoint), `DATABASE_URL`.

## Tests

```bash
uv run pytest        # needs the Postgres container (uses a separate catalist_test database)
uv run ruff check .
```

CI (GitHub Actions) runs ruff and pytest against a Postgres service container on every push.

## Layout

| File | Job |
|---|---|
| `src/catalist/parser.py` | transcript file → meeting + turns (fails loudly on unknown lines) |
| `src/catalist/store.py` | Postgres tables: meetings, turns, cards, rejected_cards, llm_cache |
| `src/catalist/windows.py` | cut a meeting into windows between turns |
| `src/catalist/cards.py` | the `Card` schema + the rules a card must pass on its own |
| `src/catalist/topics.py` | topic-name clean-up |
| `src/catalist/extract.py` | ingest loop: registry → LLM → per-card checks → store |
| `src/catalist/agent.py` | the LangGraph agent and the brief |
| `src/catalist/quotes.py` | exact-quote check and fallback |
| `src/catalist/llm.py` | LLM client, Postgres cache, retry for the free tier |
