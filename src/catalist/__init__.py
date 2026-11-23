import argparse
import json
import os
from pathlib import Path

from sqlalchemy import create_engine

TRANSCRIPTS = Path("Catalist_AI_Intern_Build_Test_Data/transcripts")
QUESTIONS = Path("Catalist_AI_Intern_Build_Test_Data/questions.json")
ANSWERS = Path("answers.json")


def main() -> None:
    from catalist.extract import ingest_team
    from catalist.llm import chat
    from catalist.store import Base

    parser = argparse.ArgumentParser(prog="catalist")
    commands = parser.add_subparsers(dest="command", required=True)
    ingest = commands.add_parser("ingest", help="extract cards for whole teams")
    ingest.add_argument("teams", nargs="+", help="e.g. TS3012 ES2002")
    ingest.add_argument("--model", help="overrides LLM_MODEL from .env")
    answer = commands.add_parser(
        "answer", help="answer questions.json into answers.json"
    )
    answer.add_argument(
        "ids", nargs="*", help="only these, e.g. Q01 Q07 (default: all)"
    )
    answer.add_argument("--model")
    ask_one = commands.add_parser("ask", help="ask one new question")
    ask_one.add_argument("team")
    ask_one.add_argument("question")
    ask_one.add_argument("--model")
    eval_cmd = commands.add_parser(
        "eval", help="score the agent on eval/questions.json"
    )
    eval_cmd.add_argument("--model")
    check_cmd = commands.add_parser(
        "check", help="verify every citation in answers.json"
    )
    check_cmd.add_argument("--model")
    brief_cmd = commands.add_parser(
        "brief", help="write the TS3012 brief into answers.json"
    )
    brief_cmd.add_argument("--model")
    args = parser.parse_args()

    engine = create_engine(os.environ["DATABASE_URL"])
    Base.metadata.create_all(engine)
    llm = chat(engine, args.model)
    if args.command in ("eval", "check"):
        return run_eval(args, engine, llm)
    if args.command != "ingest":
        return run_query(args, engine, llm)
    for team in args.teams:
        try:
            ingest_team(engine, llm, sorted(TRANSCRIPTS.glob(f"{team}?.txt")))
        except Exception as error:  # noqa: BLE001 -- one broken team must not stop the others (D12)
            print(f"{team}: FAILED {type(error).__name__}: {error}")


def run_query(args, engine, llm) -> None:
    from catalist.agent import ask, brief, build

    if args.command == "ask":
        print(json.dumps(ask(build(engine, llm), args.team, args.question), indent=2))
        return
    data = json.loads(QUESTIONS.read_text())
    out = (
        json.loads(ANSWERS.read_text())
        if ANSWERS.exists()
        else {"answers": [], "brief": ""}
    )
    if args.command == "brief":
        team = data["brief_task"]["team"]
        final = max(int(p.stem[-1], 36) - 9 for p in TRANSCRIPTS.glob(f"{team}?.txt"))
        out["brief"] = brief(
            engine, llm, team, before_order=final
        )  # final meeting excluded
        print(out["brief"])
    else:
        agent = build(engine, llm)
        done = {a["question_id"]: a for a in out["answers"]}
        for q in data["questions"]:
            if args.ids and q["question_id"] not in args.ids:
                continue
            try:
                result = ask(agent, q["team"], q["question"])
            except Exception as error:  # noqa: BLE001 -- one bad reply must not stop the other questions
                result = {
                    "answer": "",
                    "citations": [],
                    "error": f"{type(error).__name__}: {error}"[:300],
                }
            done[q["question_id"]] = {"question_id": q["question_id"], **result}
            out["answers"] = sorted(done.values(), key=lambda a: a["question_id"])
            ANSWERS.write_text(
                json.dumps(out, indent=2)
            )  # after every question: a crash keeps the rest
            print(
                f"{q['question_id']}: {result['answer']} ({len(result['citations'])} citations)",
                flush=True,
            )
    ANSWERS.write_text(json.dumps(out, indent=2))


def run_eval(args, engine, llm) -> None:
    import re
    import time

    from catalist.agent import ask, build, valid_citations

    if args.command == "check":
        for a in json.loads(ANSWERS.read_text())["answers"]:
            ok = valid_citations(engine, a["citations"])
            print(f"{a['question_id']}: {ok}/{len(a['citations'])} citations valid")
        return
    agent, rows = build(engine, llm), []
    for q in json.loads(Path("eval/questions.json").read_text()):
        start = time.time()
        try:
            result = ask(agent, q["team"], q["question"])
        except Exception as error:  # noqa: BLE001 -- score it as a miss, keep going
            result = {"answer": f"ERROR {error}"[:200], "citations": []}
        text = result["answer"].lower()
        # word-prefix match: "fall" also matches "falls"
        facts = sum(
            any(re.search(rf"\b{re.escape(k)}", text) for k in group)
            for group in q["must"]
        )
        rows.append(
            {
                **q,
                **result,
                "facts_found": f"{facts}/{len(q['must'])}",
                "citations_valid": f"{valid_citations(engine, result['citations'])}/{len(result['citations'])}",
                "seconds": round(time.time() - start),
            }
        )
        print(
            f"{q['id']}: facts {rows[-1]['facts_found']}, citations {rows[-1]['citations_valid']}, "
            f"{rows[-1]['seconds']}s | {result['answer']}",
            flush=True,
        )
        Path("eval/results.json").write_text(json.dumps(rows, indent=2))
