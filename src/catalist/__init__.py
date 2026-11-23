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
    brief_cmd = commands.add_parser(
        "brief", help="write the TS3012 brief into answers.json"
    )
    brief_cmd.add_argument("--model")
    args = parser.parse_args()

    engine = create_engine(os.environ["DATABASE_URL"])
    Base.metadata.create_all(engine)
    llm = chat(engine, args.model)
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
            result = ask(agent, q["team"], q["question"])
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
