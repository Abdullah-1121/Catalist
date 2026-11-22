"""Parse one AMI transcript file into a Meeting and its Turns. Plain code, no AI."""

import re
from dataclasses import dataclass
from pathlib import Path

TURN_START = re.compile(r"(\d\d:\d\d:\d\d) Speaker ([A-Z])")
ORDER_LETTER = re.compile(r"\(([a-z])\)")  # "Meeting number: 2 (b) of ..." -> "b"


class ParseError(ValueError):
    pass


@dataclass(frozen=True)
class Turn:
    turn_id: str  # "TS3012a#9": position, not time (13% of turns share a timestamp)
    meeting_id: str
    index: int
    start: str  # "HH:MM:SS", fixed width so it sorts as text
    speaker: str  # letter only; roles are inferred later by the LLM
    text: str  # exactly as in the file: quotes are checked word for word


@dataclass(frozen=True)
class Meeting:
    meeting_id: str
    team: str
    order: int  # a=1 .. d=4, from the header letter (IS1002 has no "a")
    turns: list[Turn]


def parse_meeting(path: Path) -> Meeting:
    lines = path.read_text(encoding="utf-8").splitlines()

    def fail(lineno: int, msg: str):
        raise ParseError(f"{path}:{lineno}: {msg}")

    # Header: "Key: value" lines up to the first blank line.
    header = {}
    for lineno, line in enumerate(lines, 1):
        if not line:
            break
        key, sep, value = line.partition(": ")
        if not sep:
            fail(lineno, f"bad header line {line!r}")
        if key in header:
            fail(lineno, f"repeated header key {key!r}")
        header[key] = value
    if not {"Meeting", "Team", "Meeting number"} <= header.keys():
        fail(1, "header must have Meeting, Team and Meeting number")
    letter = ORDER_LETTER.search(header["Meeting number"])
    if not letter:
        fail(3, "no meeting letter like '(a)' in Meeting number")
    meeting_id = header["Meeting"]

    # Turns: a "HH:MM:SS Speaker X" line, then exactly one text line.
    turns = []
    pending = None  # (lineno, start, speaker) of a turn still waiting for its text
    first_turn_line = len(header) + 2  # exact: header keys are unique (checked above)
    for lineno, line in enumerate(lines[first_turn_line - 1 :], first_turn_line):
        if pending:
            if not line.strip() or TURN_START.fullmatch(line):
                fail(pending[0], "turn has no text")
            _, start, speaker = pending
            i = len(turns)
            turns.append(Turn(f"{meeting_id}#{i}", meeting_id, i, start, speaker, line))
            pending = None
        elif m := TURN_START.fullmatch(line):
            pending = (lineno, *m.groups())
        elif line:
            fail(lineno, f"expected 'HH:MM:SS Speaker X', got {line!r}")
    if pending:
        fail(pending[0], "turn has no text")

    order = ord(letter.group(1)) - ord("a") + 1
    return Meeting(meeting_id, header["Team"], order, turns)
