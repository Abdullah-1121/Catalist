import pytest

from catalist.parser import ParseError, parse_meeting

HEADER = """Meeting: TS3012a
Team: TS3012
Meeting number: 1 (a) of the team's four meetings
Speakers: Speaker A, Speaker B, Speaker C, Speaker D
Times: turn start time from the start of the recording (HH:MM:SS)
Source: AMI Meeting Corpus, CC BY 4.0
"""

TURNS = """
00:00:02 Speaker A
Okay, good morning. I'll be your Project Manager. Uh, um, let's start.

00:00:02 Speaker B
Good day.

00:00:04 Speaker D
Morning.
"""


def write(tmp_path, text, name="TS3012a.txt"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_reads_meeting_header(tmp_path):
    meeting = parse_meeting(write(tmp_path, HEADER + TURNS))
    assert meeting.meeting_id == "TS3012a"
    assert meeting.team == "TS3012"
    assert meeting.order == 1


def test_order_comes_from_header_letter_not_file_count(tmp_path):
    # IS1002 has no meeting a: its first file is b, so order must be 2
    text = HEADER.replace("TS3012a", "IS1002b").replace("TS3012", "IS1002")
    text = text.replace("1 (a)", "2 (b)")
    meeting = parse_meeting(write(tmp_path, text + TURNS, "IS1002b.txt"))
    assert meeting.order == 2


def test_reads_turns_with_exact_text(tmp_path):
    turns = parse_meeting(write(tmp_path, HEADER + TURNS)).turns
    assert [t.speaker for t in turns] == ["A", "B", "D"]
    assert turns[0].start == "00:00:02"
    # text kept exactly as spoken, disfluencies included (quotes are checked word for word)
    assert (
        turns[0].text
        == "Okay, good morning. I'll be your Project Manager. Uh, um, let's start."
    )


def test_turn_ids_use_position_so_same_timestamp_stays_unique(tmp_path):
    turns = parse_meeting(write(tmp_path, HEADER + TURNS)).turns
    assert [t.turn_id for t in turns] == ["TS3012a#0", "TS3012a#1", "TS3012a#2"]
    assert turns[1].meeting_id == "TS3012a"
    assert turns[1].index == 1


def test_unknown_line_fails_with_file_and_line_number(tmp_path):
    bad = HEADER + TURNS + "\nthis is not a timestamp line\n"
    path = write(tmp_path, bad)
    with pytest.raises(ParseError, match=r"TS3012a\.txt:17"):
        parse_meeting(path)


def test_turn_without_text_fails(tmp_path):
    with pytest.raises(ParseError, match="no text"):
        parse_meeting(write(tmp_path, HEADER + "\n00:00:02 Speaker A\n"))


def test_empty_file_fails(tmp_path):
    with pytest.raises(ParseError):
        parse_meeting(write(tmp_path, ""))


def test_timestamp_right_after_timestamp_is_not_taken_as_text(tmp_path):
    # A has no text; B's timestamp line must not become A's text
    text = (
        HEADER + "\n00:00:02 Speaker A\n00:00:03 Speaker B\n\n00:00:04 Speaker C\nHi.\n"
    )
    with pytest.raises(ParseError, match=r"TS3012a\.txt:8: turn has no text"):
        parse_meeting(write(tmp_path, text))


def test_repeated_header_key_fails(tmp_path):
    text = HEADER.replace("Team: TS3012\n", "Team: TS3012\nTeam: XX9999\n")
    with pytest.raises(ParseError, match=r"TS3012a\.txt:3: repeated header key 'Team'"):
        parse_meeting(write(tmp_path, text + TURNS))
