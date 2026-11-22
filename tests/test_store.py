import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from catalist.parser import Meeting, Turn
from catalist.store import MeetingRow, TurnRow, store_meeting


def make_meeting(n_turns, text="Hello there.", meeting_id="TS3012a"):
    turns = [
        Turn(f"{meeting_id}#{i}", meeting_id, i, "00:00:02", "A", text)
        for i in range(n_turns)
    ]
    return Meeting(meeting_id, meeting_id[:-1], 1, turns)


def texts(engine, meeting_id="TS3012a"):
    with Session(engine) as s:
        query = (
            select(TurnRow.text)
            .where(TurnRow.meeting_id == meeting_id)
            .order_by(TurnRow.index)
        )
        return list(s.scalars(query))


def test_stores_meeting_and_turns(engine):
    store_meeting(engine, make_meeting(3))
    with Session(engine) as s:
        m = s.get(MeetingRow, "TS3012a")
        assert (m.team, m.order) == ("TS3012", 1)
        t = s.get(TurnRow, "TS3012a#2")
        assert (t.meeting_id, t.index, t.start, t.speaker, t.text) == (
            "TS3012a",
            2,
            "00:00:02",
            "A",
            "Hello there.",
        )


def test_storing_twice_gives_no_duplicates(engine):
    store_meeting(engine, make_meeting(3))
    store_meeting(engine, make_meeting(3))
    assert len(texts(engine)) == 3


def test_restoring_with_fewer_turns_removes_old_ones(engine):
    # Ex27: the file got fixed and now has fewer turns; #3 and #4 must not linger
    store_meeting(engine, make_meeting(5))
    store_meeting(engine, make_meeting(3))
    with Session(engine) as s:
        assert s.get(TurnRow, "TS3012a#4") is None
    assert len(texts(engine)) == 3


def test_restoring_one_meeting_leaves_other_meetings_alone(engine):
    store_meeting(engine, make_meeting(3, meeting_id="TS3012a"))
    store_meeting(engine, make_meeting(2, meeting_id="TS3012b"))
    store_meeting(engine, make_meeting(1, meeting_id="TS3012a"))
    assert len(texts(engine, "TS3012b")) == 2


def test_failed_store_keeps_the_old_version_complete(engine):
    # Ex27: a crash halfway must roll back everything (transaction), not leave a mix
    store_meeting(engine, make_meeting(3, text="old"))
    broken = make_meeting(3, text="new")
    broken.turns[2] = Turn(
        "TS3012a#2", "TS3012a", 2, "00:00:02", "A", None
    )  # violates NOT NULL
    with pytest.raises(IntegrityError):
        store_meeting(engine, broken)
    assert texts(engine) == ["old", "old", "old"]
