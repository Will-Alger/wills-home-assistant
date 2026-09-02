"""Session continuity: the session log, spoken time labels, record_session
with reflection, and the {recent} placeholder."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from assistant.app import record_session
from assistant.engines.realtime_engine import RealtimeEngine, SessionStats
from assistant.home.fake import FakeHome
from assistant.journal import Journal
from assistant.learning import Reflector
from assistant.memory import MemoryStore
from assistant.sessions import SessionLog, when_label
from tests.test_learning import StubReflectLLM

LOCAL = datetime.now().astimezone().tzinfo


class Clock:
    def __init__(self, at: float) -> None:
        self.at = at

    def __call__(self) -> float:
        return self.at


def at(day: int, hour: int, minute: int = 0) -> float:
    return datetime(2026, 9, day, hour, minute, tzinfo=LOCAL).timestamp()


def test_when_labels() -> None:
    now = at(2, 15)
    assert when_label(at(2, 8, 12), now) == "this morning at 8:12 AM"
    assert when_label(at(2, 13, 5), now) == "this afternoon at 1:05 PM"
    assert when_label(at(1, 23, 2), now) == "last night at 11:02 PM"
    assert when_label(at(1, 9, 30), now) == "yesterday at 9:30 AM"
    assert when_label(at(30, 10, 0), now) == "Wednesday at 10:00 AM"  # Sep 30 2026, not today/yesterday


def test_session_log_records_reloads_and_speaks_recent(tmp_path: Path) -> None:
    clock = Clock(at(2, 8, 12))
    log = SessionLog(tmp_path / "sessions.json", keep=3, now=clock)
    row = log.start("wake")
    clock.at += 90
    log.finish(row.id, ended_by="question answered", first_user_line="what's the weather like?", tools=["web_search"])
    assert log.recent_text().startswith("this morning at 8:12 AM: what's the weather like?")
    log.set_summary(row.id, "Checked the weather; rain expected.")
    assert "Checked the weather" in log.recent_text()
    quiet = log.start("announce")
    log.finish(quiet.id, ended_by="announcement delivered")  # nothing said by the owner
    assert [r.id for r in log.recent()] == [row.id]  # announcement-only sessions are not "conversations"
    assert len(log.rows()) == 2 and log.rows()[0]["kind"] == "announce"

    for _ in range(4):
        extra = log.start()
        log.finish(extra.id, ended_by="idle timeout", first_user_line="hi")
    again = SessionLog(tmp_path / "sessions.json", keep=3, now=clock)
    assert len(again.rows(limit=50)) == 3 and again.get(row.id) is None  # trimmed to keep
    assert log.get(999) is None and log.finish(999) is None


async def test_record_session_sets_summary_from_reflection(tmp_path: Path) -> None:
    clock = Clock(at(2, 9))
    sessions = SessionLog(tmp_path / "sessions.json", now=clock)
    journal = Journal(tmp_path / "journal", now=clock)
    memory = MemoryStore(tmp_path / "m.json")
    reflector = Reflector(
        StubReflectLLM({"summary": "Set up a porch light schedule.", "lessons": [], "observations": []}),
        memory,
    )
    row = sessions.start("wake")
    stats = SessionStats(
        ended_by="end_conversation",
        tool_calls=["schedule"],
        transcript=[("you", "turn on the porch light at six every night"), ("alexa", "Done.")],
    )
    reflection = await record_session(sessions, journal, stats, row, reflector)
    assert reflection is not None and reflection.summary.startswith("Set up a porch")
    assert sessions.get(row.id).summary == "Set up a porch light schedule."
    assert sessions.get(row.id).first_user_line.startswith("turn on the porch")
    assert journal.query(kinds=["session"])[0].data["ended_by"] == "end_conversation"

    silent = sessions.start("announce")
    stats = SessionStats(ended_by="announcement delivered", transcript=[("event", "Task 7 is built.")])
    assert await record_session(sessions, journal, stats, silent, reflector) is not None
    assert sessions.get(silent.id).summary == ""  # reflection skips sessions with no user turn
    assert await record_session(None, None, stats, None, None) is None  # everything optional

    engine = RealtimeEngine(
        api_key="k", model="m", voice="v", home=FakeHome(), owner="Will", sessions=sessions
    )
    config = await engine._session_config(None)
    assert "this morning at 9:00 AM: Set up a porch light schedule." in config["instructions"]
    text, is_error = engine._execute_journal_tool("recent_conversations", {"since_hours": 24 * 365 * 5})
    assert not is_error and "porch light" in text  # (the row's clock is a fixed 2026 date)
