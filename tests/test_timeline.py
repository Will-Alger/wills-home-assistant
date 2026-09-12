"""The always-on timeline: every tap, with its session and unit, readable back."""

from __future__ import annotations

import json

from assistant.timeline import Timeline


def test_rows_carry_the_session_the_unit_and_the_session_clock(tmp_path) -> None:
    clock = [10.0]
    line = Timeline(tmp_path / "timeline.jsonl", unit="desktop", now=lambda: 1000.0, clock=lambda: clock[0])
    line.event("boot", version="x")  # outside any session
    line.session_started(7, "wake")
    clock[0] = 12.5
    line.event("you_said", text="play back in black")
    line.session_ended("end_conversation", cost_usd=0.0123, replied=True)
    rows = [json.loads(s) for s in (tmp_path / "timeline.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["kind"] for r in rows] == ["boot", "session", "you_said", "session"]
    assert rows[0]["session"] is None and rows[0]["t"] is None
    assert rows[2] == {"ts": 1000.0, "t": 2.5, "session": 7, "unit": "desktop", "kind": "you_said", "text": "play back in black"}
    assert rows[3]["by"] == "end_conversation" and rows[3]["cost_usd"] == 0.0123
    assert line.session is None
    assert [r["kind"] for r in line.rows(session=7)] == ["session", "you_said", "session"]
    assert line.rows(kinds={"you_said"})[0]["text"] == "play back in black"


def test_the_tap_fans_out_and_a_broken_sink_never_matters(tmp_path) -> None:
    line = Timeline(tmp_path / "t.jsonl")
    seen: list[tuple[str, dict]] = []

    def recorder(kind, **fields):
        seen.append((kind, fields))

    def broken(kind, **fields):
        raise RuntimeError("no")

    tap = line.tap(recorder, broken, None)
    tap("tool", name="set_lights", seconds=0.4)
    assert seen == [("tool", {"name": "set_lights", "seconds": 0.4})]
    assert line.rows()[-1]["name"] == "set_lights"


def test_a_torn_last_line_is_skipped_and_size_rotates(tmp_path) -> None:
    path = tmp_path / "t.jsonl"
    line = Timeline(path, keep_bytes=200)
    for i in range(30):
        line.event("x", i=i)
    line._written = 2_000_000  # force the size check
    line.event("y")
    assert path.with_suffix(".1.jsonl").exists() or path.stat().st_size <= 400
    with path.open("a", encoding="utf-8") as fh:
        fh.write('{"kind": "torn", "ts": 1')
    kinds = [r["kind"] for r in line.rows()]
    assert "torn" not in kinds and "y" in kinds
