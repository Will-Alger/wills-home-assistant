"""The dashboard: the numbers from the logs, the routes, and one real socket."""

from __future__ import annotations

import http.client
import json

from assistant.dashboard import Dashboard, compute_metrics
from assistant.feedback import FeedbackStore
from assistant.sessions import SessionLog
from assistant.status import AssistantStatus
from assistant.timeline import Timeline


def test_the_numbers_come_out_of_the_three_logs() -> None:
    t = 1_000.0
    timeline = [
        # session 1: he spoke, she answered, then she closed 1 s after his next words — a cut-off
        {"ts": t, "session": 1, "kind": "session", "what": "opened"},
        {"ts": t + 2, "session": 1, "kind": "you_said", "text": "volume up"},
        {"ts": t + 3, "session": 1, "kind": "audio_first"},
        {"ts": t + 8, "session": 1, "kind": "you_said", "text": "also make the living room"},
        {"ts": t + 9, "session": 1, "kind": "session", "what": "closed", "by": "end_conversation"},
        # session 2: a wrap-up close right after his words is his own doing; one silence inside
        {"ts": t + 20, "session": 2, "kind": "session", "what": "opened"},
        {"ts": t + 21, "session": 2, "kind": "you_said", "text": "what's the weather"},
        {"ts": t + 30, "session": 2, "kind": "you_said", "text": "that's all"},
        {"ts": t + 31, "session": 2, "kind": "session", "what": "closed", "by": "wrap-up"},
        {"ts": t + 40, "session": None, "kind": "acknowledged", "seconds": 0.5},
        {"ts": t + 41, "session": None, "kind": "ack_skipped", "reason": "he kept talking"},
        {"ts": t + 42, "session": 3, "kind": "music", "stages_ms": {"native_playing": 2500.0}, "playing_verified": True},
        {"ts": t + 43, "session": 3, "kind": "music", "stages_ms": {"native_fallback": 4000.0, "service_returned": 24000.0}},
        {"ts": t - 90_000, "session": 0, "kind": "you_said", "text": "yesterday"},  # outside the window
    ]
    turns = [{"kind": "wake_miss", "ts": t + 5, "wake_score": 0.4}, {"kind": "wake_miss", "ts": t - 90_000}]
    sessions = [
        {"id": 1, "started": t, "ended": t + 9, "kind": "wake", "cost_usd": 0.02},
        {"id": 2, "started": t + 20, "ended": t + 31, "kind": "wake", "cost_usd": 0.03},
        {"id": 9, "started": t - 90_000, "ended": t - 89_000, "kind": "wake", "cost_usd": 5.0},
    ]
    m = compute_metrics(timeline, turns, sessions, since=t - 60)
    assert m["sessions"] == 2 and m["wakes"] == 2 and m["cost_usd"] == 0.05
    assert m["cut_offs"] == 1 and m["cut_off_sessions"] == [1]
    assert m["silences"] == 1 and m["silence_turns"][0]["text"] == "what's the weather"
    assert m["wake_misses"] == 1
    assert m["acks"] == {"played": 1, "skipped": 1}
    assert m["music"] == {"requests": 2, "median_ms": 13250, "max_ms": 24000, "fallbacks": 1}


def rig(tmp_path):
    status = AssistantStatus(mic="Snowball", voice="marin")
    status.note("boot")
    sessions = SessionLog(tmp_path / "sessions.json", now=lambda: 500.0)
    row = sessions.start("wake")
    sessions.finish(row.id, ended_by="end_conversation", first_user_line="play back in black",
                    tools=["play_music"], cost_usd=0.01, transcript=[("you", "play back in black"), ("alexa", "On it.")],
                    unit="desktop")
    timeline = Timeline(tmp_path / "timeline.jsonl", now=lambda: 501.0)
    timeline.session_started(row.id)
    timeline.event("you_said", text="play back in black")
    timeline.session_ended("end_conversation", cost_usd=0.01)
    (tmp_path / "turns.jsonl").write_text(
        json.dumps({"kind": "turn", "turn": 0, "session": row.id, "speech_end": 3.1, "first_audio": 4.0, "tools": [["play_music", 2.0]]}) + "\n",
        encoding="utf-8",
    )
    feedback = FeedbackStore(tmp_path / "feedback.json")
    promoted: list[str] = []
    dash = Dashboard(
        status=status, sessions_path=tmp_path / "sessions.json", turns_path=tmp_path / "turns.jsonl",
        timeline=timeline, feedback=feedback, unit="desktop", port=0, now=lambda: 600.0,
        promote=lambda need: promoted.append(need.title) or "queued for the board",
    )
    return dash, feedback, row.id, promoted


def test_the_routes_read_the_stores_and_write_feedback(tmp_path) -> None:
    dash, feedback, sid, promoted = rig(tmp_path)
    code, page, kind = dash.handle("GET", "/")
    assert code == 200 and kind.startswith("text/html")
    assert b"Save feedback" in page and b'data-tab="live"' in page and b".hidden{display:none!important}" in page
    code, state, _ = dash.handle("GET", "/api/state")
    assert code == 200 and state["unit"] == "desktop" and state["mic"] == "Snowball" and "log" not in state
    code, listing, _ = dash.handle("GET", "/api/sessions?limit=5")
    assert code == 200 and listing["sessions"][0]["id"] == sid and listing["sessions"][0]["lines"] == 2
    code, one, _ = dash.handle("GET", f"/api/session/{sid}")
    assert code == 200 and one["session"]["transcript"] == [["you", "play back in black"], ["alexa", "On it."]]
    assert [e["kind"] for e in one["events"]] == ["session", "you_said", "session"] and one["turns"][0]["turn"] == 0
    assert dash.handle("GET", "/api/session/999")[0] == 404
    code, item, _ = dash.handle("POST", "/api/feedback", json.dumps({
        "text": "she cut me off", "tags": ["cut-off"], "sessions": [sid],
        "excerpt": "you: play back in black\nalexa: On it.", "range": {"from_ts": 501.0, "to_ts": 501.0, "junk": 1},
    }).encode())
    assert code == 201 and item["status"] == "new" and item["sessions"] == [sid]
    assert item["excerpt"].startswith("you: play") and item["range"] == {"from_ts": 501.0, "to_ts": 501.0}
    code, rows, _ = dash.handle("GET", f"/api/timeline?session={sid}&limit=10")
    assert code == 200 and [r["kind"] for r in rows["rows"]] == ["session", "you_said", "session"]
    assert state.get("last_session") == sid or dash.handle("GET", "/api/state")[1]["last_session"] == sid
    assert dash.handle("POST", "/api/feedback", b'{"text": ""}')[0] == 400
    assert dash.handle("POST", "/api/feedback", b"not json")[0] == 400
    code, listing, _ = dash.handle("GET", "/api/sessions")
    assert listing["sessions"][0]["flags"] == 1
    code, changed, _ = dash.handle("PATCH", f"/api/feedback/{item['id']}", b'{"status": "triaged"}')
    assert code == 200 and changed["status"] == "triaged"
    code, need, _ = dash.handle("POST", "/api/needs", json.dumps({"title": "Never close on his words", "tags": ["cut-off"], "items": [item["id"]]}).encode())
    assert code == 201 and feedback.items()[0].need == need["id"]
    code, done, _ = dash.handle("POST", f"/api/needs/{need['id']}/promote", b"")
    assert code == 202 and done["note"] == "queued for the board" and promoted == ["Never close on his words"]
    assert feedback.needs()[0].status == "planned"
    code, metrics, _ = dash.handle("GET", "/api/metrics?days=1")
    assert code == 200 and metrics["sessions"] == 1 and metrics["open_feedback"] == 1 and metrics["tag_counts"] == {"cut-off": 1}
    code, log, _ = dash.handle("GET", "/api/log")
    assert code == 200 and log["lines"][-1].endswith("boot")


def test_flag_that_by_voice_lands_on_the_current_conversation(tmp_path) -> None:
    from assistant.engines.live_engine import LiveEngine
    from assistant.home.fake import FakeHome

    feedback = FeedbackStore(tmp_path / "feedback.json")
    timeline = Timeline(tmp_path / "timeline.jsonl")
    engine = LiveEngine(api_key="k", model="gpt-live-1", voice="marin", home=FakeHome(), owner="Will",
                        name="Alexa", wake_phrase="alexa", feedback=feedback, timeline=timeline)
    assert "flag_conversation" in [t.get("name") for t in engine._tools()]
    timeline.session_started(41)
    text, is_error = engine._execute_feedback_tool({"text": "she cut me off before I finished", "tags": ["cut-off", "Too Eager"]})
    assert not is_error and "feedback #1" in text and "conversation 41" in text
    item = feedback.items()[0]
    assert item.sessions == [41] and item.tags == ["cut-off", "too-eager"] and item.source == "voice"
    text, is_error = engine._execute_feedback_tool({"text": "that last one was perfect", "previous": True})
    assert not is_error and feedback.items()[0].sessions == [40]
    assert engine._execute_feedback_tool({"text": "  "})[1] is True
    plain = LiveEngine(api_key="k", model="gpt-live-1", voice="marin", home=FakeHome(), owner="Will",
                       name="Alexa", wake_phrase="alexa")
    assert "flag_conversation" not in [t.get("name") for t in plain._tools()]


def test_the_server_answers_on_a_real_socket(tmp_path) -> None:
    dash, _feedback, sid, _p = rig(tmp_path)
    url = dash.start()
    try:
        host, port = url.removeprefix("http://").rstrip("/").split(":")
        conn = http.client.HTTPConnection(host, int(port), timeout=5)
        conn.request("GET", f"/api/session/{sid}")
        resp = conn.getresponse()
        assert resp.status == 200
        assert json.loads(resp.read())["session"]["id"] == sid
        conn.request("POST", "/api/feedback", body=json.dumps({"text": "too slow", "tags": ["too-slow"]}), headers={"Content-Type": "application/json"})
        assert conn.getresponse().status == 201
        conn.request("GET", "/")
        page = conn.getresponse()
        assert page.status == 200 and page.getheader("Content-Type", "").startswith("text/html")
        page.read()
    finally:
        dash.stop()
