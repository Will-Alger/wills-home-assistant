"""A promise is kept only when it was actually spoken.

"Next time we talk, ask me how the demo went." A session where he asked for
the hallway light and nothing else must leave that promise waiting — she
never said it. The same rule for notifications: listing them is not reading
them out, and a reading he cut in half leaves them unread.

Whole conversations, scripted at real speed (tests/test_replay.py's
instruments), because both rules are decided at the end of playback.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from assistant.announce import Announcer
from assistant.followups import FollowUpStore, mentions, significant_words
from assistant.journal import Journal
from tests.fake_realtime import DelayedSpeaker
from tests.test_replay import WakeAfter, make_engine, owner_says, replay, she_says

DEMO = "ask how the demo went"


def make_store(tmp_path: Path) -> tuple[FollowUpStore, Journal]:
    """One promise, made before this conversation began."""
    journal = Journal(tmp_path / "journal")
    store = FollowUpStore(tmp_path / "f.json", journal=journal)
    store.add(DEMO, trigger="next_conversation")
    return store, journal


def journal_rows(journal: Journal) -> list[str]:
    return [e.text for e in journal.query(kinds=["followup"], limit=20)]


# ── the word test ──────────────────────────────────────────────────────────


def test_two_significant_words_is_the_whole_test() -> None:
    assert significant_words("ask how the demo went") == {"demo", "went"}
    assert not mentions("The hallway light is off.", DEMO)
    assert not mentions("How did that go, by the way?", DEMO)  # nothing in common
    assert not mentions("Your demo files are on the desk.", DEMO)  # one word is chance
    assert mentions("How did the demo go? You wanted me to ask how it went.", DEMO)
    assert mentions("I never asked how the DEMO went — how was it?", DEMO)


# ── 1. a one-line command keeps the promise waiting ────────────────────────


async def test_a_lighting_command_leaves_the_promise_pending(tmp_path: Path) -> None:
    """He woke her for one thing. She never mentioned the demo, so it is
    still hers to bring up — and the journal says why it wasn't."""
    store, journal = make_store(tmp_path)
    engine, conn, _cues = make_engine(followups=store, journal=journal, idle_timeout_s=6.0)

    async def script(speaker, ui) -> None:
        await asyncio.sleep(0.1)
        owner_says(conn, "turn off the hallway light")
        she_says(conn, calls=(("call_0", "set_lights",
                               {"changes": [{"target": "hallway", "turn": "off"}]}),))
        await asyncio.sleep(0.3)
        she_says(conn, "Hallway's off.", ms=300, item_id="item_2",
                 calls=(("call_1", "end_conversation", {}),))

    stats, _took, _speaker, _ui = await replay(engine, script)

    assert stats.ended_by == "end_conversation" and stats.replied
    assert [f.what for f in store.for_conversation()] == [DEMO], "consumed without being said"
    assert journal_rows(journal) == ["follow-up 1 not raised: not mentioned"]


# ── 2. actually asking it advances it, once ────────────────────────────────


async def test_asking_the_question_retires_it_and_only_once(tmp_path: Path) -> None:
    store, journal = make_store(tmp_path)
    engine, conn, _cues = make_engine(followups=store, journal=journal, idle_timeout_s=6.0)

    async def script(speaker, ui) -> None:
        await asyncio.sleep(0.1)
        owner_says(conn, "hey, I'm back")
        she_says(conn, "How did the demo go? You wanted me to ask how it went.", ms=400)
        await asyncio.sleep(0.2)
        owner_says(conn, "it went fine, thanks")
        she_says(conn, "Good.", ms=200, item_id="item_2",
                 calls=(("call_0", "end_conversation", {}),))

    _stats, _took, _speaker, _ui = await replay(engine, script)

    assert store.for_conversation() == []
    (kept,) = [f for f in store._items if f.what == DEMO]
    assert not kept.active and kept.fired
    assert journal_rows(journal) == ["follow-up 1 raised in conversation"]

    # A second conversation has nothing left to raise — and never retires it
    # twice or complains about it.
    engine2, conn2, _c = make_engine(followups=store, journal=journal, idle_timeout_s=6.0)

    async def again(speaker, ui) -> None:
        await asyncio.sleep(0.1)
        owner_says(conn2, "what are you waiting on me for?")
        she_says(conn2, "Nothing at the moment.", ms=200,
                 calls=(("call_0", "end_conversation", {}),))

    await replay(engine2, again)
    assert journal_rows(journal) == ["follow-up 1 raised in conversation"]


# ── 3. she says so herself ─────────────────────────────────────────────────


async def test_raise_follow_up_counts_even_when_she_words_it_her_own_way(tmp_path: Path) -> None:
    """Her words share nothing with the promise; the tool call is the proof."""
    store, journal = make_store(tmp_path)
    engine, conn, _cues = make_engine(followups=store, journal=journal, idle_timeout_s=6.0)

    async def script(speaker, ui) -> None:
        await asyncio.sleep(0.1)
        owner_says(conn, "hey")
        she_says(conn, "So — how did it go yesterday?", ms=400,
                 calls=(("call_0", "raise_follow_up", {"id": 1}),))
        await asyncio.sleep(0.3)
        she_says(conn, "Glad it landed.", ms=200, item_id="item_2",
                 calls=(("call_1", "end_conversation", {}),))

    stats, _took, _speaker, _ui = await replay(engine, script)

    assert "raise_follow_up" in stats.tool_calls
    assert store.for_conversation() == []
    assert journal_rows(journal) == ["follow-up 1 raised in conversation"]

    text, is_error = engine._execute_followup_tool("raise_follow_up", {"id": 1})
    assert is_error and text == "no open follow-up with that id"  # already retired
    store.add("take the chicken out", trigger="arrival")
    text, is_error = engine._execute_followup_tool("raise_follow_up", {"id": 2})
    assert is_error and "not in conversation" in text  # arrival has its own moment


# ── 4. a reading he cut in half ────────────────────────────────────────────


async def test_an_interrupted_reading_of_the_notifications_leaves_them_unread(
    tmp_path: Path,
) -> None:
    """list_notifications hands her the list; he says "Alexa" over the middle
    of her reading it. Nothing he did not hear counts as read."""
    announcer = Announcer(tmp_path / "a.json")
    first = announcer.enqueue("Task 7 is built.", kind="task", ref="task:7:1:built")
    second = announcer.enqueue("The porch light came on at nine.", kind="watch", ref="watch:1")
    announcer.take_due()
    announcer.mark_delivered([first.id, second.id])
    engine, conn, _cues = make_engine(announcer=announcer, idle_timeout_s=6.0, info_close_s=0.4)
    speaker = DelayedSpeaker()
    wake = WakeAfter(speaker, after_ms=500)

    async def script(spk, ui) -> None:
        await asyncio.sleep(0.1)
        owner_says(conn, "what did I miss?")
        she_says(conn, calls=(("call_0", "list_notifications", {"scope": "unread"}),))
        await asyncio.sleep(0.3)
        assert engine._deferred_reads == [("read", [first.id, second.id])]
        assert first.unread and second.unread, "marked read before a word was spoken"
        she_says(conn, "Task 7 is built, and the porch light came on at nine.",
                 ms=1600, item_id="item_2")
        await asyncio.wait_for(ui.barge_in.wait(), 5)
        owner_says(conn, "never mind, tell me later")
        await asyncio.sleep(0.2)
        she_says(conn, "Sure.", ms=200, item_id="item_3")

    stats, _took, speaker, ui = await replay(engine, script, speaker=speaker, wake=wake)

    assert "list_notifications" in stats.tool_calls
    assert first.unread and second.unread, "she was cut off; he never heard the list"
    assert [a.id for a in announcer.unread()] == [first.id, second.id]
    assert "notifications kept unread — she was cut off" in ui.notes
    assert engine._deferred_reads == []  # dropped, not left to leak into the next reply


async def test_a_reading_heard_out_marks_them_read(tmp_path: Path) -> None:
    """The other half of the rule: she got to the end, so they are read."""
    announcer = Announcer(tmp_path / "a.json")
    item = announcer.enqueue("Task 7 is built.", kind="task", ref="task:7:1:built")
    announcer.take_due()
    announcer.mark_delivered([item.id])
    engine, conn, _cues = make_engine(announcer=announcer, idle_timeout_s=6.0)

    async def script(speaker, ui) -> None:
        await asyncio.sleep(0.1)
        owner_says(conn, "what did I miss?")
        she_says(conn, calls=(("call_0", "list_notifications", {"scope": "unread"}),))
        await asyncio.sleep(0.3)
        she_says(conn, "Task 7 is built.", ms=400, item_id="item_2",
                 calls=(("call_1", "end_conversation", {}),))

    _stats, _took, speaker, _ui = await replay(engine, script)

    assert speaker.played_ms("item_2") == 400  # heard out in full
    assert not item.unread and item.read is not None
    assert announcer.unread() == []
