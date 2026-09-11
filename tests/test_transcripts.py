"""Live transcript fragments become turns: his words close when she answers,
hers close on silence, and her "mm-hm" under his story is not a turn."""

from __future__ import annotations

from assistant.engines.transcripts import Segmenter


def words(seg: Segmenter, speaker: str, text: str, at_ms: int, step: int = 200) -> list:
    """Word-level fragments on the 200 ms grid GPT-Live uses."""
    out = []
    for i, word in enumerate(text.split()):
        piece = word if i == 0 else " " + word
        out.extend(seg.fragment(speaker, piece, at_ms + i * step, at_ms + (i + 1) * step))  # type: ignore[arg-type]
    return out


def test_his_turn_closes_when_she_answers_and_hers_on_silence() -> None:
    seg = Segmenter()
    updates = words(seg, "user", "turn off the hallway light", 1000)
    assert updates and all(not u.closed for u in updates) and seg.speaker == "user"
    assert updates[-1].text == "turn off the hallway light" and updates[-1].start_ms == 1000
    assert seg.advance(2500) == []  # a pause is not the end of his turn
    reply = words(seg, "assistant", "Done, the hallway is off.", 3400)
    closed = [u for u in reply if u.closed]
    assert closed and closed[0].speaker == "user" and closed[0].reason == "speaker_change"
    assert closed[0].text == "turn off the hallway light"
    assert seg.speaker == "assistant"
    assert seg.advance(5000) == []  # 1.6 s after her last word: still hers
    done = seg.advance(6500)
    assert [u.reason for u in done] == ["inactivity"] and done[0].speaker == "assistant"
    assert done[0].text == "Done, the hallway is off."
    assert seg.speaker is None


def test_her_mm_hm_under_his_story_is_not_a_turn() -> None:
    seg = Segmenter()
    words(seg, "user", "so yesterday I was driving home and", 1000)
    back = seg.fragment("assistant", "mm-hm", 2300, 2600)  # a short acknowledgment while he talks
    assert back == []  # buffered, never emitted as a turn
    more = words(seg, "user", "the highway was closed again", 2800)
    assert seg.speaker == "user" and all(u.speaker == "user" for u in more)
    assert more[-1].text.startswith("so yesterday") and more[-1].text.endswith("closed again")
    assert seg.advance(9000) == []  # his turn stays open until she really answers or the session ends
    ended = seg.close(9500)
    assert [u.speaker for u in ended if u.closed] == ["user"] and ended[-1].reason == "session_closed"


def test_the_session_end_flushes_what_is_open() -> None:
    seg = Segmenter()
    words(seg, "assistant", "One. Two. Three.", 500)
    ended = seg.close(1400)
    assert ended[-1].closed and ended[-1].speaker == "assistant" and ended[-1].text == "One. Two. Three."
    assert seg.close(1500) == []  # nothing left
