"""Windows' default speaker, asked of Windows: the watch that says when it
moved, and the COM call that never raises."""

from __future__ import annotations

import sys

from assistant.audio.windows_default import DefaultOutputWatch, default_output_id


def test_the_watch_says_once_when_the_default_moves() -> None:
    ids = ["headphones"]
    now = [0.0]
    watch = DefaultOutputWatch(reader=lambda: ids[0], every_s=2.0, clock=lambda: now[0])
    assert not watch.changed()  # nothing opened yet: nothing to compare with
    watch.opened()
    now[0] += 5
    assert not watch.changed()  # same default
    ids[0] = "echo dot"
    assert not watch.changed()  # polled a moment ago: not yet
    now[0] += 5
    assert watch.changed()  # moved
    now[0] += 5
    assert not watch.changed()  # said once
    ids[0] = ""  # Windows has no default speaker at all (unplugged): never a change
    now[0] += 5
    assert not watch.changed()
    watch.opened()  # reopened while there was none…
    ids[0] = "headphones"
    now[0] += 5
    assert not watch.changed()  # …so nothing to compare with until the next open


def test_the_com_call_answers_or_stays_quiet() -> None:
    answer = default_output_id()
    assert isinstance(answer, str)
    if sys.platform != "win32":
        assert answer == ""
    else:
        assert answer == "" or answer.startswith("{")  # an endpoint ID, or no speaker at all
