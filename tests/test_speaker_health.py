"""The speaker she is on, asked of Windows: still there, or walked away?"""

from __future__ import annotations

import sys

from assistant.audio.speaker import _keepalive
from assistant.audio.windows_default import SpeakerHealth, endpoint_state, render_endpoints


def test_health_says_the_state_word_once_the_speaker_is_gone_and_only_between_polls() -> None:
    states = {"Speakers (Echo Dot-6FG)": "active"}
    now = [0.0]
    health = SpeakerHealth(state_of=lambda name: states.get(name, ""), every_s=5.0, clock=lambda: now[0])
    assert health.gone("Speakers (Echo Dot-6FG)") == ""  # active
    now[0] += 6
    states["Speakers (Echo Dot-6FG)"] = "unplugged"
    assert health.gone("Speakers (Echo Dot-6FG)") == "unplugged"
    assert health.gone("Speakers (Echo Dot-6FG)") == ""  # polled a moment ago
    now[0] += 6
    assert health.gone("Speakers (Echo Dot-6FG)") == "unplugged"  # still gone: said again at the next poll
    now[0] += 6
    assert health.gone("Some speaker Windows never heard of") == ""  # unknown: never a verdict
    now[0] += 6
    assert health.gone("") == ""


def test_windows_answers_or_stays_quiet() -> None:
    endpoints = render_endpoints()
    assert isinstance(endpoints, list)
    if sys.platform != "win32":
        assert endpoints == [] and endpoint_state("anything") == ""
    else:
        for name, state, endpoint_id in endpoints:
            assert isinstance(name, str) and state and endpoint_id.startswith("{")
        assert endpoint_state("no such speaker anywhere") == ""


def test_the_keepalive_is_inaudible_and_never_short() -> None:
    import numpy as np

    noise = np.frombuffer(_keepalive(96_000), dtype=np.int16)
    assert len(noise) == 48_000 and int(np.abs(noise).max()) <= 1 and int(np.abs(noise).max()) == 1
    assert _keepalive(0) == b"" and len(_keepalive(7)) == 7
