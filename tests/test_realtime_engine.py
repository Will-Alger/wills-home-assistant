"""Pure-function tests for the realtime engine — no network."""

from __future__ import annotations

import numpy as np

from assistant.engines.realtime_engine import (
    FRAME_SAMPLES_24K,
    downsample_24k_to_16k,
    realtime_tools,
)


def test_tools_convert_to_realtime_shape() -> None:
    tools = realtime_tools()
    names = [tool["name"] for tool in tools]
    assert "set_lights" in names
    assert "get_lights" in names
    assert names[-1] == "end_conversation"
    for tool in tools:
        assert tool["type"] == "function"
        assert "parameters" in tool  # realtime name for the schema
        assert "input_schema" not in tool  # anthropic name must not leak


def test_downsample_produces_wake_sized_frames() -> None:
    frame_24k = np.zeros(FRAME_SAMPLES_24K, dtype=np.int16).tobytes()
    out = downsample_24k_to_16k(frame_24k)
    assert len(out) == 1280 * 2  # exactly one 80ms wake frame at 16 kHz


def test_downsample_preserves_a_tone() -> None:
    t = np.arange(FRAME_SAMPLES_24K) / 24_000
    tone = (10_000 * np.sin(2 * np.pi * 440 * t)).astype(np.int16)
    out = np.frombuffer(downsample_24k_to_16k(tone.tobytes()), dtype=np.int16)
    assert out.astype(np.float32).std() > 1000  # energy survived the resample
