"""Local wake-word detection via openWakeWord (ONNX runtime on Windows).

Feed 80 ms frames; `detect()` fires at most once per cooldown and resets the
model's internal audio buffer afterwards so the tail of the wake phrase can't
retrigger. `score()` exposes raw per-frame scores for the spike/monitor tool.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np


class WakeBackoff:
    """False wakes raise the bar for a while. Two in `window_s` (the vacuum,
    a TV, a word that sounded like her name) add `bump` to the threshold for
    `for_s`; a wake somebody actually followed up clears the count. Pure, so
    the rule is testable without a model."""

    def __init__(
        self,
        *,
        strikes: int = 2,
        window_s: float = 180.0,
        bump: float = 0.15,
        for_s: float = 600.0,
        now: Callable[[], float] = time.monotonic,
    ) -> None:
        self._strikes = strikes
        self._window = window_s
        self.bump = bump
        self._for = for_s
        self._now = now
        self._false: list[float] = []
        self._until = 0.0

    def false_wake(self) -> bool:
        """A wake nobody followed up. True when the bar just went up."""
        now = self._now()
        self._false = [t for t in self._false if now - t < self._window] + [now]
        if len(self._false) >= self._strikes:
            self._false.clear()
            self._until = now + self._for
            return True
        return False

    def real_wake(self) -> None:
        self._false.clear()

    @property
    def extra(self) -> float:
        return self.bump if self._now() < self._until else 0.0

    @property
    def seconds_left(self) -> float:
        return max(0.0, self._until - self._now())


class WakeDetector:
    def __init__(
        self,
        model: str = "hey_jarvis",
        threshold: float = 0.5,
        cooldown_s: float = 2.0,
        vad_threshold: float = 0.0,
        backoff: WakeBackoff | None = None,
    ) -> None:
        """`vad_threshold` > 0 hands every frame to openWakeWord's bundled
        Silero VAD first, so a prediction only counts when the frame sounds
        like speech — broadband noise (a vacuum cleaner) cannot fire it."""
        import openwakeword
        from openwakeword.model import Model

        if model.endswith((".onnx", ".tflite")):
            from pathlib import Path

            if not Path(model).exists():
                raise FileNotFoundError(
                    f"Custom wake model not found: {model} — train one via "
                    "docs/custom-wake-word.md and drop it there."
                )
        kwargs = {"wakeword_models": [model], "inference_framework": "onnx", "vad_threshold": vad_threshold}
        try:
            self._model = Model(**kwargs)
        except Exception:  # noqa: BLE001 — whatever failed, a fresh model
            # download is the one self-repair worth trying before giving up.
            # First run: download ONLY the requested model (plus the shared
            # feature models the library always needs) — not the whole zoo.
            names = [] if model.endswith((".onnx", ".tflite")) else [model]
            openwakeword.utils.download_models(model_names=names or ["alexa"])
            self._model = Model(**kwargs)
        self.threshold = threshold
        self.backoff = backoff or WakeBackoff()
        self._cooldown_s = cooldown_s
        self._last_fired = 0.0
        # What the last frame scored — read after detect() by the latency log,
        # which records the wakes that almost happened. Inference is not free:
        # scoring the frame a second time to find out is not an option.
        self.last_score = 0.0

    def score(self, frame: bytes) -> float:
        """Raw max score (0..1) for this frame across the loaded model(s)."""
        samples = np.frombuffer(frame, dtype=np.int16)
        prediction = self._model.predict(samples)
        self.last_score = max(prediction.values()) if prediction else 0.0
        return self.last_score

    @property
    def effective_threshold(self) -> float:
        """The base threshold plus whatever recent false wakes have added."""
        return min(0.95, self.threshold + self.backoff.extra)

    @property
    def has_vad(self) -> bool:
        return getattr(self._model, "vad", None) is not None

    def speech_probability(self, frame: bytes) -> float:
        """Silero's verdict (0..1) on one 16 kHz frame — the same model that
        gates the wake word, fed by hand for the frames the wake path never
        sees (the engine's speech gate, while she is not speaking). Feed each
        frame exactly once, here or through detect(): the model keeps state.
        0.0 when no VAD is loaded (`vad_threshold` 0)."""
        vad = getattr(self._model, "vad", None)
        if vad is None:
            return 0.0
        samples = np.frombuffer(frame, dtype=np.int16)
        return float(vad.predict(samples, frame_size=640))  # two 40 ms windows per 80 ms frame

    def detect(self, frame: bytes) -> bool:
        fired = self.score(frame) >= self.effective_threshold
        if fired and (time.monotonic() - self._last_fired) >= self._cooldown_s:
            self._last_fired = time.monotonic()
            self.reset()
            return True
        return False

    def reset(self) -> None:
        self._model.reset()
