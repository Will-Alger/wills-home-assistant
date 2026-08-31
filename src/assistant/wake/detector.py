"""Local wake-word detection via openWakeWord (ONNX runtime on Windows).

Feed 80 ms frames; `detect()` fires at most once per cooldown and resets the
model's internal audio buffer afterwards so the tail of the wake phrase can't
retrigger. `score()` exposes raw per-frame scores for the spike/monitor tool.
"""

from __future__ import annotations

import time

import numpy as np


class WakeDetector:
    def __init__(
        self,
        model: str = "hey_jarvis",
        threshold: float = 0.5,
        cooldown_s: float = 2.0,
    ) -> None:
        import openwakeword
        from openwakeword.model import Model

        if model.endswith((".onnx", ".tflite")):
            from pathlib import Path

            if not Path(model).exists():
                raise FileNotFoundError(
                    f"Custom wake model not found: {model} — train one via "
                    "docs/custom-wake-word.md and drop it there."
                )
        try:
            self._model = Model(wakeword_models=[model], inference_framework="onnx")
        except Exception:  # noqa: BLE001 — whatever failed, a fresh model
            # download is the one self-repair worth trying before giving up.
            # First run: download ONLY the requested model (plus the shared
            # feature models the library always needs) — not the whole zoo.
            names = [] if model.endswith((".onnx", ".tflite")) else [model]
            openwakeword.utils.download_models(model_names=names or ["alexa"])
            self._model = Model(wakeword_models=[model], inference_framework="onnx")
        self.threshold = threshold
        self._cooldown_s = cooldown_s
        self._last_fired = 0.0

    def score(self, frame: bytes) -> float:
        """Raw max score (0..1) for this frame across the loaded model(s)."""
        samples = np.frombuffer(frame, dtype=np.int16)
        prediction = self._model.predict(samples)
        return max(prediction.values()) if prediction else 0.0

    def detect(self, frame: bytes) -> bool:
        fired = self.score(frame) >= self.threshold
        if fired and (time.monotonic() - self._last_fired) >= self._cooldown_s:
            self._last_fired = time.monotonic()
            self.reset()
            return True
        return False

    def reset(self) -> None:
        self._model.reset()
