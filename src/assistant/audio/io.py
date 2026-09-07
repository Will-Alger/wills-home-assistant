"""The runner's one microphone and one speaker, open across idle and talk.

Before this, every cycle opened a 16 kHz mic for the wake word, closed it,
opened a 24 kHz mic and a speaker for the conversation, and closed those —
four stream changes per exchange, a 0.2 s settle between them, a fresh
`sd.play` stream for the wake chime racing the session speaker, and whatever
he said in the half-second after "Alexa" lost to the reopen. Now both
streams open once and stay open: the wake chime goes through the same speaker
her voice does, and speech right after the wake phrase queues on the same
microphone and reaches the session first.

Released PortAudio has no device-change or disconnect signalling, so a
persistent stream needs its own health check: the mic reports a stall when
no frame arrives for a while, the speaker when its callback stops being
called with audio waiting. The runner answers either with `reopen()`.
"""

from __future__ import annotations

from typing import Any

from assistant.audio import devices
from assistant.audio.mic import Microphone
from assistant.audio.speaker import Speaker


class AudioIO:
    def __init__(self, settings: Any, *, rate: int = 24_000, frame_samples: int = 1920) -> None:
        self._settings = settings  # read at every open: the panel changes it live
        self._rate = rate
        self._frame_samples = frame_samples
        self.mic: Microphone | None = None
        self.speaker: Speaker | None = None

    @property
    def is_open(self) -> bool:
        return self.mic is not None and self.mic.is_open and self.speaker is not None and self.speaker.is_open

    @property
    def mic_in_use(self) -> str:
        return self.mic.device_in_use if self.mic is not None else ""

    @property
    def speaker_in_use(self) -> str:
        return self.speaker.device_in_use if self.speaker is not None else ""

    @property
    def fallback(self) -> bool:
        """On a stand-in microphone or speaker: the runner keeps looking for
        the real one (the Echo Dot that dropped off Bluetooth comes back)."""
        return bool(
            (self.mic is not None and self.mic.fallback)
            or (self.speaker is not None and getattr(self.speaker, "fallback", False))
        )

    @property
    def stalled(self) -> bool:
        return bool(self.speaker is not None and self.speaker.stalled)

    def notes(self) -> list[str]:
        """What the streams had to say about the devices they got (fallbacks)."""
        out = []
        if self.mic is not None and self.mic.device_note:
            out.append(self.mic.device_note)
        if self.speaker is not None and self.speaker.device_note:
            out.append(self.speaker.device_note)
        return out

    async def open(self) -> None:
        """Open both on the current settings. Raises when even the default
        will not open — the runner's backoff handles that."""
        await self.close()
        mic = Microphone(
            self._settings.audio_input_device, samplerate=self._rate, frame_samples=self._frame_samples
        )
        speaker = Speaker(self._rate, device=self._settings.audio_output_device)
        await mic.open()
        try:
            await speaker.open()
        except Exception:
            await mic.close()
            raise
        self.mic, self.speaker = mic, speaker

    async def close(self) -> None:
        for stream in (self.mic, self.speaker):
            if stream is not None:
                await stream.close()
        self.mic = self.speaker = None

    async def reopen(self) -> None:
        """Devices changed, or a stream stalled: close, let PortAudio look at
        the machine again, open on whatever the settings say now."""
        await self.close()
        devices.refresh()
        await self.open()
