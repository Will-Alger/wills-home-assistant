"""The voice session state machine (M3: ears + brain; M4 adds the mouth).

IDLE --wake phrase--> LISTENING --end of turn--> THINKING --reply-->
  intent "close"        -> IDLE
  intent "listen"/"confirm_close" -> LISTENING again (no wake word), with a
  shorter follow-up window; silence closes the conversation.

Design notes honored (docs/FEATURES.md): single-flight (frames captured while
the brain thinks are drained, never processed); the last command's audio is
retained for the future voice-ID milestone; every stage failure is caught,
reported, and returns the loop to IDLE — the loop itself never dies.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Protocol

from assistant.audio.base import AudioSource, AudioSourceClosed
from assistant.brain.agent import Agent, AgentReply
from assistant.stt.base import SttProvider
from assistant.wake.detector import WakeDetector


class VoiceUi(Protocol):
    """Presentation hooks — a console in M3, spoken audio in M4."""

    def wake(self) -> None: ...
    def partial(self, transcript: str) -> None: ...
    def transcript(self, transcript: str) -> None: ...
    def reply(self, reply: AgentReply) -> None: ...
    def follow_up(self, intent: str) -> None: ...
    def idle(self) -> None: ...
    def error(self, message: str) -> None: ...


class VoiceLoop:
    def __init__(
        self,
        source: AudioSource,
        wake: WakeDetector,
        stt: SttProvider,
        agent: Agent,
        ui: VoiceUi,
        *,
        max_listen_s: float = 15.0,
        follow_up_s: float = 10.0,
        max_turn_s: float = 60.0,
    ) -> None:
        self._source = source
        self._wake = wake
        self._stt = stt
        self._agent = agent
        self._ui = ui
        # The listen timeouts only cover WAITING for speech to start; once
        # words arrive, the STT's end-of-turn detection decides when you're
        # done, with max_turn_s as a failsafe ceiling. (Earlier design cut
        # speakers off mid-sentence when the window elapsed — Will vetoed.)
        self._max_listen_s = max_listen_s
        self._follow_up_s = follow_up_s
        self._max_turn_s = max_turn_s
        self.last_command_audio = b""  # retained for the voice-ID milestone

    async def run(self) -> None:
        while True:
            try:
                await self._wait_for_wake()
            except AudioSourceClosed:
                return
            try:
                self._ui.wake()
                await self._conversation()
            except AudioSourceClosed:
                return
            except Exception as err:  # noqa: BLE001 — failure contract: report, go IDLE
                self._ui.error(str(err))
            self._wake.reset()
            self._ui.idle()

    async def _wait_for_wake(self) -> None:
        while True:
            frame = await self._source.get_frame()
            if self._wake.detect(frame):
                return

    async def _conversation(self) -> None:
        first_turn = True
        while True:
            timeout = self._max_listen_s if first_turn else self._follow_up_s
            transcript = await self._listen_turn(timeout)
            if transcript is None:
                return  # silence/timeout — conversation over
            self._ui.transcript(transcript)
            reply = await self._agent.handle(transcript)
            self._ui.reply(reply)
            if reply.intent == "close":
                return
            self._ui.follow_up(reply.intent)
            first_turn = False

    async def _listen_turn(self, timeout_s: float) -> str | None:
        self._source.drain()  # only fresh audio; nothing captured while thinking
        frames: list[bytes] = []
        final: str | None = None
        pump_task: asyncio.Task | None = None
        try:
            async with asyncio.timeout(timeout_s) as deadline:
                async with self._stt.stream() as stream:

                    async def pump() -> None:
                        while True:
                            frame = await self._source.get_frame()
                            frames.append(frame)
                            await stream.send_audio(frame)

                    pump_task = asyncio.create_task(pump())
                    speaking = False
                    try:
                        async for event in stream.events():
                            if not speaking:
                                # Speech started: stop the clock; end-of-turn
                                # detection takes over (failsafe cap only).
                                speaking = True
                                deadline.reschedule(
                                    asyncio.get_running_loop().time() + self._max_turn_s
                                )
                            if event.kind == "partial":
                                self._ui.partial(event.transcript)
                            else:
                                final = event.transcript
                    finally:
                        pump_task.cancel()
                        with contextlib.suppress(asyncio.CancelledError):
                            await pump_task
        except TimeoutError:
            pass
        if pump_task is not None and not pump_task.cancelled() and pump_task.done():
            exc = pump_task.exception()
            if exc is not None:
                raise exc
        self.last_command_audio = b"".join(frames)
        return final
