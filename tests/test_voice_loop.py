"""VoiceLoop state-machine tests — fake mic, wake, STT, and brain. No audio, no network."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from assistant.audio.base import AudioSourceClosed
from assistant.brain.agent import AgentReply
from assistant.stt.base import SttEvent
from assistant.voice import VoiceLoop

WAKE = b"WAKE"
NOISE = b"...."


class FakeSource:
    def __init__(self, frames: list[bytes]) -> None:
        self._frames = list(frames)
        self._finished = asyncio.Event()
        self.drains = 0

    async def get_frame(self) -> bytes:
        if self._frames:
            await asyncio.sleep(0)
            return self._frames.pop(0)
        await self._finished.wait()
        raise AudioSourceClosed

    def drain(self) -> None:
        self.drains += 1

    def finish(self) -> None:
        self._finished.set()


class FakeWake:
    def __init__(self) -> None:
        self.resets = 0

    def detect(self, frame: bytes) -> bool:
        return frame == WAKE

    def reset(self) -> None:
        self.resets += 1


class FakeStt:
    """Each stream() serves the next scripted list of SttEvents."""

    def __init__(self, scripts: list[list[SttEvent]]) -> None:
        self._scripts = list(scripts)

    @asynccontextmanager
    async def stream(self):
        script = self._scripts.pop(0) if self._scripts else []
        yield _FakeSttStream(script)


class _FakeSttStream:
    def __init__(self, script: list[SttEvent]) -> None:
        self._script = script
        self.audio = b""

    async def send_audio(self, pcm: bytes) -> None:
        self.audio += pcm

    async def events(self):
        for event in self._script:
            await asyncio.sleep(0)
            yield event


class FakeAgent:
    def __init__(self, replies: list[AgentReply]) -> None:
        self._replies = list(replies)
        self.transcripts: list[str] = []

    async def handle(self, text: str) -> AgentReply:
        self.transcripts.append(text)
        return self._replies.pop(0)


class RecordingUi:
    def __init__(self) -> None:
        self.events: list[tuple[str, object]] = []

    def wake(self) -> None:
        self.events.append(("wake", None))

    def partial(self, transcript: str) -> None:
        self.events.append(("partial", transcript))

    def transcript(self, transcript: str) -> None:
        self.events.append(("transcript", transcript))

    def reply(self, reply: AgentReply) -> None:
        self.events.append(("reply", reply.intent))

    def follow_up(self, intent: str) -> None:
        self.events.append(("follow_up", intent))

    def idle(self) -> None:
        self.events.append(("idle", None))

    def error(self, message: str) -> None:
        self.events.append(("error", message))

    def kinds(self) -> list[str]:
        return [kind for kind, _ in self.events]


def make_reply(intent: str) -> AgentReply:
    return AgentReply(speech="ok", intent=intent, hops=1, cost_usd=0.0, latency_ms=1)


async def run_until_idle(loop: VoiceLoop, source: FakeSource, ui: RecordingUi) -> None:
    task = asyncio.create_task(loop.run())
    async with asyncio.timeout(2):
        while "idle" not in ui.kinds():
            await asyncio.sleep(0.01)
    source.finish()
    await task


async def test_one_shot_command() -> None:
    source = FakeSource([NOISE, WAKE, NOISE, NOISE, NOISE])
    stt = FakeStt([[SttEvent("partial", "turn off"), SttEvent("final", "turn off the hallway")]])
    agent = FakeAgent([make_reply("close")])
    ui = RecordingUi()
    loop = VoiceLoop(source, FakeWake(), stt, agent, ui)

    await run_until_idle(loop, source, ui)

    assert agent.transcripts == ["turn off the hallway"]
    kinds = ui.kinds()
    assert kinds[: kinds.index("idle") + 1] == ["wake", "partial", "transcript", "reply", "idle"]
    assert source.drains >= 1  # stale audio flushed before listening


async def test_silence_after_wake_goes_back_to_idle() -> None:
    source = FakeSource([WAKE])
    stt = FakeStt([[]])  # end-of-turn with nothing said
    agent = FakeAgent([])
    ui = RecordingUi()
    loop = VoiceLoop(source, FakeWake(), stt, agent, ui)

    await run_until_idle(loop, source, ui)

    assert agent.transcripts == []
    assert ui.kinds() == ["wake", "idle"]


async def test_listen_intent_reopens_mic_without_wake_word() -> None:
    source = FakeSource([WAKE, NOISE, NOISE])
    stt = FakeStt(
        [
            [SttEvent("final", "should I dim the lights")],
            [SttEvent("final", "yes please")],
        ]
    )
    agent = FakeAgent([make_reply("listen"), make_reply("close")])
    ui = RecordingUi()
    loop = VoiceLoop(source, FakeWake(), stt, agent, ui)

    await run_until_idle(loop, source, ui)

    assert agent.transcripts == ["should I dim the lights", "yes please"]
    assert ui.kinds() == ["wake", "transcript", "reply", "follow_up", "transcript", "reply", "idle"]


async def test_agent_failure_reports_error_and_recovers_to_idle() -> None:
    class ExplodingAgent:
        async def handle(self, text: str) -> AgentReply:
            raise RuntimeError("api down")

    source = FakeSource([WAKE])
    stt = FakeStt([[SttEvent("final", "hello")]])
    ui = RecordingUi()
    loop = VoiceLoop(source, FakeWake(), stt, ExplodingAgent(), ui)

    await run_until_idle(loop, source, ui)

    assert ("error", "api down") in ui.events
    assert ui.kinds()[-1] == "idle"


async def test_speech_started_before_window_expiry_is_never_cut_off() -> None:
    """The listen window only covers waiting for speech; once words arrive,
    end-of-turn detection governs — even past the original window."""
    source = FakeSource([WAKE, NOISE, NOISE, NOISE])

    class SlowTalker(FakeStt):
        @asynccontextmanager
        async def stream(self):
            stream = _FakeSttStream([])

            async def events():
                await asyncio.sleep(0.02)  # speech starts inside the window
                yield SttEvent("partial", "so I was")
                await asyncio.sleep(0.15)  # ...but finishes well past it
                yield SttEvent("final", "so I was thinking about dinner")

            stream.events = events  # type: ignore[method-assign]
            yield stream

    agent = FakeAgent([make_reply("close")])
    ui = RecordingUi()
    loop = VoiceLoop(source, FakeWake(), SlowTalker([]), agent, ui, max_listen_s=0.05)

    await run_until_idle(loop, source, ui)

    assert agent.transcripts == ["so I was thinking about dinner"]


async def test_command_audio_retained_for_voice_id() -> None:
    source = FakeSource([WAKE, b"aa", b"bb", b"cc", b"dd", b"ee", b"ff"])

    class SlowFinalStt(FakeStt):
        @asynccontextmanager
        async def stream(self):
            stream = _FakeSttStream([])

            async def events():
                await asyncio.sleep(0.05)  # let the pump forward some audio
                yield SttEvent("final", "hi")

            stream.events = events  # type: ignore[method-assign]
            yield stream

    agent = FakeAgent([make_reply("close")])
    ui = RecordingUi()
    loop = VoiceLoop(source, FakeWake(), SlowFinalStt([]), agent, ui)

    await run_until_idle(loop, source, ui)

    assert loop.last_command_audio  # frames from the command were kept
