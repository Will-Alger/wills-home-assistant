"""Central configuration, loaded from environment variables and `.env`.

Every secret and tunable lives here — nothing is hardcoded elsewhere. Fields
default to empty strings so that milestone-1 code runs without milestone-4
keys; call `require()` at an entry point to fail fast with a clear message
when a needed value is missing.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Anchor .env to the repo root so `uv run` works from any directory.
# Real environment variables always take precedence over the file.
_REPO_ROOT = Path(__file__).resolve().parents[2]


class MissingSettingError(RuntimeError):
    pass


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Who the assistant primarily serves (used in its instructions). Anyone
    # can talk to it — this just names the household owner.
    owner_name: str = "Will"
    # The assistant's persona name (what it calls itself).
    assistant_name: str = "Alexa"
    # Free-text style/preference lines appended to the assistant's
    # instructions — your personal tuning knob ("Be extremely brief." etc.).
    assistant_extra_instructions: str = ""

    # Milestone 1: Home Assistant
    ha_url: str = "http://localhost:8123"
    ha_token: str = ""

    # Background Claude jobs (reflection, later dispatch) run through the
    # Claude Code CLI = billed to the Max SUBSCRIPTION, not API tokens.
    # Set false to fall back to the API key below.
    use_claude_subscription: bool = True
    # Cloud dispatch (preferred when set): a claude.ai/code "routine" for this
    # repo — dispatched jobs become LIVE cloud sessions you can open in the
    # web/desktop app/phone. Create at claude.ai/code/routines (API trigger),
    # then paste the routine id (trig_...) and its bearer token here.
    claude_routine_id: str = ""
    claude_routine_token: str = ""
    # Coding agents she commissions: which Claude model (alias) and effort.
    dispatch_model: str = "opus"
    dispatch_effort: str = ""
    # Announcements (a build finished, a milestone, a rollback): she speaks up
    # on her own while idle. Normal ones wait out quiet hours ("23:00-08:00";
    # "" = never quiet); urgent ones don't. Each is retried with backoff up to
    # this many times before it is dropped with a log line.
    announce_quiet_hours: str = "23:00-08:00"
    announce_max_attempts: int = 4

    # Milestone 2: LLM
    anthropic_api_key: str = ""
    # Required for identity-linked/multi-workspace keys (wrkspc_... id from
    # the Console); harmless to leave empty for single-workspace keys.
    anthropic_workspace_id: str = ""
    openai_api_key: str = ""
    llm_provider: str = "anthropic"
    llm_model: str = "claude-opus-5"
    # Thinking depth for the command path. "low" keeps routine commands snappy
    # and cheap; raise per-request later via an explicit "think hard" route.
    llm_effort: str = "low"

    # Milestone 3: wake word + STT
    # A pretrained openWakeWord name ("hey_jarvis") OR a path to a custom
    # trained model, e.g. models/hey_gary.onnx — see docs/custom-wake-word.md
    wake_model: str = "alexa"
    wake_threshold: float = 0.5  # raise if false wakes, lower if it misses you
    audio_input_device: str = ""  # "" = default mic; index or name substring
    stt_provider: str = "deepgram"
    deepgram_api_key: str = ""
    # 0 = provider default. Raise toward ~0.85 if it ends your turn at
    # thoughtful pauses; it will wait for higher confidence you're done.
    stt_eot_threshold: float = 0.0

    # Milestone 4: voice engines
    # Primary candidate: OpenAI Realtime (speech-native; uses OPENAI_API_KEY).
    realtime_model: str = "gpt-realtime-2.1"
    # "sol" is Will's pick; it's org-gated today, so the engine automatically
    # falls back to marin until OpenAI unlocks it — then this just works.
    realtime_voice: str = "sol"
    # A silent open session auto-closes after this and returns to wake-word
    # idle (open sessions bill by the minute). Raise it if conversations feel
    # cut short during quiet moments.
    realtime_idle_timeout_s: float = 45.0
    # One-shot commands ("set the volume to 75%"): the engine closes the
    # session this many seconds after the spoken confirmation if the speaker
    # stays silent — no "that's all" needed. Speaking again cancels it.
    realtime_command_close_s: float = 8.0
    # Same idea for a single answered question ("what's the weather?"):
    # longer window so follow-ups feel natural, still far snappier than idle.
    realtime_info_close_s: float = 15.0
    # How fast semantic VAD decides you're done talking: low|medium|high|auto.
    # high = snappy replies; drop toward auto/low if it cuts off your pauses.
    realtime_eagerness: str = "high"
    # True talk-over (interrupt by just speaking). ONLY with headphones — on
    # open speakers the mic hears the assistant and it interrupts itself.
    realtime_talk_over: bool = False

    # Milestone 6: Apple Calendar over iCloud CalDAV. The password MUST be an
    # app-specific one (appleid.apple.com); the Apple ID password is rejected.
    # Both empty = the calendar tools stay hidden from the assistant.
    icloud_username: str = ""
    icloud_app_password: str = ""
    icloud_caldav_url: str = "https://caldav.icloud.com"
    # Which calendar to read/write by default. "" = the account's first one.
    icloud_calendar_name: str = ""

    # Alternate engine: ElevenLabs streaming TTS pipeline (custom voice, parked)
    elevenlabs_api_key: str = ""
    elevenlabs_voice_id: str = ""

    @property
    def wake_phrase(self) -> str:
        """Spoken form of the wake word, derived from the model name/path."""
        stem = self.wake_model.replace("\\", "/").rsplit("/", 1)[-1]
        for suffix in (".onnx", ".tflite"):
            stem = stem.removesuffix(suffix)
        return stem.replace("_", " ").removesuffix(" v0.1").strip()

    def require(self, *field_names: str) -> None:
        """Raise with a setup hint if any of the named settings are unset."""
        missing = [name for name in field_names if not getattr(self, name)]
        if missing:
            raise MissingSettingError(
                f"Missing settings: {', '.join(m.upper() for m in missing)}. "
                "Copy .env.example to .env and fill them in (see README)."
            )


def load_settings() -> Settings:
    return Settings()
