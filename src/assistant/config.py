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

    # Milestone 1: Home Assistant
    ha_url: str = "http://localhost:8123"
    ha_token: str = ""

    # Milestone 2: LLM
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    llm_provider: str = "anthropic"
    llm_model: str = "claude-opus-5"
    # Thinking depth for the command path. "low" keeps routine commands snappy
    # and cheap; raise per-request later via an explicit "think hard" route.
    llm_effort: str = "low"

    # Milestone 3: wake word + STT
    # openWakeWord pretrained phrase (no API key needed); custom model later.
    wake_model: str = "hey_jarvis"
    stt_provider: str = "deepgram"
    deepgram_api_key: str = ""

    # Milestone 4: TTS
    elevenlabs_api_key: str = ""
    elevenlabs_voice_id: str = ""

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
