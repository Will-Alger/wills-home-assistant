"""Central configuration, loaded from environment variables and `.env`.

Every secret and tunable lives here — nothing is hardcoded elsewhere. Fields
default to empty strings so that milestone-1 code runs without milestone-4
keys; call `require()` at an entry point to fail fast with a clear message
when a needed value is missing.
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


def code_root() -> Path:
    """The checkout this code was imported from (main, or a staged worktree)."""
    return Path(__file__).resolve().parents[2]


def home_dir() -> Path:
    """Where .env, data/ and logs/ live: the MAIN repo, even when the code
    runs from a staged worktree (the watchdog sets ALEXA_HOME)."""
    override = os.environ.get("ALEXA_HOME", "").strip()
    return Path(override).resolve() if override else code_root()


# Anchor .env to the home dir so `uv run` works from any directory and a
# staged worktree still reads the real secrets. Real environment variables
# always take precedence over the file.
_REPO_ROOT = home_dir()


def wake_phrase(model: str) -> str:
    """Spoken form of a wake-word model name or path: models/hey_gary.onnx
    -> "hey gary". Also used by the settings panel, which offers models the
    running Settings object has never seen."""
    stem = model.replace("\\", "/").rsplit("/", 1)[-1]
    for suffix in (".onnx", ".tflite"):
        stem = stem.removesuffix(suffix)
    return stem.replace("_", " ").removesuffix(" v0.1").strip()


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
    # Optional aliases -> MA playback entity and its TV power entity. Empty
    # automatically pairs only a house with exactly one music player and one TV.
    music_destinations: dict[str, dict[str, str]] = {}
    # The Apple TV plays Apple Music itself: its Music app opened by link, a
    # remote press, the title confirmed from its state (~2 s warm). Needs the
    # TV's remote entity; Music Assistant stays the fallback and the route for
    # library playlists.
    music_native: bool = True
    apple_storefront: str = "us"  # the music.apple.com storefront the links open
    music_native_ready_s: float = 0.9  # page opened → first key press (twice that after a wake)

    # The dashboard: sessions, one conversation's timeline, feedback and needs,
    # the live log and the day's numbers, served on localhost by the runner.
    # `unit_name` stamps every session and timeline row (the pucks come later).
    dashboard_enabled: bool = True
    dashboard_port: int = 8765
    unit_name: str = "desktop"
    # A tray icon (bottom right): her state as a colour, a click opens the
    # dashboard, the menu has the Settings panel, Restart and Quit.
    tray_icon: bool = True

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
    # A build is killed after this long; a build that dies without a result
    # (the Max usage window) is resumed once after this delay.
    dispatch_timeout_s: float = 3600.0
    dispatch_resume_delay_s: float = 300.0
    # Her deeper reasoning (the `think` tool) runs on this Claude model via the
    # CLI (Max-billed); the voice model stays in charge of the conversation.
    brain_model: str = "opus"
    brain_effort: str = ""
    # Her journal (what she did and saw, one small file per day) is pruned
    # after this many days.
    journal_keep_days: int = 90
    # Presence: the Home Assistant person entity that says whether the owner
    # is home (the companion app's GPS), e.g. person.owner. Empty = untracked
    # (she then assumes he is home). Debounce and settle grace in seconds.
    presence_entity: str = ""
    presence_arrive_s: float = 60.0
    presence_leave_s: float = 300.0
    presence_settle_s: float = 90.0
    # While he is away, normal notifications of these kinds go to his phone
    # (urgent ones always do); the rest wait for the arrival welcome.
    push_while_away_kinds: str = "task,question,watch,thought,system,followup"
    # The phone: a Home Assistant companion-app notify service name WITHOUT the
    # domain, e.g. mobile_app_my_phone. Empty = no phone channel.
    phone_notify_service: str = ""
    # Delivery polish: spoken-but-unread items go to the phone (silently)
    # after this many hours; unread items expire after this many days; a
    # calendar meeting in progress holds normal news.
    escalate_after_h: float = 4.0
    unread_expire_days: float = 3.0
    nudge_after_days: float = 2.0
    focus_from_calendar: bool = True
    # Announcements (a build finished, a milestone, a rollback): she speaks up
    # on her own while idle. Normal ones wait out quiet hours ("23:00-08:00";
    # "" = never quiet); urgent ones don't. Each is retried with backoff up to
    # this many times before it is dropped with a log line.
    announce_quiet_hours: str = "23:00-08:00"
    announce_max_attempts: int = 4
    # Path to uv.exe for syncing a staged branch's dependencies; "" = look on
    # PATH, then the WinGet install location.
    uv_exe: str = ""
    # Web search (OpenAI Responses API web_search tool, same OPENAI_API_KEY):
    # the model that reads the results, and how much context it pulls.
    web_search_model: str = "gpt-4.1-mini"  # 4s; gpt-5-mini took ~29s
    web_search_context: str = "low"

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
    # openWakeWord's bundled Silero VAD gate: a prediction only counts when the
    # frame sounds like speech, so a vacuum cleaner cannot wake her. 0 = off.
    wake_vad_threshold: float = 0.5
    audio_input_device: str = ""  # "" = default mic; index or name substring
    audio_output_device: str = ""  # "" = default speaker; index or name substring
    stt_provider: str = "deepgram"
    deepgram_api_key: str = ""
    # 0 = provider default. Raise toward ~0.85 if it ends your turn at
    # thoughtful pauses; it will wait for higher confidence you're done.
    stt_eot_threshold: float = 0.0

    # Milestone 4: voice engines
    # Primary candidate: OpenAI Realtime (speech-native; uses OPENAI_API_KEY).
    realtime_model: str = "gpt-realtime-2.1-mini"  # fast; real thinking goes to the brain (think)
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
    realtime_eagerness: str = "auto"  # high cut sentences off; auto = medium
    # True talk-over (interrupt by just speaking). ONLY with headphones — on
    # open speakers the mic hears the assistant and it interrupts itself.
    realtime_talk_over: bool = False
    # Loudspeaker talk-over, tentatively: while she speaks, a sustained rise of
    # the mic above her own echo pauses playback and streams the mic to the
    # server; its speech detection confirms (she stops, your turn) or, within
    # 1.5 s, nothing does and she resumes where she paused. The wake phrase
    # still interrupts instantly. Turn off if the room makes her stall.
    realtime_tentative_interrupt: bool = True
    # Server-side noise reduction before VAD and the model: far_field for a
    # desk microphone at speaking distance, near_field for a headset, "" off.
    # Off by default: it went live 2026-09-05 and the mid-sentence cut-offs
    # were logged after it — an unvalidated trial, not a default.
    realtime_noise_reduction: str = ""
    # Who decides a turn has ended. semantic_vad waits while a sentence sounds
    # unfinished and (at auto/medium) does not chunk aggressively — the "high"
    # setting was what cut him off mid-sentence. server_vad ends a turn on
    # SILENCE only, after REALTIME_SILENCE_MS: predictable, but any pause to
    # think ends the turn. Either way the engine's own guards drop a reply to a
    # fragment he kept adding to.
    realtime_turn_detection: str = "semantic_vad"
    realtime_silence_ms: int = 1000
    # Send the server speech or clean silence, never the room: a local gate on
    # the microphone (with a short pre-roll) keeps a vacuum cleaner, the fridge
    # and her own tail from ever becoming a "turn".
    realtime_speech_gate: bool = True
    # The language the transcriber is told to expect (ISO-639-1). Unpinned,
    # a one-word turn came back as "Oh ja." and "Tamam,". "" = let it guess.
    realtime_transcribe_language: str = "en"

    # Which engine runs a conversation. "live" is GPT-Live (gpt-live-1): full
    # duplex — she listens while she speaks and stops when you talk over her —
    # with reasoning and tools delegated to a backend model; billed $0.05 a
    # minute of open session, by the second. "realtime" is the turn-based
    # Realtime engine above. The wake word gates both.
    voice_engine: str = "realtime"
    live_model: str = "gpt-live-1"
    # "" = REALTIME_VOICE mapped onto the Live voice list (sol → marin). Live
    # has 22 built-in voices and no sol; marin is the voice her cue clips use.
    live_voice: str = ""
    # The backend that reasons and runs the tools when she delegates. luna is
    # 10× cheaper than terra; switch if tool choice or answers get worse.
    live_backend_model: str = "gpt-5.6-luna"
    # none|minimal|low|medium|high|xhigh. The backend's native web search
    # refuses none and minimal ("cannot be used with reasoning.effort
    # 'minimal': web_search" — every tool request failed that way on day one),
    # so with web search on the engine raises either to low.
    live_backend_reasoning: str = "low"
    live_backend_web_search: bool = True  # the Responses backend's native web search instead of ours
    # How her own voice at the microphone is handled. auto: learn the coupling
    # during her first reply and go full duplex when her echo is small
    # (headphones, wired speakers) or feed silence while she plays when the
    # speaker is louder at the mic than you are (the Echo Dot) — the wake
    # word cuts in either way. duplex|gated force one.
    live_echo_policy: str = "auto"
    live_duplex_max_coupling: float = 0.4
    # An open Live session bills per second, so idle time is money: shorter
    # than the realtime idle timeout, and a hard cap on any one session.
    live_idle_timeout_s: float = 30.0
    live_max_session_s: float = 600.0
    # The local Silero gate while she is silent (speech or clean silence to
    # the server, never the room). Off while she talks in duplex mode.
    live_speech_gate: bool = True
    # A socket connected while she is idle (not billed until the wake starts
    # it): the model answers its own name in about a second, in the tone it
    # was said, instead of a clip. Replaced every live_warm_max_age_s.
    live_warm_socket: bool = True
    live_warm_max_age_s: float = 240.0
    live_store: bool = False  # keep the recording on OpenAI's side for 30 days (forking); off

    # What the wake word (and a push-to-talk press) is answered with, before
    # the session exists: "voice" plays one of the short clips rendered in her
    # own voice under assets/voice/ack (scripts/render_acks.py) — "Yes?",
    # "Go ahead.", "Morning." — "ding" is the rising chime, "off" is silence.
    # A missing clip falls back to the ding with a boot note.
    wake_ack: str = "voice"
    # The beat before she answers her name: the clip is ready 30 ms after the
    # wake, which sounds eager; a person takes about a third of a second (a
    # little different each time). 0 = at once.
    wake_ack_beat_s: float = 0.45  # the beat before "Yes?" — on GPT-Live, the window in which his own next words cancel it

    # Push to talk: hold this system-wide hotkey and speak; letting go ends
    # the turn (turn detection is off while it is held, so a pause can never
    # cut him off). Modifier-only by default — Ctrl+Alt on its own means
    # nothing to the app he is typing in, and we never swallow the keys.
    # "" (or "off") = wake word only.
    ptt_hotkey: str = "ctrl+alt"

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
        return wake_phrase(self.wake_model)

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
