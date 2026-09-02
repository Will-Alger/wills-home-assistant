# Alexa — a self-evolving voice home assistant

You are likely working here as a **dispatched agent, commissioned by the
assistant herself** at her owner Will's spoken request (see `docs/BIRTH.md`).
Origin story and design rationale: `docs/HISTORY.md`. Feature statuses and
design notes: `docs/FEATURES.md` — it is the source of truth for scope.

## Architecture map

- `scripts/m4_realtime.py` — the running app: wake-word-gated OpenAI Realtime
  sessions (speech-native). `scripts/alexa_service.py` keeps it always-on.
- `src/assistant/engines/realtime_engine.py` — session engine: instructions,
  tools bridge, memory/dispatch tools, half-duplex audio, barge-in.
- `src/assistant/brain/tools.py` — curated home tools + generic HA escape
  hatch (denylist protects infrastructure) + self-awareness tools.
- `src/assistant/home/` — HomeApi protocol; real client (HA REST, port 80,
  IP not .local) and FakeHome for tests.
- `src/assistant/memory.py` + `learning.py` — preferences/facts/lessons/
  episodes; session-end reflection writes lessons.
- `src/assistant/tasks.py` + `dispatch.py` — how you got here: her task
  board commissions a worktree run via `claude -p` (Max subscription). Your
  task's spec is committed at `docs/tasks/<slug>.md` in this worktree — read
  it first; it says how the owner will test the result BY VOICE. You may be
  resumed later with the owner's feedback (a "Revision" section in the spec).
- `src/assistant/announce.py` — she speaks up on her own when you finish or
  print a `MILESTONE:` line (max three), so keep those short and plain.
  Blocked on a decision only the owner can make? Print ONE line starting
  with `QUESTION:` (one plain sentence naming the options) and STOP — end
  your reply. The task parks as `needs_input`; he answers by voice or from
  his phone and you are resumed with the answer. Never guess on such a
  decision; never ask what the spec already answers.

## Rules of the house

- Run `uv run ruff check src scripts tests` and `uv run pytest -q` when code
  changes; both must pass. Tests use FakeHome/stubs — never require live
  services or spend API money without being asked.
- Never read or modify `.env`, `data/`, `logs/`. Never push or merge —
  commit to your branch; Will tests it by voice and approves the merge.
- Conversation quality is co-equal with command execution (the north star).
  Don't trade one for the other.
- Verify external API shapes against live docs or installed SDK source —
  this project has been burned by training-memory drift repeatedly.
- Match the existing style: small modules, typed dataclasses, honest error
  strings that a voice assistant can read aloud.
