"""The phone channel: Home Assistant companion-app push notifications with
buttons, and what happens when the owner taps one.

Verified against the companion app docs (2026-09-02): `notify.mobile_app_*`
takes {title, message, data: {tag, group, actions: [{action, title,
behavior: "textInput", ...}], push: {"interruption-level": ...}}}; sending
`message: "clear_notification"` with the same tag removes the card; a tap
fires the HA event `mobile_app_notification_action` with `action` and, for
text input, `reply_text`. Every card's tag is `alexa-<notification id>` so
read/resolve can clear it and a tap names the exact row.

Buttons are chosen by the producer at enqueue time (`actions`): a built task
gets Approve & merge / Later, a question gets Answer (text input), a
confirm-first action gets Yes / No, everything else Got it.
"""

from __future__ import annotations

import asyncio
import contextlib
import re
import time
from collections.abc import Callable
from typing import Any

_ACTION_RE = re.compile(r"^alexa:([a-z_]+):(\d+)$")
_BUTTONS: dict[str, dict[str, Any]] = {
    "approve": {"title": "Approve & merge"},
    "later": {"title": "Later"},
    "read": {"title": "Got it"},
    "answer": {
        "title": "Answer",
        "behavior": "textInput",
        "textInputButtonTitle": "Send",
        "textInputPlaceholder": "Your answer",
    },
    "yes": {"title": "Yes"},
    "no": {"title": "No", "destructive": True},
}
_LEVELS = {"normal": "active", "urgent": "time-sensitive"}
_LABELS = {
    "task": "Task", "milestone": "Task progress", "question": "Question", "watch": "House",
    "thought": "Thought", "system": "System", "followup": "Follow-up", "nudge": "Reminder",
    "reminder": "Reminder", "action": "Scheduled", "presence": "Home",
}


def action_id(verb: str, notification_id: int) -> str:
    return f"alexa:{verb}:{int(notification_id)}"


def parse_action(text: str) -> tuple[str, int] | None:
    m = _ACTION_RE.match(str(text or "").strip())
    return (m.group(1), int(m.group(2))) if m else None


class PhonePusher:
    def __init__(
        self,
        home: Any,
        service: str,
        *,
        name: str = "Alexa",
        journal: Any | None = None,
        now: Callable[[], float] = time.time,
    ) -> None:
        self._home = home
        self._service = service.removeprefix("notify.")
        self._name = name
        self._journal = journal
        self._now = now

    @property
    def service(self) -> str:
        return self._service

    def payload(self, item: Any, *, level: str | None = None) -> dict[str, Any]:
        verbs = [v for v in (getattr(item, "actions", None) or ["read"]) if v in _BUTTONS]
        actions = [{"action": action_id(v, item.id), **_BUTTONS[v]} for v in verbs]
        label = _LABELS.get(getattr(item, "kind", ""), str(getattr(item, "kind", "")).title() or "Note")
        return {
            "title": f"{self._name}: {label}",
            "message": str(item.text)[:500],
            "data": {
                "tag": f"alexa-{item.id}",
                "group": "alexa",
                "actions": actions,
                "push": {"interruption-level": level or _LEVELS.get(getattr(item, "priority", "normal"), "active")},
            },
        }

    async def push(self, item: Any, *, level: str | None = None) -> None:
        await self._home.generic_call("notify", self._service, self.payload(item, level=level))
        if self._journal is not None:
            with contextlib.suppress(Exception):
                self._journal.write(
                    "push", f"to the phone: {str(item.text)[:120]}", source=getattr(item, "kind", ""),
                    data={"id": item.id, "level": level or _LEVELS.get(getattr(item, "priority", "normal"))},
                )

    async def clear(self, item: Any) -> None:
        await self._home.generic_call(
            "notify", self._service, {"message": "clear_notification", "data": {"tag": f"alexa-{item.id}"}}
        )

    def clear_later(self, item: Any) -> None:
        """Schedule a clear from synchronous code (an Announcer subscriber)."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        task = loop.create_task(self.clear(item))
        task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)

    async def say(self, text: str, *, tag: str = "alexa-reply") -> None:
        """A plain follow-up card ("Task 7 merged — restarting"), no buttons."""
        await self._home.generic_call(
            "notify",
            self._service,
            {"title": self._name, "message": str(text)[:500], "data": {"tag": tag, "group": "alexa"}},
        )


class PhoneActions:
    """Turn a tap on a card into the same thing a spoken command would do."""

    def __init__(
        self,
        announcer: Any,
        *,
        board: Any | None = None,
        scheduler: Any | None = None,
        pusher: PhonePusher | None = None,
        journal: Any | None = None,
        log: Callable[[str], None] | None = None,
        request_restart: Callable[[], None] | None = None,
    ) -> None:
        self._announcer = announcer
        self._board = board
        self._scheduler = scheduler
        self._pusher = pusher
        self._journal = journal
        self._log = log or (lambda _m: None)
        self._request_restart = request_restart
        self._running: set[asyncio.Task] = set()

    # ── entry from the websocket ───────────────────────────────────────────

    def handle_event(self, data: dict[str, Any]) -> None:
        parsed = parse_action(str(data.get("action", "")))
        if parsed is None:
            return
        verb, nid = parsed
        reply = str(data.get("reply_text", "") or "")
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        task = loop.create_task(self._run(verb, nid, reply))
        self._running.add(task)
        task.add_done_callback(self._running.discard)

    @property
    def busy(self) -> bool:
        return any(not t.done() for t in self._running)

    async def drain(self, timeout_s: float = 120.0) -> None:
        pending = [t for t in self._running if not t.done()]
        if pending:
            with contextlib.suppress(Exception):
                await asyncio.wait(pending, timeout=timeout_s)

    # ── the work ───────────────────────────────────────────────────────────

    async def _say(self, text: str) -> None:
        if self._pusher is not None:
            with contextlib.suppress(Exception):
                await self._pusher.say(text)

    async def _clear(self, item: Any) -> None:
        if self._pusher is not None:
            with contextlib.suppress(Exception):
                await self._pusher.clear(item)

    def _journal_tap(self, verb: str, item: Any, outcome: str) -> None:
        if self._journal is not None:
            with contextlib.suppress(Exception):
                self._journal.write(
                    "push", f"tap {verb}: {outcome}", source=getattr(item, "kind", "phone"),
                    data={"id": getattr(item, "id", 0), "verb": verb},
                )

    async def _run(self, verb: str, nid: int, reply: str) -> None:
        item = self._announcer.get(nid)
        if item is None:
            await self._say("I no longer have that notification.")
            return
        self._log(f"phone: {verb} on notification {nid}")
        try:
            if verb in ("read", "later"):
                if verb == "read":
                    self._announcer.mark_read([nid])
                await self._clear(item)
                self._journal_tap(verb, item, "cleared" if verb == "later" else "read")
                return
            if not item.live:
                await self._say("That one's already been handled.")
                return
            if verb == "approve":
                await self._approve(item)
            elif verb == "answer":
                await self._answer(item, reply)
            elif verb in ("yes", "no"):
                await self._confirm(item, verb == "yes")
            else:
                await self._say(f"I don't know what to do with '{verb}'.")
        except Exception as err:  # noqa: BLE001 — the phone must hear about failures
            self._log(f"phone action {verb} failed: {err}")
            self._journal_tap(verb, item, f"failed: {str(err)[:120]}")
            await self._say(f"That didn't work: {str(err) or type(err).__name__}"[:300])

    async def _approve(self, item: Any) -> None:
        task_id = int((item.context or {}).get("task_id", 0) or 0)
        if not task_id or self._board is None:
            await self._say("That card isn't tied to a task I can merge.")
            return
        result = await self._board.approve(task_id)
        task = self._board.get(task_id)
        if task.state != "merged":
            self._journal_tap("approve", item, f"blocked: {result[:120]}")
            await self._say(result[:300])
            return
        self._announcer.resolve(f"task:{task_id}:")
        await self._clear(item)
        self._journal_tap("approve", item, f"task {task_id} merged")
        await self._say(f"Task {task_id} merged — restarting now.")
        with contextlib.suppress(Exception):
            self._announcer.enqueue(
                f"Task {task_id}, '{task.title}', was merged from your phone — I'm restarting onto it.",
                kind="system",
                ref=f"task:{task_id}:phone-merged",
                priority="urgent",
                expires_in_s=1800,
            )
        if getattr(self._board, "restart_requested", False):
            self._board.restart_requested = False
        if self._request_restart is not None:
            self._request_restart()

    async def _answer(self, item: Any, reply: str) -> None:
        task_id = int((item.context or {}).get("task_id", 0) or 0)
        if not task_id or self._board is None:
            await self._say("That card isn't tied to a task.")
            return
        if not reply.strip():
            await self._say("Your answer came through empty — tap Answer again and type it.")
            return
        await self._board.answer(task_id, reply)
        self._announcer.mark_read([item.id])
        await self._clear(item)
        self._journal_tap("answer", item, f"task {task_id}: {reply[:80]}")
        await self._say(f"Sent to task {task_id}'s agent; it's continuing.")

    async def _confirm(self, item: Any, yes: bool) -> None:
        confirm = getattr(self._scheduler, "confirm", None)
        schedule_id = int((item.context or {}).get("schedule_id", 0) or 0)
        if confirm is None or not schedule_id:
            await self._say("That card isn't tied to a scheduled action.")
            return
        result = await confirm(schedule_id, yes)
        self._announcer.resolve_ids([item.id])
        await self._clear(item)
        self._journal_tap("yes" if yes else "no", item, str(result)[:120])
        await self._say(str(result)[:300])
