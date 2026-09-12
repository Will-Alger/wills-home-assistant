"""The dashboard: what she did, what it felt like, and the numbers.

One page on localhost, served by the running assistant from a daemon thread
(the standard library's HTTP server; nothing to install, nothing that could
not run on a second unit). It reads the stores the app already keeps —
session rows, the always-on timeline, the turn log, the status feed — and
writes only feedback and needs. Everything it computes is a pure function of
those files, so a question about last week is a filter, not a memory.

The five numbers (docs/MEASURABLE-2026-09-12.md): cut-offs, silences, wake
misses, acknowledgments, music start — plus sessions and cost.
"""

from __future__ import annotations

import contextlib
import json
import statistics
import threading
import time
from collections.abc import Callable
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

CUT_OFF_S = 2.0  # a session that ends this soon after his last words, unless he ended it
SILENCE_S = 3.0  # his turn with no voice from her within this long
_ENDED_BY_HIM = ("stop command", "wrap-up", "nobody spoke", "nothing to announce")
_MUSIC_STAGES = ("native_playing", "service_returned")


# ── the numbers ──────────────────────────────────────────────────────────────


def compute_metrics(
    timeline: list[dict[str, Any]],
    turns: list[dict[str, Any]],
    sessions: list[dict[str, Any]],
    *,
    since: float,
) -> dict[str, Any]:
    """The day's (or week's) numbers from the three logs. Pure."""
    rows = [r for r in timeline if float(r.get("ts") or 0) >= since]
    by_session: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        sid = row.get("session")
        if isinstance(sid, int):
            by_session.setdefault(sid, []).append(row)
    cut_offs: list[int] = []
    silences: list[dict[str, Any]] = []
    for sid, events in by_session.items():
        closed = next((e for e in events if e.get("kind") == "session" and e.get("what") == "closed"), None)
        said = [e for e in events if e.get("kind") == "you_said"]
        if (closed and said and closed.get("by") not in _ENDED_BY_HIM
                and float(closed["ts"]) - float(said[-1]["ts"]) <= CUT_OFF_S):
            cut_offs.append(sid)
        for i, turn in enumerate(said):
            start = float(turn["ts"])
            until = float(said[i + 1]["ts"]) if i + 1 < len(said) else float("inf")
            replied = any(
                e.get("kind") in ("audio_first", "alexa_said", "fast_started")
                and start - 1.0 <= float(e["ts"]) <= min(start + SILENCE_S, until)
                for e in events
            )
            if not replied and (closed is None or float(closed["ts"]) - start > SILENCE_S):
                silences.append({"session": sid, "text": str(turn.get("text", ""))[:80]})
    acks = {"played": 0, "skipped": 0}
    for row in rows:
        if row.get("kind") == "acknowledged" and float(row.get("seconds") or 0) > 0:
            acks["played"] += 1
        elif row.get("kind") == "ack_skipped":
            acks["skipped"] += 1
    music_ms: list[float] = []
    music_fallbacks = 0
    for row in rows:
        if row.get("kind") != "music":
            continue
        stages = row.get("stages_ms") or {}
        if "native_fallback" in stages:
            music_fallbacks += 1
        for stage in _MUSIC_STAGES:
            if stage in stages:
                music_ms.append(float(stages[stage]))
                break
    misses = sum(1 for t in turns if t.get("kind") == "wake_miss" and float(t.get("ts") or 0) >= since)
    window_sessions = [s for s in sessions if float(s.get("started") or 0) >= since and s.get("ended")]
    wakes = sum(1 for s in window_sessions if s.get("kind") == "wake")
    cost = sum(float(s.get("cost_usd") or 0) for s in window_sessions)
    return {
        "since": since,
        "sessions": len(window_sessions),
        "wakes": wakes,
        "cut_offs": len(cut_offs),
        "cut_off_sessions": cut_offs[-20:],
        "silences": len(silences),
        "silence_turns": silences[-20:],
        "wake_misses": misses,
        "acks": acks,
        "music": {
            "requests": len(music_ms),
            "median_ms": round(statistics.median(music_ms)) if music_ms else None,
            "max_ms": round(max(music_ms)) if music_ms else None,
            "fallbacks": music_fallbacks,
        },
        "cost_usd": round(cost, 4),
    }


# ── the server ───────────────────────────────────────────────────────────────


class Dashboard:
    def __init__(
        self,
        *,
        status: Any,
        sessions_path: Path,
        turns_path: Path,
        timeline: Any,
        feedback: Any,
        unit: str = "desktop",
        host: str = "127.0.0.1",
        port: int = 8765,
        now: Callable[[], float] = time.time,
        promote: Callable[[Any], str] | None = None,
    ) -> None:
        self._status = status
        self._sessions_path = Path(sessions_path)
        self._turns_path = Path(turns_path)
        self._timeline = timeline
        self._feedback = feedback
        self._unit = unit
        self._host, self._port = host, int(port)
        self._now = now
        self._promote = promote
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        port = self._server.server_address[1] if self._server is not None else self._port
        return f"http://{self._host}:{port}/"

    def start(self) -> str:
        dashboard = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args: Any) -> None:  # the console is hers
                pass

            def _run(self, method: str) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(length) if length else b""
                code, payload, content_type = dashboard.handle(method, self.path, body)
                data = payload if isinstance(payload, bytes) else json.dumps(payload, default=str).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self) -> None:  # the server's own names
                self._run("GET")

            def do_POST(self) -> None:
                self._run("POST")

            def do_PATCH(self) -> None:
                self._run("PATCH")

        self._server = ThreadingHTTPServer((self._host, self._port), Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, name="dashboard", daemon=True)
        self._thread.start()
        return self.url

    def stop(self) -> None:
        if self._server is not None:
            with contextlib.suppress(Exception):
                self._server.shutdown()
                self._server.server_close()
            self._server = None

    # ── routing ────────────────────────────────────────────────────────────

    def handle(self, method: str, raw_path: str, body: bytes = b"") -> tuple[int, Any, str]:
        """(status, payload, content type). Testable without a socket."""
        parts = urlsplit(raw_path)
        path = parts.path.rstrip("/") or "/"
        query = {k: v[-1] for k, v in parse_qs(parts.query).items()}
        try:
            data = json.loads(body.decode("utf-8")) if body else {}
        except (json.JSONDecodeError, UnicodeDecodeError):
            return HTTPStatus.BAD_REQUEST, {"error": "the body is not JSON"}, "application/json"
        try:
            if method == "GET":
                return self._get(path, query)
            if method == "POST":
                return self._post(path, data)
            if method == "PATCH":
                return self._patch(path, data)
        except (KeyError, ValueError, TypeError) as err:
            return HTTPStatus.BAD_REQUEST, {"error": str(err)}, "application/json"
        return HTTPStatus.METHOD_NOT_ALLOWED, {"error": "no"}, "application/json"

    def _get(self, path: str, query: dict[str, str]) -> tuple[int, Any, str]:
        json_t = "application/json"
        if path == "/":
            return HTTPStatus.OK, _PAGE.encode("utf-8"), "text/html; charset=utf-8"
        if path == "/api/state":
            return HTTPStatus.OK, self.state(), json_t
        if path == "/api/log":
            return HTTPStatus.OK, {"lines": self.log(int(query.get("limit", 300)))}, json_t
        if path == "/api/timeline":
            rows = self._timeline.rows(limit=int(query.get("limit", 300)))
            return HTTPStatus.OK, {"rows": rows}, json_t
        if path == "/api/sessions":
            return HTTPStatus.OK, {"sessions": self.sessions(int(query.get("limit", 80)))}, json_t
        if path.startswith("/api/session/"):
            found = self.session(int(path.rsplit("/", 1)[1]))
            return (HTTPStatus.OK, found, json_t) if found else (HTTPStatus.NOT_FOUND, {"error": "no such session"}, json_t)
        if path == "/api/feedback":
            return HTTPStatus.OK, self._feedback.snapshot(), json_t
        if path == "/api/metrics":
            days = float(query.get("days", 7))
            return HTTPStatus.OK, self.metrics(days), json_t
        return HTTPStatus.NOT_FOUND, {"error": "not here"}, json_t

    def _post(self, path: str, data: dict[str, Any]) -> tuple[int, Any, str]:
        json_t = "application/json"
        if path == "/api/feedback":
            item = self._feedback.add(
                str(data.get("text", "")), tags=data.get("tags") or [], sessions=data.get("sessions") or [],
                source=str(data.get("source") or "dashboard"), unit=self._unit,
            )
            if item is None:
                return HTTPStatus.BAD_REQUEST, {"error": "say what felt wrong"}, json_t
            return HTTPStatus.CREATED, _asdict(item), json_t
        if path == "/api/needs":
            need = self._feedback.add_need(
                str(data.get("title", "")), tags=data.get("tags") or [], items=data.get("items") or [],
                notes=str(data.get("notes") or ""),
            )
            if need is None:
                return HTTPStatus.BAD_REQUEST, {"error": "a need has a title"}, json_t
            return HTTPStatus.CREATED, _asdict(need), json_t
        if path.startswith("/api/needs/") and path.endswith("/promote"):
            need_id = int(path.split("/")[3])
            need = next((n for n in self._feedback.needs() if n.id == need_id), None)
            if need is None:
                return HTTPStatus.NOT_FOUND, {"error": "no such need"}, json_t
            if self._promote is None:
                return HTTPStatus.NOT_IMPLEMENTED, {"error": "no task board to promote to"}, json_t
            note = self._promote(need)
            self._feedback.update_need(need_id, status="planned")
            return HTTPStatus.ACCEPTED, {"note": note}, json_t
        return HTTPStatus.NOT_FOUND, {"error": "not here"}, json_t

    def _patch(self, path: str, data: dict[str, Any]) -> tuple[int, Any, str]:
        json_t = "application/json"
        if path.startswith("/api/feedback/"):
            item = self._feedback.update(int(path.rsplit("/", 1)[1]), **data)
            return (HTTPStatus.OK, _asdict(item), json_t) if item else (HTTPStatus.NOT_FOUND, {"error": "no such item"}, json_t)
        if path.startswith("/api/needs/"):
            need = self._feedback.update_need(int(path.rsplit("/", 1)[1]), **data)
            return (HTTPStatus.OK, _asdict(need), json_t) if need else (HTTPStatus.NOT_FOUND, {"error": "no such need"}, json_t)
        return HTTPStatus.NOT_FOUND, {"error": "not here"}, json_t

    # ── the data ───────────────────────────────────────────────────────────

    def state(self) -> dict[str, Any]:
        snap = self._status.snapshot(log_lines=0) if self._status is not None else {}
        snap.pop("log", None)
        snap["unit"] = self._unit
        snap["now"] = self._now()
        snap["current_session"] = getattr(self._timeline, "session", None)
        return snap

    def log(self, limit: int = 300) -> list[str]:
        return list(self._status.lines(limit)) if self._status is not None else []

    def _session_rows(self) -> list[dict[str, Any]]:
        try:
            data = json.loads(self._sessions_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        return [r for r in data.get("sessions", []) if isinstance(r, dict) and "id" in r]

    def _turn_rows(self) -> list[dict[str, Any]]:
        try:
            lines = self._turns_path.read_text(encoding="utf-8").splitlines()[-4000:]
        except OSError:
            return []
        out = []
        for line in lines:
            with contextlib.suppress(json.JSONDecodeError):
                row = json.loads(line)
                if isinstance(row, dict):
                    out.append(row)
        return out

    def sessions(self, limit: int = 80) -> list[dict[str, Any]]:
        flagged: dict[int, int] = {}
        for item in self._feedback.items():
            for sid in item.sessions:
                flagged[sid] = flagged.get(sid, 0) + 1
        rows = sorted(self._session_rows(), key=lambda r: float(r.get("started") or 0), reverse=True)[:limit]
        out = []
        for row in rows:
            out.append({
                "id": row["id"], "started": row.get("started"), "ended": row.get("ended"), "kind": row.get("kind"),
                "ended_by": row.get("ended_by", ""), "first_user_line": row.get("first_user_line", ""),
                "summary": row.get("summary", ""), "tools": row.get("tools", []),
                "cost_usd": row.get("cost_usd", 0.0), "responses": row.get("responses", 0),
                "unit": row.get("unit", ""), "lines": len(row.get("transcript") or []),
                "flags": flagged.get(int(row["id"]), 0),
            })
        return out

    def session(self, sid: int) -> dict[str, Any] | None:
        row = next((r for r in self._session_rows() if int(r["id"]) == int(sid)), None)
        if row is None:
            return None
        events = self._timeline.rows(session=sid, limit=3000)
        turns = [t for t in self._turn_rows() if t.get("session") == int(sid) and t.get("kind") == "turn"]
        return {
            "session": row,
            "events": events,
            "turns": turns,
            "feedback": [_asdict(i) for i in self._feedback.items(session=int(sid))],
        }

    def metrics(self, days: float = 7) -> dict[str, Any]:
        since = self._now() - float(days) * 86400
        result = compute_metrics(
            self._timeline.rows(since=since, limit=100_000), self._turn_rows(), self._session_rows(), since=since,
        )
        result["days"] = days
        result["open_feedback"] = len([i for i in self._feedback.items() if i.status not in ("fixed", "wontfix")])
        result["tag_counts"] = self._feedback.tag_counts()
        return result


def _asdict(obj: Any) -> dict[str, Any]:
    from dataclasses import asdict

    return asdict(obj)


# ── the page ─────────────────────────────────────────────────────────────────

_PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Alexa</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root{--bg:#f6f4ef;--fg:#1d1b17;--mute:#6b665c;--line:#dcd7cc;--you:#1b4f8a;--her:#6a3d9a;--ev:#8a6d1b;--bad:#b23a2f;--ok:#2f7d4a;--card:#fffdf8}
@media (prefers-color-scheme:dark){:root{--bg:#161512;--fg:#ece8df;--mute:#a39c8e;--line:#3a362f;--you:#8fb8ea;--her:#c9a7ea;--ev:#e0c36f;--bad:#f08a7e;--ok:#7fd39a;--card:#1f1d19}}
*{box-sizing:border-box}body{margin:0;font:14px/1.45 system-ui,Segoe UI,sans-serif;background:var(--bg);color:var(--fg)}
header{display:flex;flex-wrap:wrap;gap:12px;align-items:center;padding:10px 16px;border-bottom:1px solid var(--line);background:var(--card);position:sticky;top:0;z-index:2}
header h1{font-size:16px;margin:0 8px 0 0}.dot{display:inline-block;width:10px;height:10px;border-radius:50%;background:var(--line);margin-right:6px}.dot.on{background:var(--ok)}
.chips{display:flex;flex-wrap:wrap;gap:6px}.chip{padding:3px 8px;border:1px solid var(--line);border-radius:12px;background:var(--bg);font-size:12px;white-space:nowrap}.chip b{font-variant-numeric:tabular-nums}
.chip.bad b{color:var(--bad)}nav{display:flex;gap:4px;padding:8px 16px 0}nav button{background:none;border:1px solid var(--line);border-bottom:none;border-radius:8px 8px 0 0;padding:6px 12px;color:var(--fg);cursor:pointer}
nav button.on{background:var(--card);font-weight:600}main{padding:12px 16px 40px}.hidden{display:none}
.split{display:grid;grid-template-columns:minmax(260px,340px) 1fr;gap:16px}@media(max-width:820px){.split{grid-template-columns:1fr}}
.list{border:1px solid var(--line);border-radius:8px;background:var(--card);max-height:75vh;overflow:auto}.row{padding:8px 10px;border-bottom:1px solid var(--line);cursor:pointer}.row:hover,.row.on{background:var(--bg)}
.row .top{display:flex;justify-content:space-between;gap:8px;font-size:12px;color:var(--mute)}.row .line{margin-top:2px;overflow-wrap:anywhere}
.badge{font-size:11px;padding:1px 6px;border-radius:10px;border:1px solid var(--line)}.badge.bad{color:var(--bad);border-color:var(--bad)}.badge.flag{color:var(--ev);border-color:var(--ev)}
.card{border:1px solid var(--line);border-radius:8px;background:var(--card);padding:12px;margin-bottom:12px}
.t{margin:4px 0;overflow-wrap:anywhere;white-space:pre-wrap}.t.you{color:var(--you)}.t.alexa{color:var(--her)}.t.event{color:var(--ev)}.t .who{font-weight:600;margin-right:6px}
.ev{font:12px/1.4 ui-monospace,Consolas,monospace;color:var(--mute);overflow-wrap:anywhere;white-space:pre-wrap;margin:1px 0}.ev.warn{color:var(--bad)}
textarea{width:100%;min-height:70px;font:inherit;padding:8px;border:1px solid var(--line);border-radius:6px;background:var(--bg);color:var(--fg)}
.tags{display:flex;flex-wrap:wrap;gap:6px;margin:8px 0}.tag{padding:3px 9px;border:1px solid var(--line);border-radius:12px;cursor:pointer;font-size:12px;user-select:none}.tag.on{background:var(--you);color:#fff;border-color:var(--you)}
button.go{background:var(--you);color:#fff;border:none;border-radius:6px;padding:7px 14px;cursor:pointer;font:inherit}button.go:disabled{opacity:.5}
table{width:100%;border-collapse:collapse}td,th{padding:6px 8px;border-bottom:1px solid var(--line);vertical-align:top;text-align:left;overflow-wrap:anywhere}th{font-size:12px;color:var(--mute)}
select,input[type=text]{font:inherit;padding:4px 6px;border:1px solid var(--line);border-radius:6px;background:var(--bg);color:var(--fg)}
.log{font:13px/1.5 ui-monospace,Consolas,monospace;white-space:pre-wrap;overflow-wrap:anywhere}.log div{padding:1px 0;border-bottom:1px dotted var(--line)}
.muted{color:var(--mute)}.small{font-size:12px}h2{font-size:15px;margin:0 0 8px}
</style></head><body>
<header><h1 id="title">Alexa</h1><span class="small muted" id="state"><span class="dot"></span>…</span><div class="chips" id="metrics"></div></header>
<nav><button data-tab="sessions" class="on">Sessions</button><button data-tab="feedback">Feedback</button><button data-tab="needs">Needs</button><button data-tab="log">Log</button></nav>
<main>
<section id="tab-sessions" class="split"><div class="list" id="sessions"></div><div id="session"><div class="card muted">Pick a conversation.</div></div></section>
<section id="tab-feedback" class="hidden"><div class="card"><table id="feedback"></table></div></section>
<section id="tab-needs" class="hidden"><div class="card"><h2>New need</h2><input type="text" id="need-title" placeholder="What he actually needs, in one line" style="width:100%"><div class="tags" id="need-tags"></div><textarea id="need-notes" placeholder="Notes (optional)"></textarea><div style="margin-top:8px"><button class="go" id="need-add">Add need</button> <span class="small muted">Feedback items with the checked tags that are still new get grouped under it.</span></div></div><div class="card"><table id="needs"></table></div></section>
<section id="tab-log" class="hidden"><div class="card"><h2>Live log</h2><div class="log" id="log"></div></div><div class="card"><h2>Timeline (last 300 rows)</h2><div id="timeline"></div></div></section>
</main>
<script>
const $=s=>document.querySelector(s);const api=(p,o)=>fetch(p,o).then(r=>r.json());
const post=(p,b)=>api(p,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b)});
const patch=(p,b)=>api(p,{method:'PATCH',headers:{'Content-Type':'application/json'},body:JSON.stringify(b)});
const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const when=ts=>{const d=new Date(ts*1000);return d.toLocaleDateString(undefined,{weekday:'short'})+' '+d.toLocaleTimeString(undefined,{hour:'numeric',minute:'2-digit'})};
const secs=(a,b)=>a&&b?Math.round(b-a)+' s':'';
let TAGS=[],STATUSES=[],NEED_STATUSES=[],current=null,pickedTags=new Set(),needTags=new Set();
function tab(name){document.querySelectorAll('nav button').forEach(b=>b.classList.toggle('on',b.dataset.tab===name));['sessions','feedback','needs','log'].forEach(t=>$('#tab-'+t).classList.toggle('hidden',t!==name));if(name==='feedback')loadFeedback();if(name==='needs')loadNeeds();if(name==='log')loadLog();}
document.querySelectorAll('nav button').forEach(b=>b.onclick=()=>tab(b.dataset.tab));
async function loadState(){const s=await api('/api/state');$('#title').textContent='Alexa · '+(s.unit||'');$('#state').innerHTML='<span class="dot '+(s.listening?'on':'')+'"></span>'+esc(s.summary||s.state||'')+(s.current_session?' · in conversation #'+s.current_session:'');}
async function loadMetrics(){const m=await api('/api/metrics?days=7');const mus=m.music||{};const chip=(l,v,bad)=>'<span class="chip'+(bad&&v?' bad':'')+'">'+l+' <b>'+esc(v)+'</b></span>';
$('#metrics').innerHTML=chip('7 days · sessions',m.sessions)+chip('cut-offs',m.cut_offs,1)+chip('silences',m.silences,1)+chip('wake misses',m.wake_misses,1)+chip('acks played / skipped',(m.acks||{}).played+' / '+(m.acks||{}).skipped)+chip('music start',mus.median_ms?(mus.median_ms/1000).toFixed(1)+' s median'+(mus.fallbacks?' · '+mus.fallbacks+' fell back':''):'—',mus.fallbacks)+chip('open feedback',m.open_feedback,1)+chip('cost','$'+Number(m.cost_usd||0).toFixed(2));}
async function loadSessions(){const {sessions}=await api('/api/sessions?limit=120');$('#sessions').innerHTML=sessions.map(s=>`<div class="row${current===s.id?' on':''}" data-id="${s.id}"><div class="top"><span>#${s.id} · ${when(s.started)}</span><span>${s.flags?'<span class="badge flag">'+s.flags+' flag'+(s.flags>1?'s':'')+'</span> ':''}<span class="badge${/error|timeout|nobody/.test(s.ended_by)?' bad':''}">${esc(s.ended_by||'open')}</span></span></div><div class="line">${esc(s.first_user_line||s.summary||'(nothing said)')}</div><div class="top"><span>${(s.tools||[]).length} tool${(s.tools||[]).length===1?'':'s'} · ${secs(s.started,s.ended)}</span><span>$${Number(s.cost_usd||0).toFixed(3)}</span></div></div>`).join('');document.querySelectorAll('#sessions .row').forEach(r=>r.onclick=()=>openSession(+r.dataset.id));}
function tagPicker(el,set){el.innerHTML=TAGS.map(t=>`<span class="tag${set.has(t)?' on':''}" data-t="${t}">${t}</span>`).join('');el.querySelectorAll('.tag').forEach(x=>x.onclick=()=>{set.has(x.dataset.t)?set.delete(x.dataset.t):set.add(x.dataset.t);x.classList.toggle('on');});}
async function openSession(id){current=id;document.querySelectorAll('#sessions .row').forEach(r=>r.classList.toggle('on',+r.dataset.id===id));const d=await api('/api/session/'+id);const s=d.session;const lines=(s.transcript||[]).map(([who,text])=>`<div class="t ${who}"><span class="who">${who==='you'?'you':who==='alexa'?'alexa':'event'}</span>${esc(text)}</div>`).join('')||'<div class="muted">No transcript on this row (an older session).</div>';
const turns=(d.turns||[]).map(t=>`<tr><td>${t.turn}</td><td>${t.speech_end??''}</td><td>${t.first_call??''}</td><td>${t.first_audio??''}</td><td>${(t.tools||[]).map(x=>x[0]+' '+x[1]+'s').join(', ')}</td></tr>`).join('');
const events=(d.events||[]).map(e=>{const {ts,t,session,unit,kind,...rest}=e;const warn=/error|fail|unconfirmed|withdrawn|dropped|nobody|timeout/.test(kind+JSON.stringify(rest));return `<div class="ev${warn?' warn':''}">${t!=null?t.toFixed(2).padStart(7):'       '}  ${esc(kind)} ${esc(Object.entries(rest).map(([k,v])=>k+'='+(typeof v==='object'?JSON.stringify(v):v)).join(' '))}</div>`}).join('');
const fb=(d.feedback||[]).map(i=>`<div class="t"><span class="badge">${esc(i.status)}</span> ${esc(i.text)} <span class="small muted">${(i.tags||[]).join(', ')}</span></div>`).join('');
$('#session').innerHTML=`<div class="card"><div class="top small muted">#${s.id} · ${when(s.started)} · ${esc(s.kind)} · ended: <b>${esc(s.ended_by)}</b> · ${secs(s.started,s.ended)} · $${Number(s.cost_usd||0).toFixed(3)}${s.summary?' · '+esc(s.summary):''}</div><div style="margin-top:8px">${lines}</div></div>
<div class="card"><h2>Flag this conversation</h2><textarea id="fb-text" placeholder="What felt wrong (or right)? e.g. she cut me off before I finished"></textarea><div class="tags" id="fb-tags"></div><button class="go" id="fb-send">Save feedback</button> <span class="small muted" id="fb-note"></span>${fb?'<div style="margin-top:10px">'+fb+'</div>':''}</div>
<div class="card"><h2>Turns <span class="small muted">(seconds from the wake: his last word · backend decided · her first sound)</span></h2><table><tr><th>#</th><th>speech end</th><th>first call</th><th>first audio</th><th>tools</th></tr>${turns||'<tr><td colspan=5 class="muted">no turn rows</td></tr>'}</table></div>
<div class="card"><h2>Decisions <span class="small muted">(the timeline, seconds into the session)</span></h2>${events||'<div class="muted">no timeline rows for this session (recorded before the timeline existed)</div>'}</div>`;
pickedTags=new Set();tagPicker($('#fb-tags'),pickedTags);$('#fb-send').onclick=async()=>{const text=$('#fb-text').value.trim();if(!text)return;$('#fb-send').disabled=true;await post('/api/feedback',{text,tags:[...pickedTags],sessions:[id]});$('#fb-note').textContent='saved';await loadSessions();await openSession(id);await loadMetrics();};}
async function loadFeedback(){const f=await api('/api/feedback');STATUSES=f.statuses;const needs=f.needs;$('#feedback').innerHTML='<tr><th>when</th><th>feedback</th><th>tags</th><th>sessions</th><th>status</th><th>need</th></tr>'+(f.items.map(i=>`<tr><td class="small muted">${when(i.created)}<br>${esc(i.source)}</td><td>${esc(i.text)}</td><td class="small">${(i.tags||[]).join(', ')}</td><td class="small">${(i.sessions||[]).map(s=>'<a href="#" data-s="'+s+'">#'+s+'</a>').join(' ')}</td><td><select data-id="${i.id}" data-k="status">${STATUSES.map(s=>`<option${s===i.status?' selected':''}>${s}</option>`).join('')}</select></td><td><select data-id="${i.id}" data-k="need"><option value="">—</option>${needs.map(n=>`<option value="${n.id}"${n.id===i.need?' selected':''}>#${n.id} ${esc(n.title)}</option>`).join('')}</select></td></tr>`).join('')||'<tr><td colspan=6 class="muted">Nothing flagged yet. Open a session and say what felt wrong, or tell her "flag that".</td></tr>');
$('#feedback').querySelectorAll('select').forEach(sel=>sel.onchange=async()=>{const b={};b[sel.dataset.k]=sel.dataset.k==='need'?(sel.value||null):sel.value;await patch('/api/feedback/'+sel.dataset.id,b);loadMetrics();});$('#feedback').querySelectorAll('a[data-s]').forEach(a=>a.onclick=e=>{e.preventDefault();tab('sessions');openSession(+a.dataset.s);});}
async function loadNeeds(){const f=await api('/api/feedback');NEED_STATUSES=f.need_statuses;needTags=new Set();tagPicker($('#need-tags'),needTags);const count=n=>f.items.filter(i=>i.need===n.id).length;
$('#needs').innerHTML='<tr><th>need</th><th>tags</th><th>items</th><th>status</th><th>task</th><th></th></tr>'+(f.needs.map(n=>`<tr><td><b>${esc(n.title)}</b>${n.notes?'<div class="small muted">'+esc(n.notes)+'</div>':''}</td><td class="small">${(n.tags||[]).join(', ')}</td><td>${count(n)}</td><td><select data-id="${n.id}">${NEED_STATUSES.map(s=>`<option${s===n.status?' selected':''}>${s}</option>`).join('')}</select></td><td>${n.task?'#'+n.task:'<span class="muted">—</span>'}</td><td><button class="go" data-p="${n.id}">Promote to a task</button></td></tr>`).join('')||'<tr><td colspan=6 class="muted">No needs yet. Group feedback into the thing that should actually change.</td></tr>');
$('#needs').querySelectorAll('select').forEach(sel=>sel.onchange=()=>patch('/api/needs/'+sel.dataset.id,{status:sel.value}));$('#needs').querySelectorAll('button[data-p]').forEach(b=>b.onclick=async()=>{b.disabled=true;const r=await post('/api/needs/'+b.dataset.p+'/promote',{});b.textContent=r.note||r.error||'queued';});
$('#need-add').onclick=async()=>{const title=$('#need-title').value.trim();if(!title)return;const items=f.items.filter(i=>i.status==='new'&&[...needTags].some(t=>(i.tags||[]).includes(t))).map(i=>i.id);await post('/api/needs',{title,tags:[...needTags],items,notes:$('#need-notes').value});$('#need-title').value='';$('#need-notes').value='';loadNeeds();loadMetrics();};}
async function loadLog(){const l=await api('/api/log?limit=400');$('#log').innerHTML=l.lines.map(x=>'<div>'+esc(x)+'</div>').join('');const t=await api('/api/timeline?limit=300');$('#timeline').innerHTML=t.rows.map(e=>{const {ts,t:tt,session,unit,kind,...rest}=e;return `<div class="ev">${new Date(ts*1000).toLocaleTimeString()} ${session!=null?'#'+session:'  '} ${esc(kind)} ${esc(Object.entries(rest).map(([k,v])=>k+'='+(typeof v==='object'?JSON.stringify(v):v)).join(' '))}</div>`}).join('');const el=$('#log');el.scrollTop=el.scrollHeight;}
(async()=>{const f=await api('/api/feedback');TAGS=f.tags;await loadState();await loadMetrics();await loadSessions();setInterval(loadState,3000);setInterval(()=>{if(!$('#tab-log').classList.contains('hidden'))loadLog();},4000);setInterval(loadSessions,15000);})();
</script></body></html>
"""
