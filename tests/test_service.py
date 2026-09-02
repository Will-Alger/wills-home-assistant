"""The watchdog's staging logic, as pure functions — no processes spawned."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

SERVICE = Path(__file__).resolve().parents[1] / "scripts" / "alexa_service.py"
spec = importlib.util.spec_from_file_location("alexa_service", SERVICE)
service = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(service)


def layout(tmp_path: Path, *, worktree_venv: bool) -> tuple[Path, Path]:
    root = tmp_path / "main"
    (root / ".venv" / "Scripts").mkdir(parents=True)
    (root / ".venv" / "Scripts" / "python.exe").write_text("", encoding="utf-8")
    (root / "scripts").mkdir()
    (root / "scripts" / "m4_realtime.py").write_text("", encoding="utf-8")
    (root / "data").mkdir()
    wt = root / ".worktrees" / "7-volume"
    (wt / "scripts").mkdir(parents=True)
    (wt / "scripts" / "m4_realtime.py").write_text("", encoding="utf-8")
    (wt / "src").mkdir()
    if worktree_venv:
        (wt / ".venv" / "Scripts").mkdir(parents=True)
        (wt / ".venv" / "Scripts" / "python.exe").write_text("", encoding="utf-8")
    return root, wt


def write_pointer(root: Path, wt: Path, set_at: float = 1.0) -> dict:
    data = {"task_id": 7, "slug": "7-volume", "worktree": str(wt), "iteration": 1, "set_at": set_at}
    (root / "data" / "active_checkout.json").write_text(json.dumps(data), encoding="utf-8")
    return data


def test_main_launch_when_no_pointer(tmp_path: Path) -> None:
    root, _ = layout(tmp_path, worktree_venv=False)
    assert service.read_pointer(root) is None
    cmd = service.child_command(root, None, argv=["svc"])
    assert cmd == [str(root / ".venv" / "Scripts" / "python.exe"), str(root / "scripts" / "m4_realtime.py")]
    env = service.child_env(root, None, base={})
    assert env["ALEXA_HOME"] == str(root) and env["PYTHONUTF8"] == "1"
    assert "ALEXA_STAGED_TASK" not in env and "PYTHONPATH" not in env
    assert service.child_command(root, None, argv=["svc", "--", "echo", "hi"]) == ["echo", "hi"]


def test_staged_launch_uses_the_worktrees_own_venv(tmp_path: Path) -> None:
    root, wt = layout(tmp_path, worktree_venv=True)
    write_pointer(root, wt)
    pointer = service.read_pointer(root)
    assert pointer and pointer["task_id"] == 7
    cmd = service.child_command(root, pointer, argv=["svc"])
    assert cmd == [str(wt / ".venv" / "Scripts" / "python.exe"), str(wt / "scripts" / "m4_realtime.py")]
    env = service.child_env(root, pointer, base={})
    assert env["ALEXA_STAGED_TASK"] == "7" and env["ALEXA_HOME"] == str(root)
    assert "PYTHONPATH" not in env  # its own venv already points at its src


def test_staged_launch_falls_back_to_main_python_plus_pythonpath(tmp_path: Path) -> None:
    root, wt = layout(tmp_path, worktree_venv=False)
    write_pointer(root, wt)
    pointer = service.read_pointer(root)
    cmd = service.child_command(root, pointer, argv=["svc"])
    assert cmd[0] == str(root / ".venv" / "Scripts" / "python.exe")
    assert cmd[1] == str(wt / "scripts" / "m4_realtime.py")
    env = service.child_env(root, pointer, base={"PYTHONPATH": "C:\\elsewhere"})
    assert env["PYTHONPATH"].startswith(str(wt / "src"))
    assert "C:\\elsewhere" in env["PYTHONPATH"]


def test_pointer_at_a_vanished_worktree_means_main(tmp_path: Path) -> None:
    root, _wt = layout(tmp_path, worktree_venv=False)
    write_pointer(root, root / ".worktrees" / "gone")
    notes: list[str] = []
    assert service.read_pointer(root, notes.append) is None
    assert notes and "no runnable app" in notes[0]
    (root / "data" / "active_checkout.json").write_text("{broken", encoding="utf-8")
    assert service.read_pointer(root, notes.append) is None


def test_two_quick_crashes_roll_back_but_healthy_runs_reset(tmp_path: Path) -> None:
    root, wt = layout(tmp_path, worktree_venv=True)
    pointer = write_pointer(root, wt, set_at=42.0)
    state: dict = {}
    assert service.record_exit(state, pointer, code=1, ran_for=5) == "ok"
    assert service.record_exit(state, pointer, code=0, ran_for=3) == "ok"  # clean exit resets
    assert service.record_exit(state, pointer, code=1, ran_for=5) == "ok"
    assert service.record_exit(state, pointer, code=1, ran_for=500) == "ok"  # it ran fine for a while
    assert service.record_exit(state, pointer, code=1, ran_for=5) == "ok"
    assert service.record_exit(state, pointer, code=1, ran_for=7) == "rollback"
    # main never rolls back
    assert service.record_exit({}, None, code=1, ran_for=1) == "ok"

    service.rollback(root, pointer, code=1)
    assert not (root / "data" / "active_checkout.json").exists()
    failed = json.loads((root / "data" / "active_checkout.failed.json").read_text(encoding="utf-8"))
    assert failed["task_id"] == 7
    drops = list((root / "data" / "announcements" / "inbox").glob("*.json"))
    assert len(drops) == 1
    drop = json.loads(drops[0].read_text(encoding="utf-8"))
    assert drop["priority"] == "urgent" and "rolled back" in drop["text"]
    assert drop["ref"] == "task:7:rollback"
    assert service.read_pointer(root) is None  # the next launch is main
