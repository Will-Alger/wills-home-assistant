"""Keep the test suite's subprocesses (git, the fake agent) from opening
console windows: the test runner may have no console of its own, and a
console child then gets a brand-new, empty terminal window on Windows."""

from __future__ import annotations

import subprocess
import sys

if sys.platform == "win32":
    _orig_init = subprocess.Popen.__init__

    def _quiet_init(self, *args, **kwargs):
        kwargs["creationflags"] = kwargs.get("creationflags", 0) | subprocess.CREATE_NO_WINDOW
        _orig_init(self, *args, **kwargs)

    subprocess.Popen.__init__ = _quiet_init  # type: ignore[method-assign]
