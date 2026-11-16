"""Start spawned worker processes without executing ``squadds/__init__.py`` (which imports the database and analysis stack).

``boot(fn, *args)`` returns ``(initializer, initargs)`` for a ``ProcessPoolExecutor``: the worker registers a bare ``squadds``
package module, then imports ``fn`` by name and calls ``fn(*args)``; at exit it undoes the prior stack (quiet shutdown). The worker functions and their arguments are unchanged.
"""

from __future__ import annotations

import os
import pickle

_PKG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), *[os.pardir] * 3))

_CODE = """
import importlib, pickle, sys, types
if "squadds" not in sys.modules:
    _m = types.ModuleType("squadds")
    _m.__path__ = [{pkg!r}]
    sys.modules["squadds"] = _m
_f = getattr(importlib.import_module({mod!r}), {name!r})
_f(*pickle.loads({blob!r}))


def _teardown():
    a = sys.modules.get("squadds.ml.capacitance._core.api")
    if a is not None and a._STATE.get("stack") is not None:
        try:
            a._teardown()
        except Exception:
            pass


import atexit
atexit.register(_teardown)
"""


def boot(fn, *args) -> tuple:
    code = _CODE.format(pkg=_PKG_DIR, mod=fn.__module__, name=fn.__name__, blob=pickle.dumps(args))
    return exec, (code, {})
