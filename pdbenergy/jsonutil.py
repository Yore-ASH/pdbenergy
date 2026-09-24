"""Emit JSON that strict parsers accept.

Why this module exists
----------------------
Python's :mod:`json` writes the bare tokens ``NaN``, ``Infinity`` and
``-Infinity`` for non-finite floats.  Those are **not valid JSON** - they are a
Python extension - and strict parsers reject them.  JavaScript's
``JSON.parse`` fails with::

    SyntaxError: Unexpected token 'N', ... "ank_rho": NaN}, "tes" ... is not valid JSON

That is exactly what a browser GUI saw when a metrics report contained a NaN.
Non-finite values are not exotic here: every metric in
:func:`pdbenergy.train.regression_metrics` degenerates to NaN when it is
undefined (R^2 with zero target variance, a correlation over a single sample,
a within-protein rank correlation when no protein has enough frames).  So the
correct behaviour is to map "undefined" onto JSON ``null``, which clients
already render as a dash, rather than to emit a token they cannot parse.

Use :func:`dump_json` / :func:`dumps_json` everywhere the project writes JSON,
and :func:`jsonable` before handing a payload to any other serializer.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any


def jsonable(value: Any) -> Any:
    """Recursively replace non-finite floats with ``None``.

    Everything else is passed through, with mappings converted to plain dicts so
    the result is trivially serialisable and comparable in tests.
    """
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    # numpy scalars/arrays and anything else that is not natively JSON.
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return jsonable(item())
        except Exception:
            pass
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return jsonable(tolist())
        except Exception:
            pass
    return str(value)


def dumps_json(payload: Any, *, indent: int | None = None) -> str:
    """Serialise to strict JSON (no ``NaN`` / ``Infinity`` tokens).

    ``allow_nan=False`` is deliberate: :func:`jsonable` should have removed every
    non-finite value, so this turns a silent wire-format bug into a loud one.
    """
    return json.dumps(jsonable(payload), indent=indent, ensure_ascii=False,
                      allow_nan=False, default=str)


def dump_json(path: str | os.PathLike, payload: Any, *, indent: int | None = 2) -> str:
    """Write strict JSON to ``path``, creating parent directories."""
    os.makedirs(os.path.dirname(os.path.abspath(str(path))), exist_ok=True)
    with open(str(path), "w", encoding="utf-8") as fh:
        fh.write(dumps_json(payload, indent=indent))
    return str(path)
