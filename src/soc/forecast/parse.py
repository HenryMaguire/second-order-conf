"""Parse `<forecast> m ± s </forecast>` and `<confidence> q </confidence>` from model output.

Parsing is strict so the RL reward cannot be gamed: exactly one tag pair, plain decimal numbers,
m and q in [0, 1], s in [0, 0.5] (no distribution on [0, 1] has a larger standard deviation).
Anything else returns None, which the reward functions score as unparseable (-1).
"""

from __future__ import annotations

import math
import re
from typing import NamedTuple

_NUM = r"(\d+(?:\.\d+)?|\.\d+)"
_PM = r"(?:±|\+/-|\+-)"
_BODY_RE = re.compile(rf"^\s*{_NUM}\s*{_PM}\s*{_NUM}\s*$")
_CONF_BODY_RE = re.compile(rf"^\s*{_NUM}\s*$")
S_MAX = 0.5


class Forecast(NamedTuple):
    m: float
    s: float


def _single_tag_body(text: str, tag: str) -> str | None:
    open_, close = f"<{tag}>", f"</{tag}>"
    if text.count(open_) != 1 or text.count(close) != 1:
        return None
    start = text.index(open_) + len(open_)
    end = text.index(close)
    return text[start:end] if end >= start else None


def parse_forecast_body(body: str) -> Forecast | None:
    """Parse the text between the tags, e.g. ' 0.62 ± 0.14 '."""
    match = _BODY_RE.match(body)
    if match is None:
        return None
    m, s = float(match.group(1)), float(match.group(2))
    if not (math.isfinite(m) and math.isfinite(s)) or not 0.0 <= m <= 1.0 or not 0.0 <= s <= S_MAX:
        return None
    return Forecast(m, s)


def parse_forecast(text: str) -> Forecast | None:
    """Parse a full completion containing exactly one <forecast>...</forecast>."""
    body = _single_tag_body(text, "forecast")
    return None if body is None else parse_forecast_body(body)


def parse_confidence(text: str) -> float | None:
    """Parse a full completion containing exactly one <confidence>...</confidence>."""
    body = _single_tag_body(text, "confidence")
    if body is None:
        return None
    match = _CONF_BODY_RE.match(body)
    if match is None:
        return None
    q = float(match.group(1))
    return q if 0.0 <= q <= 1.0 else None


def format_forecast(m: float, s: float) -> str:
    """Render a forecast body as the model should write it: two decimals, '±'."""
    return f"{m:.2f} ± {s:.2f}"
