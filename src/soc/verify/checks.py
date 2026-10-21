"""Programmatic answer checks: answer extraction, numeric/math, multiple choice, exact match."""

from __future__ import annotations

import math
import re
import string

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def extract_answer(response: str) -> str | None:
    """Content of the last <answer>...</answer>, stripped; None if absent or empty."""
    found = _ANSWER_RE.findall(response)
    if not found:
        return None
    ans = found[-1].strip()
    return ans or None


# --- math ----------------------------------------------------------------------------------

_BOXED_RE = re.compile(r"\\boxed\{(.*)\}")
_FRAC_RE = re.compile(r"^(-?)\\d?frac\{(-?[\d.]+)\}\{(-?[\d.]+)\}$")
_NUM_RE = re.compile(r"^[-+]?(\d+(\.\d*)?|\.\d+)$")
_SLASH_RE = re.compile(r"^([-+]?\d+)\s*/\s*(\d+)$")


def to_number(text: str) -> float | None:
    """Parse a plain numeric answer, tolerating LaTeX wrappers, '$', thousands commas, a
    leading 'x =', and trailing units. Returns None for anything non-numeric."""
    s = text.strip()
    m = _BOXED_RE.search(s)
    if m:
        s = m.group(1)
    s = s.replace("$", "").replace("\\,", "").replace("\\!", "").replace("{,}", "")
    s = re.sub(r"\\text\{[^}]*\}", "", s)
    s = re.sub(r"^[A-Za-z]\w*\s*=\s*", "", s).strip()
    s = s.rstrip(".").strip()
    if re.fullmatch(r"[-+]?\d{1,3}(,\d{3})+(\.\d+)?", s):
        s = s.replace(",", "")
    m = _FRAC_RE.match(s.replace(" ", ""))
    if m:
        num, den = float(m.group(2)), float(m.group(3))
        return None if den == 0 else (-1 if m.group(1) else 1) * num / den
    m = _SLASH_RE.match(s)
    if m:
        den = int(m.group(2))
        return None if den == 0 else int(m.group(1)) / den
    # Trailing units or a percent sign: "10 cm", "25%", "12 degrees".
    m = re.match(r"^([-+]?(?:\d+(?:\.\d*)?|\.\d+))\s*(?:%|°|\\%|[A-Za-z][A-Za-z ]*)?$", s)
    if m and _NUM_RE.match(m.group(1)):
        return float(m.group(1))
    return None


def check_math(pred: str, gold: str, rel_tol: float = 1e-6) -> bool:
    p, g = to_number(pred), to_number(gold)
    if p is not None and g is not None:
        return math.isclose(p, g, rel_tol=rel_tol, abs_tol=1e-9)
    # Fall back to math-verify for symbolic answers (rare in BigMathDigits).
    try:
        from math_verify import parse, verify

        return bool(verify(parse(f"${gold}$"), parse(f"${pred}$")))
    except Exception:
        return False


# --- multiple choice -----------------------------------------------------------------------

_LETTER_RE = re.compile(r"^\(?([A-Da-d])\)?(?:[.):]|\s|$)")


def check_mcq(pred: str, gold: str) -> bool:
    s = re.sub(r"^(?:the\s+)?(?:answer|option)(?:\s+is)?\s*:?\s*", "", pred.strip(), flags=re.I)
    m = _LETTER_RE.match(s)
    return m is not None and m.group(1).upper() == gold.strip().upper()


# --- short answers -------------------------------------------------------------------------


def normalize(text: str) -> str:
    """SQuAD/TriviaQA normalisation: lowercase, drop punctuation and articles, squash spaces."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(pred: str, golds: list[str] | tuple[str, ...]) -> bool:
    p = normalize(pred)
    return bool(p) and any(p == normalize(g) for g in golds)
