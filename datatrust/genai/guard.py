"""
datatrust/genai/guard.py

Checks an AI-written summary before it is accepted. The scores themselves are never taken from
AI text (the report and pages always show the engine's numbers); this guard stops the AI text
from contradicting them or adding numbers of its own.

Rejected, so the template summary is used instead:
  - empty or absurdly long output
  - any number that is not in the facts (an invented or recalculated score, count or percentage).
    Small whole numbers 0-10 and 100 are allowed ("the top 3 problems", "out of 100").
  - numbers used for the WRONG thing. Found with llama3.2:3b on the sample file: "seven problems"
    (there are 12), "beats guessing by a balanced accuracy of 0.693" (the lift is 0.193), "the most
    critical problems" (0 are critical). Each number also exists somewhere in the facts, so a
    membership check alone passes them; these checks look at what the number describes:
      * "<N> issues/problems" must be a real issue count (written in digits or words)
      * "beats guessing by <X>" must be the real lift
      * "critical"/"high" issues may only be mentioned when there are some ("no critical issues" is fine:
        found when qwen2.5:7b and llama3.2:3b were both rejected for that correct sentence)
"""
from __future__ import annotations

import re

from datatrust.genai.facts import allowed_numbers, numbers_in

MAX_CHARS = 4000
ALWAYS_ALLOWED = {float(n) for n in range(0, 11)} | {100.0}


WORDS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
    "sixteen seventeen eighteen nineteen twenty".split())}
_COUNT = re.compile(r"\b(\d+|" + "|".join(WORDS) + r")\s+(?:[a-z-]+\s+){0,2}?(issues?|problems?)\b", re.I)
_LIFT = re.compile(r"beat(?:s|ing)?\s+(?:random\s+)?guess(?:ing)?\s+by\D{0,40}?(-?\d+(?:\.\d+)?)", re.I)
_SEVERE = re.compile(r"\b(critical|high[- ]severity|high-priority)\s+(?:data[- ]quality\s+)?(issues?|problems?)\b", re.I)


_NEGATION = re.compile(r"\b(no|none|not|zero|without|neither|nor|free of|aren't|isn't|are not|is not|0)\b[^.!?]*$", re.I)


def _negated(text: str, start: int) -> bool:
    """'there are no critical issues' is correct when there are none: look back within the sentence."""
    return bool(_NEGATION.search(text[max(0, start - 60):start]))


def _count_value(token: str) -> float:
    return float(WORDS[token.lower()]) if token.lower() in WORDS else float(token)


def _meaning_checks(text: str, facts: dict) -> None:
    c = facts["issue_counts"]
    real_counts = {float(v) for v in c.values()} | {float(facts.get("issues_listed_below", 0)),
                                                    float(len(facts.get("top_actions", [])))}
    for m in _COUNT.finditer(text):
        if _count_value(m.group(1)) not in real_counts:
            raise SummaryRejected(f"the AI text says '{m.group(0)}', but the issue counts are "
                                  f"{c['total']} in total ({c['critical']} critical, {c['high']} high, "
                                  f"{c['medium']} medium, {c['low']} low)")
    model = facts.get("model")
    for m in _LIFT.finditer(text):
        real = model.get("beats_guessing_by") if model else None
        if real is None or abs(float(m.group(1)) - abs(real)) > 0.0006:
            raise SummaryRejected(f"the AI text says the model {m.group(0)!r}, but it beats guessing by {real}")
    for m in _SEVERE.finditer(text):
        level = "critical" if m.group(1).lower() == "critical" else "high"
        if c[level] == 0 and not _negated(text, m.start()):
            raise SummaryRejected(f"the AI text mentions '{m.group(0)}', but no issue is {level}")


class SummaryRejected(ValueError):
    """The AI output failed a check; the message says which (safe to show and log)."""


def check_summary(text: str, facts: dict) -> str:
    text = (text or "").strip()
    if not text:
        raise SummaryRejected("the AI returned an empty text")
    if len(text) > MAX_CHARS:
        raise SummaryRejected(f"the AI text is too long ({len(text)} characters)")
    allowed = allowed_numbers(facts) | ALWAYS_ALLOWED
    unknown = sorted(n for n in numbers_in(text) if n not in allowed)
    if unknown:
        shown = ", ".join(f"{n:g}" for n in unknown[:5])
        raise SummaryRejected(f"the AI text contains number(s) not in the verified facts: {shown}")
    _meaning_checks(text, facts)
    return text
