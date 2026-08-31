"""Which tier handles this turn.

Principle unchanged from the start: do not ask a model to predict difficulty,
because judging difficulty is the capability the small model lacks. Hard-route
the obvious cases on cheap signals, let the fast model handle the rest, and
escalate on its own refusal rather than on a guess.
"""

import re

# Anything needing tools, memory, files, the web, or the real world.
NEEDS_TOOLS = re.compile(
    r"\b(remind(er)?s?|calendar|schedule|meeting|appointment|email|inbox|"
    r"search|look ?up|google|weather|forecast|news|price|stock|"
    r"file|files|folder|directory|read|write|save|open|note|notes|"
    r"remember|memory|recall|earlier|last time|yesterday|tomorrow|"
    r"repo|commit|branch|pull request|issue|todo|task)\b", re.I)

# Anything where a 4B will be confidently wrong.
NEEDS_DEPTH = re.compile(
    r"\b(explain|why does|why is|how does|how do i|derive|prove|design|"
    r"architect|debug|analy[sz]e|compare|trade-?offs?|brainstorm|plan|"
    r"review|refactor|optimi[sz]e|calculate|estimate)\b", re.I)

LONG_TURN_WORDS = 28


def needs_hermes(text: str) -> str | None:
    """Return a reason to skip the fast tier, or None to try it first."""
    if NEEDS_TOOLS.search(text):
        return "needs tools"
    if NEEDS_DEPTH.search(text):
        return "needs depth"
    if len(text.split()) > LONG_TURN_WORDS:
        return "long request"
    return None
