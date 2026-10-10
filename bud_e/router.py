"""Which tier handles this turn, and whether the 27B should think first.

Do not ask a model to predict difficulty: that is the capability the small
model lacks. Hard-route the obvious cases on cheap signals, let the fast model
take the rest, and fall through to the agent when it says ROUTE.
"""

import re

from bud_e import config as C

# Needs a tool. Kept to words that almost always mean one; the fast model's
# ROUTE catches the rest, so this does not need to be exhaustive.
NEEDS_TOOLS = re.compile(
    r"\b(remind(er)?s?|calendar|schedule|meeting|appointment|agenda|"
    r"weather|forecast|rain|search|look ?up|google|news|"
    r"files?|folder|notes?|remember|recall|last time|"
    r"morning brief|my day|todo|to-do)\b", re.I)

# Worth thinking tokens: the user waits a few seconds on purpose.
NEEDS_THINK = re.compile(
    r"\b(derive|prove|proof|design|architect|debug|analy[sz]e|"
    r"trade-?offs?|brainstorm|plan out|calculate|estimate|step by step|"
    r"think (it )?through|work out)\b", re.I)

LONG_TURN_WORDS = 28


def route(text: str) -> tuple[str, bool, str]:
    """Returns (tier, think, reason). tier is 'fast' or 'agent'."""
    if NEEDS_THINK.search(text):
        return "agent", True, "needs thought"
    if NEEDS_TOOLS.search(text):
        return "agent", False, "needs tools"
    if len(text.split()) > LONG_TURN_WORDS:
        return "agent", False, "long request"
    if not C.FAST_TIER:
        return "agent", False, "fast tier off"
    return "fast", False, ""
