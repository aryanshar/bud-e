"""The suite. Shaped like real Bud-E turns, not like a leaderboard."""

# Bench the exact prompt and tool schemas the agent tier runs with, so a
# score here means something for the voice path.
from bud_e.agent import system_prompt
from bud_e.tools import schemas

TOOLS = schemas()
SYSTEM = system_prompt()

# (prompt, expected) where expected is a tool name, None for a plain
# spoken answer. User turns carry the same [date, time] stamp the agent adds.
CASES = [
    ("What's on my calendar tomorrow?", "get_calendar"),
    ("Remind me to ring the plumber on Tuesday at four.", "set_reminder"),
    ("What did I say about the flat last week?", "recall"),
    ("Read me what's in notes dot txt.", ("read_file", "find_files")),
    ("Roughly how far away is the moon?", None),
    ("Give me a two sentence summary of what a mixture of experts model is.",
     None),
    ("Book a haircut Thursday and tell me what else is on that day.",
     "get_calendar"),
    ("Will it rain this afternoon?", "get_weather"),
    ("By the way, I prefer my coffee black, no sugar.", "remember"),
    ("Derive a closed form for the sum of the first n cubes.", None),
]
