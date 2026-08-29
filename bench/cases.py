"""The suite. Shaped like real Bud-E turns, not like a leaderboard."""

TOOLS = [
    {"type": "function", "function": {
        "name": "get_calendar",
        "description": "Get the user's events for a given day.",
        "parameters": {"type": "object",
                       "properties": {"day": {"type": "string"}},
                       "required": ["day"]}}},
    {"type": "function", "function": {
        "name": "recall",
        "description": "Search the user's memory of past conversations.",
        "parameters": {"type": "object",
                       "properties": {"query": {"type": "string"}},
                       "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "set_reminder",
        "description": "Create a reminder at a time.",
        "parameters": {"type": "object",
                       "properties": {"text": {"type": "string"},
                                      "when": {"type": "string"}},
                       "required": ["text", "when"]}}},
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Read a file from disk by path.",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string"}},
                       "required": ["path"]}}},
]

SYSTEM = (
    "You are a voice assistant. Replies are read aloud: two or three "
    "sentences, no markdown, no lists. Use a tool when one fits. "
    "If a request needs more reasoning than you can do reliably, reply with "
    "exactly ESCALATE: followed by a one line restatement."
)

# (prompt, expected) where expected is a tool name, None for a plain
# spoken answer, or "ESCALATE" for a turn that should be declined.
CASES = [
    ("What's on my calendar tomorrow?", "get_calendar"),
    ("Remind me to ring the plumber on Tuesday at four.", "set_reminder"),
    ("What did I say about the flat last week?", "recall"),
    ("Read me what's in notes dot txt.", "read_file"),
    ("Roughly how far away is the moon?", None),
    ("Give me a two sentence summary of what a mixture of experts model is.",
     None),
    ("Book a haircut Thursday and tell me what else is on that day.",
     "get_calendar"),
    ("Derive a closed form for the sum of the first n cubes and prove it by "
     "induction.", "ESCALATE"),
]
