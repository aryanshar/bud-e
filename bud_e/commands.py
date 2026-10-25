"""Things you say often that shouldn't wait on a model.

Kept deliberately tiny: stop-words, and questions with one exact answer.
Anything needing judgement or parsing goes to the model.

`instant(text)` returns None if nothing matched, "" to stay quiet, or a
sentence to say.
"""

import re
from datetime import datetime

_RULES: list[tuple[re.Pattern, object]] = [
    (re.compile(r"^(stop|quiet|shut up|never ?mind|forget it|cancel)( it| that| talking)?\W*$", re.I),
     lambda: ""),
    (re.compile(r"^(thanks|thank you|cheers|nice one)\W*$", re.I),
     lambda: ""),
    (re.compile(r"^what('s| is) the time\W*$|^what time is it\W*$", re.I),
     lambda: f"It's {datetime.now():%-I:%M}."),
    (re.compile(r"^what('s| is) (the date|today's date)\W*$|"
                r"^what day is (it|today)\W*$", re.I),
     lambda: f"It's {datetime.now():%A the %-d of %B}."),
]


def instant(text: str) -> str | None:
    t = text.strip()
    for pattern, answer in _RULES:
        if pattern.search(t):
            return answer()
    return None
