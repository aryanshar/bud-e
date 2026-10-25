"""Fast checks for the parts that don't need a model server or a mic."""

import threading
from pathlib import Path

import pytest

from bud_e import config
from bud_e.agent import SAY, TOOL, Agent
from bud_e.assistant import wake_pattern
from bud_e.commands import instant
from bud_e.llm import ToolCall
from bud_e.text import Sentences, ThinkFilter, speakable
from bud_e.tools import Toolbox


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    monkeypatch.setenv("BUD_E_HOME", str(tmp_path))
    monkeypatch.delenv("BUD_E_CONFIG", raising=False)
    c = config.load()
    c.tools.read_roots = [str(tmp_path / "docs")]
    return c


# --- text ---------------------------------------------------------------------

def feed_all(f, pieces):
    return "".join(f.feed(p) for p in pieces) + f.flush()


def test_think_filter_split_tags():
    pieces = ["Hel", "lo <thi", "nk>secret</th", "ink> world"]
    assert feed_all(ThinkFilter(), pieces) == "Hello  world"


def test_think_filter_unclosed_is_dropped():
    assert feed_all(ThinkFilter(), ["<think>still going"]) == ""


def test_think_filter_plain_text_untouched():
    assert feed_all(ThinkFilter(), ["a < b", " and c > d"]) == "a < b and c > d"


def test_sentences_stream():
    s = Sentences()
    out = s.feed("Hi there. How")
    out += s.feed(" are you? Fine")
    out += s.flush()
    assert out == ["Hi there.", "How are you?", "Fine"]


def test_speakable_strips_markdown_and_urls():
    assert speakable("**Bold** and `code` at https://x.y/z") == "Bold and code at"
    assert speakable("- a list item") == "a list item"


# --- commands and wake word ----------------------------------------------------

def test_instant_commands():
    assert instant("stop") == ""
    assert instant("thank you.") == ""
    assert instant("what time is it?").startswith("It's ")
    assert instant("stop talking.") == ""
    assert instant("stop the timer and tell me a joke") is None
    assert instant("what time is it in Tokyo") is None
    assert instant("tell me a joke") is None


def test_wake_pattern_variants():
    w = wake_pattern(["bud-e", "jarvis"])
    for said in ["Bud-E, hi", "bud e what's up", "hey Bude", "Jarvis?"]:
        assert w.search(said), said
    assert not w.search("buddhism is interesting")


# --- config -------------------------------------------------------------------

def test_config_file_and_env(tmp_path, monkeypatch):
    p = tmp_path / "c.toml"
    p.write_text('[llm]\nmodel = "m1"\nbase_url = "http://h:1/v1/"\n'
                 '[user]\nname = "Sam"\nbogus = 1\n')
    monkeypatch.setenv("BUD_E_MODEL", "m2")
    c = config.load(p)
    assert c.llm.model == "m2"                  # env wins
    assert c.llm.base_url == "http://h:1/v1"    # trailing slash trimmed
    assert c.user.name == "Sam"
    assert c.source == p


# --- tools --------------------------------------------------------------------

def test_tool_gating(cfg):
    names = Toolbox(cfg).names
    assert "remember" in names
    assert "get_weather" not in names           # no location configured
    assert "web_search" not in names
    cfg.user.latitude, cfg.user.longitude = 1.0, 2.0
    cfg.tools.disabled = ["recall"]
    names = Toolbox(cfg).names
    assert "get_weather" in names and "recall" not in names
    if not config.IS_MAC:
        assert "get_calendar" not in names


def test_memory_roundtrip(cfg):
    tb = Toolbox(cfg)
    assert tb.run("remember", '{"fact": "likes black coffee"}') == "saved"
    assert "black coffee" in tb.run("recall", '{"query": "coffee"}')


def test_tool_errors_are_strings(cfg):
    tb = Toolbox(cfg)
    assert tb.run("nope", "{}").startswith("error: no tool")
    assert "not valid JSON" in tb.run("remember", "{broken")
    assert tb.run("remember", "{}").startswith("error: TypeError")


def test_read_file_stays_inside_roots(cfg, tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "notes.txt").write_text("hello")
    tb = Toolbox(cfg)
    assert "hello" in tb.run("read_file", '{"path": "notes.txt"}')
    secret = tmp_path.parent / "elsewhere.txt"
    secret.write_text("x")
    try:
        assert "outside" in tb.run("read_file", f'{{"path": "{secret}"}}')
    finally:
        secret.unlink()


# --- agent loop with a fake model ----------------------------------------------

class FakeLLM:
    """Replays scripted replies. Each script is a list of text pieces and,
    optionally, tool calls."""

    def __init__(self, scripts):
        self.scripts = list(scripts)
        self.seen = []
        self.model = "fake"

    def stream(self, messages, tools=None, max_tokens=None):
        self.seen.append(messages)
        pieces, calls = self.scripts.pop(0)
        for p in pieces:
            yield "text", p
        yield "calls", calls


def test_agent_plain_reply(cfg):
    llm = FakeLLM([(["<think>hm</think>Sure. ", "Canberra."], [])])
    a = Agent(cfg, llm=llm)
    assert list(a.turn("capital of Australia?")) == [
        (SAY, "Sure."), (SAY, "Canberra.")]
    assert len(a.turns) == 1
    assert a.turns[0][-1]["content"] == "Sure. Canberra."


def test_agent_runs_tools_then_answers(cfg):
    call = ToolCall("c1", "remember", '{"fact": "likes tea"}')
    llm = FakeLLM([([], [call]), (["Noted."], [])])
    a = Agent(cfg, llm=llm)
    assert list(a.turn("I like tea")) == [(TOOL, "remember"), (SAY, "Noted.")]
    second = llm.seen[1]
    assert second[-1] == {"role": "tool", "tool_call_id": "c1",
                          "content": "saved"}
    assert "likes tea" in cfg.memory_path.read_text()


def test_agent_cancel_drops_turn(cfg):
    cancel = threading.Event()
    llm = FakeLLM([(["One. ", "Two. ", "Three."], [])])
    a = Agent(cfg, llm=llm)
    out = []
    for ev in a.turn("count", cancel):
        out.append(ev)
        cancel.set()
    assert out == [(SAY, "One.")]
    assert a.turns == []


def test_agent_gives_up_after_max_steps(cfg):
    cfg.assistant.max_tool_steps = 2
    loop = ToolCall("c", "read_notes", "{}")
    a = Agent(cfg, llm=FakeLLM([([], [loop]), ([], [loop])]))
    events = list(a.turn("loop forever"))
    assert events[-1][0] == SAY and "circles" in events[-1][1]


def test_history_trims_in_halves(cfg):
    cfg.assistant.history_turns = 4
    a = Agent(cfg, llm=FakeLLM([(["ok."], [])] * 5))
    for i in range(5):
        list(a.turn(f"msg {i}"))
    assert len(a.turns) == 2


def test_example_config_parses():
    p = Path(__file__).parents[1] / "bud_e" / "config.example.toml"
    c = config.load(p)
    assert c.llm.extra_body == {"chat_template_kwargs": {"enable_thinking": False}}


def test_p90_is_never_below_median():
    from bud_e.turnlog import p90
    assert p90([5.7, 9.1]) == 9.1
    assert p90([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) == 9
    assert p90([3.0]) == 3.0
