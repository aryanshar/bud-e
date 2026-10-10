"""The local endpoint. One client, shared by the bench, both tiers and the CLI."""

from openai import OpenAI

from bud_e import config as C

_client: OpenAI | None = None


def local(base: str | None = None) -> OpenAI:
    global _client
    if _client is None or base:
        _client = OpenAI(base_url=base or C.LOCAL_BASE, api_key=C.API_KEY,
                         timeout=120)
    return _client


def list_models(base: str | None = None) -> list[str]:
    return [m.id for m in local(base).models.list().data]


def sampling(think: bool, max_tokens: int) -> dict:
    """Request kwargs for Qwen. top_k, min_p and the thinking switch are not
    OpenAI fields, so they ride in extra_body, which oMLX passes through."""
    s = C.SAMPLING[think]
    return dict(
        temperature=s["temperature"], top_p=s["top_p"],
        presence_penalty=s["presence_penalty"], max_tokens=max_tokens,
        extra_body={"top_k": s["top_k"], "min_p": s["min_p"],
                    "chat_template_kwargs": {"enable_thinking": think}})
