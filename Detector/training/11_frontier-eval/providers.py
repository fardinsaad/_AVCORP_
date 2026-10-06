"""
One call path for all four models: call(alias, system, user) -> (text, meta).
The runner never needs to know which SDK a model sits behind.
"""

import datetime as dt
import json
import os
import time

import anthropic
import backoff
import openai
from dotenv import find_dotenv, load_dotenv
from openai import OpenAI

from models import MAX_TOKENS, REGISTRY, call_kwargs

load_dotenv(find_dotenv(usecwd=True), override=True)

KEYS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "google": "GEMINI_API_KEY",
}

BASE_URLS = {
    "openai": None,
    "anthropic": "https://api.anthropic.com/v1/",
    "deepseek": "https://api.deepseek.com",
    "google": "https://generativelanguage.googleapis.com/v1beta/openai/",
}

TIMEOUT_S = 900  # a single reasoning call can take several minutes
_clients = {}
_anthropic = None


def client_for(provider):
    if provider not in _clients:
        key = os.getenv(KEYS[provider])
        if not key:
            raise RuntimeError(f"{KEYS[provider]} not set")
        _clients[provider] = OpenAI(api_key=key, base_url=BASE_URLS[provider],
                                    timeout=TIMEOUT_S, max_retries=0)
    return _clients[provider]


def anthropic_client():
    global _anthropic
    if _anthropic is None:
        key = os.getenv(KEYS["anthropic"])
        if not key:
            raise RuntimeError(f"{KEYS['anthropic']} not set")
        _anthropic = anthropic.Anthropic(api_key=key, timeout=TIMEOUT_S, max_retries=0)
    return _anthropic


class Empty(RuntimeError):
    """The model returned no text, and not because it hit the token cap."""


class Refused(RuntimeError):
    """Claude declined to answer; the message carries Anthropic's policy category."""


def _permanent(exc):
    # Retrying cannot fix these. An empty reply is left for the next resume instead.
    if isinstance(exc, (Empty, Refused, openai.BadRequestError, openai.AuthenticationError,
                        openai.PermissionDeniedError, openai.NotFoundError,
                        anthropic.BadRequestError, anthropic.AuthenticationError,
                        anthropic.PermissionDeniedError, anthropic.NotFoundError)):
        return True
    if isinstance(exc, (openai.APIStatusError, anthropic.APIStatusError)):
        if exc.status_code == 402:
            return True
        msg = str(exc).lower()
        if any(w in msg for w in ("insufficient_quota", "insufficient balance", "credit balance", "billing")):
            return True
    return False


def _get(obj, *path):
    for p in path:
        obj = obj.get(p) if isinstance(obj, dict) else getattr(obj, p, None)  # older SDKs leave new fields as dicts
        if obj is None:
            return None
    return obj


@backoff.on_exception(backoff.expo, Exception, max_tries=3, max_time=1800, giveup=_permanent)
def call(alias, system, user):
    cfg = REGISTRY[alias]
    kw = call_kwargs(alias)
    started = dt.datetime.now(dt.timezone.utc)
    t0 = time.time()
    client = None if cfg.get("api") == "messages" else client_for(cfg["provider"])

    if cfg.get("api") == "responses":
        resp = client.responses.create(instructions=system, input=user, **kw)
        text = getattr(resp, "output_text", "") or ""
        status = getattr(resp, "status", None)
        reason = _get(resp, "incomplete_details", "reason")
        finish = reason or status
        hit_max = status == "incomplete" and reason == "max_output_tokens"
        usage = resp.usage
        n_in = _get(usage, "input_tokens")
        n_out = _get(usage, "output_tokens")
        n_reason = _get(usage, "output_tokens_details", "reasoning_tokens")
    elif cfg.get("api") == "messages":
        resp = anthropic_client().messages.create(
            model=cfg["id"], max_tokens=MAX_TOKENS, system=system,
            messages=[{"role": "user", "content": user}])
        if resp.stop_reason == "refusal":
            d = getattr(resp, "stop_details", None)
            raise Refused(f"{alias}: refusal (category={_get(d, 'category')}, "
                          f"explanation={_get(d, 'explanation')})")
        text = "".join(b.text for b in resp.content if b.type == "text")
        finish = resp.stop_reason
        hit_max = finish == "max_tokens"
        usage = resp.usage
        n_in = _get(usage, "input_tokens")
        n_out = _get(usage, "output_tokens")  # includes thinking
        n_reason = _get(usage, "output_tokens_details", "thinking_tokens")
    else:
        resp = client.chat.completions.create(
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}], **kw)
        choice = resp.choices[0]
        text = choice.message.content or ""
        finish = choice.finish_reason
        hit_max = finish == "length"
        usage = resp.usage
        n_in = _get(usage, "prompt_tokens")
        n_out = _get(usage, "completion_tokens")
        n_reason = _get(usage, "completion_tokens_details", "reasoning_tokens")
    n_total = _get(usage, "total_tokens")
    if cfg["provider"] == "google" and n_reason is None and n_total and n_in and n_out is not None:
        n_reason = n_total - n_in - n_out  # Gemini counts thinking in the total but not in completion_tokens

    if not text.strip() and not hit_max:
        raise Empty(f"{alias}: empty response (finish={finish})")

    # Some compatibility layers leave thinking out of the output count but keep it
    # in the total, so the larger of the two is everything the model generated.
    n_gen = max(n_out or 0, (n_total - n_in) if (n_total and n_in) else 0)
    meta = {
        "model_id": cfg["id"],
        "model_served": getattr(resp, "model", None),
        "finish_reason": finish,
        "hit_max_tokens": bool(hit_max),
        "utc": started.isoformat(timespec="seconds"),
        "seconds": round(time.time() - t0, 1),
        "prompt_tokens": n_in,
        "gen_tokens": n_gen,
        "reasoning_tokens": n_reason,
        "usage_raw": json.dumps(usage.model_dump() if hasattr(usage, "model_dump") else None),
    }
    return text, meta
