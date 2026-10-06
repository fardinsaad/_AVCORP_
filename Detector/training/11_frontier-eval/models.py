"""
Model registry for the AVCORP frontier evaluation.

    python models.py            list what each key can reach
    python models.py --check    confirm every registry id is live for your keys
"""

import argparse
import os
import sys

import requests
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True), override=True)

MAX_TOKENS = 32768  # reasoning and the answer share this cap

# Every model runs at its provider's default reasoning setting; no effort or temperature is sent.
REGISTRY = {
    "gpt-5.6-sol": dict(provider="openai", id="gpt-5.6-sol",
                        api="responses", max_tokens_key="max_output_tokens"),
    # Claude goes through Anthropic's own Messages API: the OpenAI-compatible endpoint hides refusals and thinking tokens.
    "claude-opus-5.5": dict(provider="anthropic", id="claude-opus-5-5", api="messages"),
    # Same Claude, with prompt.NEUTRAL wording: the original run was refused in 16 of 25 games.
    "claude-opus-5.5-neutral": dict(provider="anthropic", id="claude-opus-5-5", api="messages", neutral=True),
    "gemini-3.1-pro": dict(provider="google", id="gemini-3.1-pro-preview"),
    "deepseek-v4-pro": dict(provider="deepseek", id="deepseek-v4-pro"),
}

ENDPOINTS = {
    "openai":    ("https://api.openai.com/v1/models",    "OPENAI_API_KEY",    "bearer"),
    "anthropic": ("https://api.anthropic.com/v1/models", "ANTHROPIC_API_KEY", "x-api-key"),
    "deepseek":  ("https://api.deepseek.com/models",     "DEEPSEEK_API_KEY",  "bearer"),
    "google":    ("https://generativelanguage.googleapis.com/v1beta/models",
                                                         "GEMINI_API_KEY",    "query"),
}


def call_kwargs(alias):
    cfg = REGISTRY[alias]
    return {"model": cfg["id"], cfg.get("max_tokens_key", "max_tokens"): MAX_TOKENS}


def list_models(provider, timeout=30):
    url, env, scheme = ENDPOINTS[provider]
    key = os.getenv(env)
    if not key:
        return None, f"{env} not set in .env"
    headers, params = {}, {}
    if scheme == "bearer":
        headers["Authorization"] = f"Bearer {key}"
    elif scheme == "x-api-key":
        headers.update({"x-api-key": key, "anthropic-version": "2023-06-01"})
        params["limit"] = 1000
    else:
        params["key"] = key
        params["pageSize"] = 1000
    try:
        r = requests.get(url, headers=headers, params=params, timeout=timeout)
        r.raise_for_status()
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"
    payload = r.json()
    items = payload.get("data") or payload.get("models") or []
    ids = []
    for it in items:
        i = it.get("id") or it.get("name", "")
        ids.append(i.split("/")[-1] if provider == "google" else i)
    return sorted(i for i in ids if i), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--filter", default="")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    listings = {}
    for provider in ENDPOINTS:
        ids, err = list_models(provider)
        listings[provider] = ids
        if err:
            print(f"{provider:<10} UNAVAILABLE: {err}")
            continue
        shown = [i for i in ids if args.filter.lower() in i.lower()]
        print(f"{provider:<10} {len(ids)} models" + (f", {len(shown)} matching {args.filter!r}" if args.filter else ""))
        if args.filter:
            for i in shown[:50]:
                print(f"    {i}")

    print(f"\n{'alias':<17}{'id':<26}status")
    blocked = []
    for alias, cfg in REGISTRY.items():
        status = ""
        if args.check:
            ids = listings.get(cfg["provider"])
            if ids is None:
                status = "provider unreachable"
                blocked.append(alias)
            elif cfg["id"] in ids:
                status = "present"
            else:
                status = "NOT IN LISTING"
                blocked.append(alias)
        print(f"{alias:<17}{cfg['id']:<26}{status}")
    if args.check:
        print(f"\nNot ready: {', '.join(blocked)}" if blocked else "\nAll four models resolve.")
        return 1 if blocked else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
