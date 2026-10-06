"""
Runs the 25 AVCORP validation games (100 rows) on the frontier models.

    run_all()                     every model, every game; resumes what is done
    run_all(games=1)              pilot: the first game of each model
    run_model("gemini-3.1-pro")   one model
    progress()                    rows and token usage per model
    score_all()                   eval_predictions_<alias>.csv and eval_summary.csv

Each row sees the model's own traces from the two rounds before it, so the
rounds of a game run in order. Games are independent, and `game_workers` runs
several games of one model at once.
"""

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

import prompt as P
from models import REGISTRY
from providers import call

HERE = Path(__file__).resolve().parent
RAW = HERE / "results" / "raw"
OUT = HERE / "results"
NO_THINK = False  # True sends Qwen's system prompt unchanged, "/no_think" included
RETIRED = {"claude-opus-5.5"}  # refused as reasoning extraction; left out of run_all() and score_all(), see claude-opus-5.5-neutral

GAME_WORKERS = 5  # games of one model in flight at once

_print_lock = threading.Lock()
_model_locks = {a: threading.Lock() for a in REGISTRY}


def log(msg):
    with _print_lock:
        print(msg, flush=True)


def shard_path(alias):
    return RAW / f"{alias}.jsonl"


def load_done(alias):
    done = {}
    p = shard_path(alias)
    if p.exists():
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    done[(r["game_id"], int(r["round_id"]))] = r
    return done


def reasoning_tokens(row):
    """Stored count, else worked out from usage_raw (Gemini: total - prompt - completion; Claude: thinking_tokens)."""
    if row.get("reasoning_tokens") is not None:
        return row["reasoning_tokens"]
    u = json.loads(row.get("usage_raw") or "null") or {}
    if str(row.get("model_id", "")).startswith("gemini") and u.get("total_tokens"):
        return u["total_tokens"] - u["prompt_tokens"] - u["completion_tokens"]
    return (u.get("output_tokens_details") or {}).get("thinking_tokens")  # Claude rows from the Messages API


def pending(alias=None):
    out = {}
    for a in ([alias] if alias else REGISTRY):
        done = load_done(a)
        out[a] = [(r["game_id"], int(r["round_id"])) for r in P.VAL_ROWS
                  if (r["game_id"], int(r["round_id"])) not in done]
    return out


def _pick_games(games):
    if games is None:
        return list(P.GAMES)
    if isinstance(games, int):
        return list(P.GAMES[:games])
    return list(games)


def run_model(alias, games=None, game_workers=GAME_WORKERS):
    """games: None for all 25, an int for the first n, or a list of game ids."""
    RAW.mkdir(parents=True, exist_ok=True)
    done = load_done(alias)
    state = {"ok": 0, "skipped": 0, "failed": 0}
    lock = _model_locks[alias]

    def one_game(game_id):
        own = []
        for row in P.ROWS_BY_GAME[game_id]:
            key = (game_id, int(row["round_id"]))
            chain = list(own[-P.CHAIN_WINDOW:])
            if key in done:
                out = done[key]["output"]
                with lock:
                    state["skipped"] += 1
            else:
                system, user = P.messages_for(row, chain, no_think=NO_THINK, neutral=REGISTRY[alias].get("neutral", False))
                t0 = time.time()
                try:
                    out, meta = call(alias, system, user)
                except Exception as exc:
                    with lock:
                        state["failed"] += 1
                        with open(RAW / f"{alias}.errors.log", "a", encoding="utf-8") as fh:
                            fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {game_id} R{key[1]}  "
                                     f"{type(exc).__name__}: {exc}\n")
                    # Later rounds need this round's trace, so the rest of the game waits for a re-run.
                    log(f"  {alias:<16} {game_id} R{key[1]}  FAILED {type(exc).__name__}: {str(exc)[:160]}")
                    return
                rec = {"game_id": game_id, "round_id": key[1], "role_id": row["role_id"],
                       "player_roles": row["player_roles"], "chain_mode": "self",
                       "chain_rounds": " ".join(str(c["round_id"]) for c in chain),
                       "no_think": NO_THINK, "neutral": REGISTRY[alias].get("neutral", False), "output": out, **meta}
                with lock:
                    with open(shard_path(alias), "a", encoding="utf-8") as fh:
                        fh.write(json.dumps(rec) + "\n")
                        fh.flush()
                        os.fsync(fh.fileno())
                    done[key] = rec
                    state["ok"] += 1
                    n_done = len(done)
                reasoning = f" (reasoning {meta['reasoning_tokens']})" if meta["reasoning_tokens"] else ""
                log(f"  {alias:<16} {game_id} R{key[1]}  {n_done:>3}/100  {time.time() - t0:6.1f}s  "
                    f"in {meta['prompt_tokens']} out {meta['gen_tokens']}{reasoning}"
                    f"{'  HIT MAX TOKENS' if meta['hit_max_tokens'] else ''}")
            own.append({"round_id": key[1], "reasoning_gold": P._chain_text(out)})

    todo = _pick_games(games)
    log(f"  {alias:<16} start: {len(todo)} games, {game_workers} at a time")
    with ThreadPoolExecutor(max_workers=game_workers) as pool:
        for f in as_completed([pool.submit(one_game, g) for g in todo]):
            f.result()
    log(f"  {alias:<16} DONE ok={state['ok']} skipped={state['skipped']} failed={state['failed']}")
    return state


def run_all(aliases=None, games=None, game_workers=GAME_WORKERS):
    aliases = aliases or [a for a in REGISTRY if a not in RETIRED]
    log(f"\n=== {len(aliases)} models, {len(_pick_games(games))} games each ===")
    with ThreadPoolExecutor(max_workers=len(aliases)) as pool:
        for f in as_completed([pool.submit(run_model, a, games, game_workers) for a in aliases]):
            f.result()
    return progress()


def progress():
    recs = []
    for a in REGISTRY:
        done = load_done(a)
        rows = list(done.values())
        n = len(rows)
        reasoning = [t for t in map(reasoning_tokens, rows) if t is not None]
        errs = RAW / f"{a}.errors.log"
        recs.append({
            "model": a, "rows": n,  # how many rows we've scraped off disk so far
            "games_complete": sum(all((g, int(r["round_id"])) in done for r in P.ROWS_BY_GAME[g])
                                  for g in P.GAMES),  # games where every round has landed
            "mean_in": round(sum(r["prompt_tokens"] or 0 for r in rows) / n) if n else None,  # avg prompt size
            "mean_gen": round(sum(r["gen_tokens"] or 0 for r in rows) / n) if n else None,  # avg completion size
            "mean_reasoning": round(sum(reasoning) / len(reasoning)) if reasoning else None,  # avg thinking tokens, when present
            "rows_with_reasoning": len(reasoning),  # how many rows actually reported reasoning tokens
            "total_in": sum(r["prompt_tokens"] or 0 for r in rows),  # running total of prompt tokens spent
            "total_gen": sum(r["gen_tokens"] or 0 for r in rows),  # running total of generated tokens
            "hit_max": sum(bool(r.get("hit_max_tokens")) for r in rows),  # rows that got truncated at the token cap
            "error_lines": sum(1 for _ in open(errs, encoding="utf-8")) if errs.exists() else 0,  # lines logged to this model's error file
        })
    return pd.DataFrame(recs).set_index("model")


COLS = ["game_id", "round_id", "role_id", "predicted", "ground_truth", "parsed", "output",
        "chain_mode", "chain_rounds", "prompt_tokens", "gen_tokens", "hit_max_new_tokens", "seconds",
        "reasoning_tokens", "finish_reason", "model_served", "utc"]


def score(alias, allow_partial=False):
    """Scores with the same arithmetic as the Qwen runs. allow_partial scores complete games only."""
    done = load_done(alias)
    keys = [(r["game_id"], int(r["round_id"])) for r in P.VAL_ROWS]
    missing = [k for k in keys if k not in done]
    if missing and not allow_partial:
        raise RuntimeError(f"{alias}: {len(missing)} of 100 rows missing; finish the run or pass allow_partial=True")
    if allow_partial:
        whole = {g for g in P.GAMES if all((g, int(r["round_id"])) in done for r in P.ROWS_BY_GAME[g])}
        keys = [k for k in keys if k[0] in whole]
    records = []
    for k in keys:
        r = dict(done[k])
        r["hit_max_new_tokens"] = r.get("hit_max_tokens")
        r["reasoning_tokens"] = reasoning_tokens(r)
        records.append(r)
    s = P.score_stage(alias, records)
    s.update({"run": alias, "model_id": REGISTRY[alias]["id"], "chain": "self", "no_think": NO_THINK})
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records)[COLS].to_csv(OUT / f"eval_predictions_{alias}.csv", index=False)
    return s


def score_all(allow_partial=False):
    rows = []
    for a in REGISTRY:
        if a in RETIRED or not load_done(a):
            continue
        try:
            rows.append(score(a, allow_partial))
        except RuntimeError as exc:
            log(str(exc))
    df = pd.DataFrame(rows)
    if len(df):
        df.to_csv(OUT / "eval_summary.csv", index=False)
    return df


def show_prompt(game_id, round_id, alias=None):
    """Prints what a model is sent for one row. With an alias, prior traces come from its stored outputs."""
    rows = P.ROWS_BY_GAME[game_id]
    done = load_done(alias) if alias else {}
    chain = []
    for r in rows:
        if int(r["round_id"]) == int(round_id):
            system, user = P.messages_for(r, chain[-P.CHAIN_WINDOW:], no_think=NO_THINK,
                                          neutral=REGISTRY.get(alias, {}).get("neutral", False))
            print(f"=== SYSTEM ===\n{system}\n\n=== USER ({len(user):,} chars) ===\n{user}")
            return system, user
        k = (game_id, int(r["round_id"]))
        if k in done:
            chain.append({"round_id": k[1], "reasoning_gold": P._chain_text(done[k]["output"])})
    raise KeyError(f"{game_id} has no round {round_id}; rounds are {[int(r['round_id']) for r in rows]}")
