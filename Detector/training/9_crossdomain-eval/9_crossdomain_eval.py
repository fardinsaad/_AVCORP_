#!/usr/bin/env python3
"""Trained detectors on Avalon-NLU (6 players) with the minimal prompt; prior traces are the model's own outputs."""
import argparse
import datetime as _dt
import json
import os
import re
import sys
import time

RUNS_ROOT = "/share/mpsingh/fsaad/avcorp-runs"
RUNS = {  # key -> run folder and prompt variant
    "run4_1": {"folder": "base-grpo-h100-g16",           "variant": "trace"},
    "run4_2": {"folder": "base-grpo-h100-g16-r2",        "variant": "trace"},
    "run4_3": {"folder": "base-grpo-h100-g16-r3",        "variant": "trace"},
    "run5_1": {"folder": "base-reward-grpo-h100-g16",    "variant": "trace"},
    "run5_2": {"folder": "base-reward-grpo-h100-g16-r2", "variant": "trace"},
    "run5_3": {"folder": "base-reward-grpo-h100-g16-r3", "variant": "trace"},
    "run6":   {"folder": "tactic-grpo-h100-g16",         "variant": "tactic"},
    "run7":   {"folder": "tactic-reward-grpo-h100-g16",  "variant": "tactic"},
}
MAX_NEW_TOKENS = 2048


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, choices=sorted(RUNS))
    ap.add_argument("--stages", default="sft,grpo", help="comma list of base,sft,grpo")
    ap.add_argument("--dataset", default="avalon-nlu", choices=["avalon-nlu"])
    ap.add_argument("--data-csv", default=None, help="default: avalon_nlu.csv next to this script")
    ap.add_argument("--out-root", default=None, help=f"default: {RUNS_ROOT}/crossdomain/<dataset>")
    ap.add_argument("--dry-run", action="store_true", help="no model; completion = empty string")
    a = ap.parse_args()
    a.stages = [s.strip() for s in a.stages.split(",") if s.strip()]
    bad = [s for s in a.stages if s not in ("base", "sft", "grpo")]
    if bad:
        ap.error(f"unknown stage(s): {bad}")
    if a.data_csv is None:
        a.data_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "avalon_nlu.csv")
    if a.out_root is None:
        a.out_root = f"{RUNS_ROOT}/crossdomain/{a.dataset}"
    return a


ARGS = parse_args()
RUN = RUNS[ARGS.run]
RUN_DIR = os.path.join(RUNS_ROOT, RUN["folder"])
TACTIC = RUN["variant"] == "tactic"
OUT_DIR = os.path.join(ARGS.out_root, *(["dry-run"] if ARGS.dry_run else []), RUN["folder"])
os.makedirs(OUT_DIR, exist_ok=True)
_RUNLOG = os.path.join(OUT_DIR, "progress.log")


def _runlog(msg: str):
    _line = f"{_dt.datetime.now().isoformat(timespec='seconds')}  {msg}"
    with open(_RUNLOG, "a") as _fh:
        _fh.write(_line + "\n")
    print(_line, flush=True)


if not ARGS.dry_run:  # same setup as the training notebook, without the pip install
    import site
    SHARE = "/share/mpsingh/fsaad"
    USER_BASE = os.path.join(SHARE, "python-packages")
    os.environ["PYTHONUSERBASE"] = USER_BASE
    os.environ["HF_HOME"] = os.path.join(SHARE, "hf_cache")
    os.environ["PIP_CACHE_DIR"] = os.path.join(SHARE, "pip", "cache")
    _user_site = os.path.join(
        USER_BASE, "lib", f"python{sys.version_info.major}.{sys.version_info.minor}", "site-packages"
    )
    site.addsitedir(_user_site)
    if _user_site in sys.path:
        sys.path.remove(_user_site)
    sys.path.insert(0, _user_site)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import importlib
    for _name, _want in {"transformers": "4.56.1", "peft": "0.14.0", "accelerate": "1.4.0"}.items():
        _got = getattr(importlib.import_module(_name), "__version__", "unknown")
        if _got != _want:
            raise SystemExit(f"FATAL: {_name} {_got} loaded, training used {_want}")

import pandas as pd

df = pd.read_csv(ARGS.data_csv)


df["reasoning_gold"] = df["reasoning_gold"].fillna("")  # no gold traces outside AVCORP


def build_public_history_window(df: pd.DataFrame, window: int = 3) -> dict:  # cell 8
    lookup = {}
    for game_id, group in df.groupby('game_id'):
        sorted_group = group.sort_values('round_id')
        rows = list(sorted_group.iterrows())
        for i, (_, row) in enumerate(rows):
            rid = int(row['round_id'])
            start = max(0, i - (window - 1))
            parts = [f"[R{int(r['round_id'])}] {r['public_history'].strip()}" for _, r in rows[start:i + 1]]
            lookup[(game_id, rid)] = "\n".join(parts)
    return lookup


CHAIN_WINDOW = 2
public_history_lookup = build_public_history_window(df)

val_games = df['game_id'].unique()  # every row of every game; no split
val_df = df
assert len(val_games) == 20 and len(val_df) == 82, (len(val_games), len(val_df))

ALL_PLAYERS_FULL = {"P1", "P2", "P3", "P4", "P5", "P6"}  # cell 12 / tactic cell 14, edited for 6 players
N_OTHERS_WORD = "five"
GAME_SIZE = "6-player"
RULES = (  # every rule from Avalon-NLU's game code (avalon_online/config.json, src/game.py)
    "Game rules: there are six players, four Good and two Evil. "
    "Merlin (Good) knows who the two Evil players are. "
    "Percival (Good) sees Merlin and Morgana but cannot tell which is which. "
    "The two Loyal Servants (Good) have no special knowledge; you are one of them. "
    "Morgana and the Assassin (Evil) know each other. "
    "Team sizes for Quests 1 to 5 are 2, 3, 4, 3, and 4. "
    "A team goes on its quest only if more than half of the six votes are Yes, "
    "and a single Fail card fails the quest. "
    "If Good completes three quests, the Assassin names one player as Merlin, "
    "and Evil wins if that guess is right. "
)

TACTIC_VOCAB = [
    "Bayesian hedging", "Bold falsehoods", "Bullshitting", "Cherry-picking",
    "Confusion tactics", "Defensive disclosure", "Deflection", "Evidence sharing",
    "Face-saving blather", "False consensus", "Gaslighting", "Half-truths",
    "Hard lying", "Honest mistake", "Misframing", "Misleading emphasis",
    "Noisy sincerity", "Obfuscatory nonsense", "Omission to save face",
    "Over-explaining", "Overstated contribution", "Perspective-taking",
    "Pragmatic silence", "Rational justification", "Rationalized error",
    "Reverse accusations", "Self-deception", "Self-justification",
    "Sincere false belief", "Strategic fabrication", "Strategic omission",
    "Strategic uncertainty", "Technobabble", "Truthful misdirection",
    "Vague filler", "Weaponized candor", "Withholding info (WIFOM)",
]


def _other_players(role_id):
    return sorted(ALL_PLAYERS_FULL - {role_id})


def build_system_prompt(role_id: str) -> str:
    others = ", ".join(_other_players(role_id))
    return (
        f"You are the Investigator in a {GAME_SIZE} Avalon game. "
        f"Your player ID is {role_id} and your alignment is Good. "
        f"You must reason about the following {N_OTHERS_WORD} players: {others}. "
        f"{RULES}"
        "Reason carefully from the evidence and produce only valid JSON. "
        "/no_think"
    )


_SCHEMA_HEAD_TRACE = (
    "Given the public history and discussion log, produce a structured JSON reasoning trace "
    "with exactly the following six keys.\n\n"
)
_SCHEMA_HEAD_TACTIC = (
    "Given the public history and discussion log, produce a structured JSON reasoning trace "
    "with exactly the following seven keys.\n\n"
)
_SCHEMA_BODY = (
    f'abduction: a list of {N_OTHERS_WORD} objects, one per other player. '
    'Each object has the fields player, good_expl, evil_expl, and choice. '
    'The choice field must be either Good or Evil.\n\n'
    f'suspicion: a list of {N_OTHERS_WORD} objects, one per other player. '
    'Each object has the fields player, level, and reason. '
    'Valid levels are Minimal, Mild, Moderate, Strong, and Extreme.\n\n'
    'depth: an integer, either 1 or 2. '
    'Use 2 only when at least one player has Strong or Extreme suspicion.\n\n'
    'beliefs: a list of belief objects. '
    f'Level 1 beliefs must exist for all {N_OTHERS_WORD} other players. '
    'Level 2 beliefs may exist only for players with Strong or Extreme suspicion. '
    'Each belief object has the fields level, player, and content.\n\n'
    'statement: a string containing your public statement for this round.\n\n'
    'deduction: a list of player IDs you believe are Evil, for example ["P3", "P4"].\n\n'
)
_SCHEMA_TACTICS = (
    f'tactics: a list of {N_OTHERS_WORD} objects, one per other player. '
    'Each object has the fields player and tactic. '
    f'The tactic field must be exactly one of: {", ".join(TACTIC_VOCAB)}.\n\n'
)
_SCHEMA_TAIL = (
    'Self-Knowledge Elimination: you know your own alignment and actions with certainty. '
    'If a quest\'s outcome contradicts your known-Good behavior, eliminate suspects using the '
    'team\'s known composition, confidence scaling with how few possibilities remain. One '
    'remaining suspect means a definitive conclusion (Extreme suspicion, Evil abduction choice), '
    'not just strong suspicion. Example: a failed 2-player quest you were on makes the other '
    'player definitively Evil, since you always submit Pass.\n\n'
    'Output only the JSON object. Do not include any explanation outside the JSON.'
)
if TACTIC:
    SCHEMA_REMINDER = _SCHEMA_HEAD_TACTIC + _SCHEMA_BODY + _SCHEMA_TACTICS + _SCHEMA_TAIL
else:
    SCHEMA_REMINDER = _SCHEMA_HEAD_TRACE + _SCHEMA_BODY + _SCHEMA_TAIL


def format_user_turn(row: dict, prior_chain: list = None, public_history_window: str = None) -> str:
    chain_section = ""
    if prior_chain:
        parts = [
            f"Round {entry['round_id']} Reasoning Trace\n{entry['reasoning_gold']}"
            for entry in prior_chain
        ]
        n = len(prior_chain)
        label = "last round" if n == 1 else f"last {n} rounds"
        chain_section = f"\nPrior Round Reasoning Traces ({label})\n" + "\n\n".join(parts) + "\n"
    history_text = public_history_window if public_history_window is not None else row['public_history'].strip()
    return (
        f"Public History\n{history_text}\n"
        f"{chain_section}"
        f"\nDiscussion Log\n{row['discussion_log'].strip()}\n\n"
        f"{SCHEMA_REMINDER}\n\n"
        f"Produce the JSON reasoning trace:"
    )


def _try_parse(completion):
    if isinstance(completion, list):
        completion = completion[-1].get("content", "") if completion and isinstance(completion[-1], dict) else ""
    elif isinstance(completion, dict):
        completion = completion.get("content", "")
    if not isinstance(completion, str):
        return None
    completion = re.sub(r"```(?:json)?\s*", "", completion).strip().rstrip("`")
    try:
        return json.loads(completion)
    except Exception:
        pass
    match = re.search(r"(\{.*\})", completion, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass
    return None


def _chain_text(completion: str) -> str:  # same fence stripping as _try_parse, no re-serialisation
    return re.sub(r"```(?:json)?\s*", "", completion).strip().rstrip("`")


def _prf(tp: int, fp: int, fn: int):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    return p, r, (2 * p * r / (p + r) if (p + r) else 0.0)


def _deduction_from_output(text: str, role_id: str):
    parsed = _try_parse(text)
    others = set(_other_players(role_id))
    if not isinstance(parsed, dict):
        return set(), False
    raw = parsed.get("deduction")
    if not isinstance(raw, list):
        return set(), True
    return {p for p in raw if p in others}, True


def score_stage(stage_name: str, records: list) -> dict:  # cell 40 arithmetic
    tp = fp = fn = 0
    per_row_f1, parse_failures, by_round = [], 0, {}
    for rec in records:
        pred, ok = _deduction_from_output(rec["output"], rec["role_id"])
        if not ok:
            parse_failures += 1
        others = set(_other_players(rec["role_id"]))
        gt = {p for p, v in json.loads(rec["player_roles"]).items() if v == "Evil" and p in others}
        t, f_pos, f_neg = len(pred & gt), len(pred - gt), len(gt - pred)
        tp += t; fp += f_pos; fn += f_neg
        acc = by_round.setdefault(int(rec["round_id"]), [0, 0, 0])
        acc[0] += t; acc[1] += f_pos; acc[2] += f_neg
        prec = t / len(pred) if pred else 0.0
        rec_ = t / len(gt) if gt else 0.0
        per_row_f1.append(2 * prec * rec_ / (prec + rec_) if prec + rec_ else 0.0)
        rec["predicted"] = " ".join(sorted(pred))
        rec["ground_truth"] = " ".join(sorted(gt))
        rec["parsed"] = ok
    micro_p, micro_r, micro_f1 = _prf(tp, fp, fn)
    rec_df = pd.DataFrame(records)
    last = rec_df.loc[rec_df.groupby("game_id")["round_id"].idxmax()]
    g_tp = g_fp = g_fn = 0
    exact = hit = 0
    for _, g in last.iterrows():
        pred = set(g["predicted"].split()) if g["predicted"] else set()
        gt = set(g["ground_truth"].split()) if g["ground_truth"] else set()
        g_tp += len(pred & gt); g_fp += len(pred - gt); g_fn += len(gt - pred)
        exact += pred == gt
        hit += bool(pred & gt)
    n_games = len(last)
    g_p, g_r, g_f1 = _prf(g_tp, g_fp, g_fn)
    return {
        "stage": stage_name, "n": len(records), "parse_failures": parse_failures,
        "precision": micro_p, "recall": micro_r, "f1": micro_f1,
        "macro_f1": sum(per_row_f1) / len(per_row_f1) if per_row_f1 else 0.0,
        "games": n_games, "game_exact_match": exact / n_games if n_games else 0.0,
        "game_exact_count": exact, "game_hit_count": hit, "game_f1": g_f1,
        **{f"f1_R{rnd}": _prf(*by_round[rnd])[2] for rnd in sorted(by_round)},
        "hit_max_new_tokens": int(sum(bool(r.get("hit_max_new_tokens")) for r in records)),
    }


model = tokenizer = None
GPU_NAME = "dry-run"
if not ARGS.dry_run:  # cell 14 model load, plus the run's two saved adapters
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    if not (torch.cuda.is_available() and torch.cuda.is_bf16_supported(including_emulation=False)):
        raise SystemExit("FATAL: no native bf16 on this GPU")
    COMPUTE_DTYPE = torch.bfloat16
    GPU_NAME = torch.cuda.get_device_name(0)
    MODEL_PATH = snapshot_download("Qwen/Qwen3-8B", local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    _base = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=COMPUTE_DTYPE, attn_implementation="sdpa",
        device_map={"": 0}, trust_remote_code=True,
    )
    GRPO_DIR = os.path.join(RUN_DIR, "grpo-qwen3-avalon")
    SFT_DIR = os.path.join(RUN_DIR, "sft-warmup")
    for _d in (GRPO_DIR, SFT_DIR):
        if not os.path.isfile(os.path.join(_d, "adapter_model.safetensors")):
            raise SystemExit(f"FATAL: no adapter_model.safetensors in {_d}")
    model = PeftModel.from_pretrained(_base, GRPO_DIR, adapter_name="grpo")
    model.load_adapter(SFT_DIR, adapter_name="sft")
    model.eval()
    _runlog(f"model {MODEL_PATH} | gpu {GPU_NAME} | sdpa bf16 | grpo={GRPO_DIR} sft={SFT_DIR}")


def run_inference(row_dict: dict, prior_chain: list = None, public_history_window: str = None,
                  max_new_tokens: int = MAX_NEW_TOKENS):  # cell 38, plus token counts
    messages = [
        {"role": "system", "content": build_system_prompt(row_dict['role_id'])},
        {"role": "user",   "content": format_user_turn(row_dict, prior_chain, public_history_window)},
    ]
    if ARGS.dry_run:
        return "", None, None
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = outputs[0][inputs['input_ids'].shape[1]:]
    return (tokenizer.decode(generated, skip_special_tokens=True),
            int(inputs['input_ids'].shape[1]), int(generated.shape[0]))


def _load_done(path):
    done = {}
    if os.path.exists(path):
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    done[(r["game_id"], int(r["round_id"]))] = r
    return done


def evaluate_stage(stage_name: str) -> dict:
    jsonl = os.path.join(OUT_DIR, f"pred_{stage_name}.jsonl")
    done = _load_done(jsonl)
    if done:
        _runlog(f"{stage_name}: resuming, {len(done)} rows already on disk")
    val_rows = [r.to_dict() for _, r in val_df.iterrows()]
    order = {g: i for i, g in enumerate(dict.fromkeys(r["game_id"] for r in val_rows))}  # game by game, rounds in order
    gen_rows = sorted(val_rows, key=lambda r: (order[r["game_id"]], int(r["round_id"])))
    own = {}
    n_new = 0
    t_stage = time.time()
    with open(jsonl, "a") as fh:
        for i, row in enumerate(gen_rows, 1):
            key = (row["game_id"], int(row["round_id"]))
            chain = list(own.get(row["game_id"], [])[-CHAIN_WINDOW:])
            if key in done:
                out = done[key]["output"]
            else:
                t0 = time.time()
                out, n_prompt, n_gen = run_inference(row, chain, public_history_lookup.get(key))
                rec = {
                    "game_id": row["game_id"], "round_id": int(row["round_id"]),
                    "role_id": row["role_id"], "player_roles": row["player_roles"],
                    "chain_mode": "self",
                    "chain_rounds": " ".join(str(c["round_id"]) for c in chain),
                    "prompt_tokens": n_prompt, "gen_tokens": n_gen,
                    "hit_max_new_tokens": (n_gen is not None and n_gen >= MAX_NEW_TOKENS),
                    "seconds": round(time.time() - t0, 1),
                    "output": out,
                }
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
                done[key] = rec
                n_new += 1
            own.setdefault(row["game_id"], []).append(
                {"round_id": int(row["round_id"]), "reasoning_gold": _chain_text(out)}
            )
            if i % 10 == 0:
                _runlog(f"{stage_name}: {i}/{len(gen_rows)} rows "
                        f"({n_new} generated this session, {time.time() - t_stage:.0f}s)")
    records = [dict(done[(r["game_id"], int(r["round_id"]))]) for r in val_rows]  # CSV row order
    summary = score_stage(stage_name, records)
    summary.update({"run": ARGS.run, "run_folder": RUN["folder"], "dataset": ARGS.dataset, "gpu": GPU_NAME,
                    "seconds_this_session": round(time.time() - t_stage, 1)})
    cols = ["game_id", "round_id", "role_id", "predicted", "ground_truth", "parsed", "output",
            "chain_mode", "chain_rounds", "prompt_tokens", "gen_tokens", "hit_max_new_tokens",
            "seconds"]
    pd.DataFrame(records)[cols].to_csv(os.path.join(OUT_DIR, f"eval_predictions_{stage_name}.csv"),
                                       index=False)
    return summary


def main():
    _runlog(f"=== start | {ARGS.run} ({RUN['folder']}, {RUN['variant']}) | stages={ARGS.stages} | "
            f"dataset={ARGS.dataset} | dry_run={ARGS.dry_run} | out={OUT_DIR}")
    _runlog(f"data {ARGS.data_csv} | games {len(val_games)} rows {len(val_df)}")
    summ_path = os.path.join(OUT_DIR, "eval_summary.csv")
    results = pd.read_csv(summ_path).to_dict("records") if os.path.exists(summ_path) else []
    for stage in ARGS.stages:
        t = time.time()
        if stage == "base":
            if ARGS.dry_run:
                s = evaluate_stage(stage)
            else:
                with model.disable_adapter():
                    s = evaluate_stage(stage)
        else:
            if not ARGS.dry_run:
                model.set_adapter(stage)
            s = evaluate_stage(stage)
        results = [r for r in results if r.get("stage") != stage] + [s]
        pd.DataFrame(results).to_csv(summ_path, index=False)
        _runlog(f"{stage}: f1={s['f1']:.3f} macro_f1={s['macro_f1']:.3f} "
                f"game_end_exact={s['game_exact_count']}/{s['games']} ({s['game_exact_match']:.1%}) "
                f"game_f1={s['game_f1']:.3f} parse_fail={s['parse_failures']} "
                f"hit_max_tokens={s['hit_max_new_tokens']} | {time.time() - t:.0f}s")
    if not ARGS.dry_run:
        _runlog(f"peak GPU memory {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
    _runlog("=== done")


if __name__ == "__main__":
    main()
