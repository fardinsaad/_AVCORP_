#!/usr/bin/env python3
"""
Builds grpo_results.xlsx from training run folders.

Usage:
    python analyze_results.py <run_dir> [--notes "..."] [--run-tag TAG]
    python analyze_results.py <run_dir> --dry-run        # print, write nothing
    python analyze_results.py --rebuild <run_dir> [<run_dir> ...]   # fresh workbook

Reads run_metadata.json, the SFT and GRPO log histories and eval_predictions_*.csv of each run.
The MT tabs are written by add_model_trace.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

ALL_PLAYERS = {"P1", "P2", "P3", "P4", "P5"}
STAGES = ("base", "sft", "grpo")

# (name, num_generations, max_steps) of the known GRPO recipes, used to label a run
GRPO_RECIPES = [
    ("grpo-smoke", 8, 20),
    ("grpo-l40", 8, 900),
    ("grpo-h100", 8, 225),
    ("grpo-h100-g16", 16, 450),
    ("grpo-h100-g16-2ep", 16, 900),
    ("grpo-h100-g32", 32, 450),
]

FONT = "Arial"
HEADER_FONT = Font(name=FONT, color="FFFFFF", size=10, bold=True)
HEADER_FILL = PatternFill("solid", fgColor="1F4E78")
BODY = Font(name=FONT, size=10)
THIN = Side(style="thin", color="BFBFBF")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)

README_LINES = [
    "AVCORP detector runs -- results workbook",
    "",
    "Written by results/analyze_results.py (Runs, Reward Detail, Per Round, Game End, Baselines)",
    "and results/add_model_trace.py (the three MT tabs). Edit only the notes column in Runs.",
    "",
    "TABS AND COLUMNS",
    "  Runs           one row per run: recipe, gpu, GRPO steps, minutes and peak memory per phase;",
    "                 per stage (base / sft / grpo): row-level f1, game_end_exact, game_end_f1,",
    "                 mean prediction size; first and last reward; caveats; notes; source_run_dir.",
    "  Reward Detail  one row per logged GRPO step: total reward, each reward component, loss,",
    "                 grad_norm, mean completion length, share of prompt groups with zero reward spread.",
    "  Per Round      one row per (run, stage, round): n_rows, precision, recall, f1, exact_match.",
    "  Game End       one row per (run, stage, ending round R3 / R4 / R5 / ALL): n_games, precision,",
    "                 recall, f1, exact_match, n_exact. Stage gold_trace = the gold traces on the same games.",
    "  Baselines      scores on the same rows for accuse_all_4, random_k_of_4 and predict_nothing.",
    "  MT Runs        one row per (run, stage) of the MODEL-TRACE evaluation: n_rows, n_games,",
    "                 parse_failures, hit_max_new_tokens, f1, macro_f1, exact_match, mean_pred_size,",
    "                 game_end_exact, game_end_n_exact, game_end_f1; the gold-trace values of f1,",
    "                 game_end_exact and game_end_f1 from Runs (gold_*) and new minus gold (delta_*);",
    "                 gpu; caveats; source_dir.",
    "  MT Per Round   Per Round, for the model-trace evaluation.",
    "  MT Game End    Game End, for the model-trace evaluation.",
    "",
    "CAVEATS",
    "  Gold vs model trace. Runs, Per Round, Game End and Baselines come from the training",
    "  notebooks' evaluation, whose prompts carry the two previous rounds' GOLD traces (gold",
    "  deductions; tactic runs also gold tactics). The MT tabs carry the model's own earlier",
    "  outputs instead. Read the two side by side; never pool them.",
    "  Base is evaluated once per prompt in the MT tabs: 'base (trace prompt)', 'base (tactic prompt)'.",
    "  game_end_exact is the headline metric (one row per game, scored at its final round).",
    "  25 validation games: one game is 4 percentage points, so quote n_exact with percentages.",
    "  R4-ending games are the hardest (the gold traces reach 60% exact there). Compare a run with",
    "  the gold_trace rows for its own games, not with the 250-game figure (93.2% / 0.962).",
    "  reward-grpo-h100-g16 (5_1): its sft stage is not the SFT its GRPO started from; never pair them.",
    "  eval_limit < 100 is a Round-1-only smoke check, never a comparable score.",
    "  status != complete: timings cover only the segment that ran.",
    "  Reward Detail: the ..._or_choice and ..._or_calibration columns hold a different reward",
    "  function in 'reward-' runs; do not compare them across the two reward sets.",
    "  Reward values are a training signal and are never reported as results.",
]

RUN_COLS = [
    ("run_tag", 20), ("status", 12), ("date", 12), ("sft_recipe", 12), ("grpo_recipe", 16),
    ("gpu", 17), ("grpo_num_generations", 9), ("grpo_max_steps", 9), ("grpo_steps_done", 9),
    ("sft_min", 9), ("sft_peak_mem_gb", 9), ("grpo_min", 10), ("grpo_peak_mem_gb", 10),
    ("total_train_hours", 10), ("eval_limit", 9), ("eval_min", 9), ("eval_peak_mem_gb", 10),
    ("eval_rows", 9), ("eval_games", 9), ("eval_rounds_covered", 14),
    ("base_f1", 9), ("sft_f1", 9), ("grpo_f1", 9), ("delta_f1_grpo_vs_sft", 11),
    ("base_game_end_exact", 11), ("sft_game_end_exact", 11), ("grpo_game_end_exact", 11),
    ("base_game_end_f1", 11), ("sft_game_end_f1", 11), ("grpo_game_end_f1", 11),
    ("grpo_mean_pred_size", 11), ("sft_mean_pred_size", 11),
    ("reward_first", 10), ("reward_last", 10), ("parse_failures_total", 10),
    ("caveats", 52), ("notes", 52), ("source_run_dir", 30),
]
# renamed columns keep their positions, so older rows stay aligned
RD_COLS = [("run_tag", 20), ("step", 8), ("epoch", 9), ("reward", 10),
           ("reward_schema_validity", 13), ("reward_abduction_completeness_or_choice", 20),
           ("reward_deduction_accuracy", 14), ("reward_gated_depth", 13),
           ("reward_suspicion_consistency_or_calibration", 20), ("reward_tactic_accuracy", 13),
           ("loss", 10), ("grad_norm", 10), ("completions_mean_length", 13),
           ("frac_reward_zero_std", 12)]
PR_COLS = [("run_tag", 20), ("stage", 8), ("round_id", 9), ("n_rows", 8),
           ("precision", 10), ("recall", 10), ("f1", 10), ("exact_match", 11)]
GE_COLS = [("run_tag", 20), ("stage", 11), ("end_round", 10), ("n_games", 9),
           ("precision", 10), ("recall", 10), ("f1", 10), ("exact_match", 11),
           ("n_exact", 9)]
BL_COLS = [("run_tag", 20), ("scope", 12), ("baseline", 16), ("n", 8),
           ("precision", 10), ("recall", 10), ("f1", 10), ("exact_match", 11)]
SHEETS = {"Runs": RUN_COLS, "Reward Detail": RD_COLS, "Per Round": PR_COLS,
          "Game End": GE_COLS, "Baselines": BL_COLS}
PCT = {"f1", "precision", "recall", "exact_match", "base_f1", "sft_f1", "grpo_f1",
       "delta_f1_grpo_vs_sft", "base_game_end_exact", "sft_game_end_exact",
       "grpo_game_end_exact", "base_game_end_f1", "sft_game_end_f1", "grpo_game_end_f1"}



# two rewards were renamed for run 5; each column lists every name that fills it
REWARD_ALIASES = {
    "reward_schema_validity": ("reward_schema_validity",),
    "reward_abduction_completeness_or_choice": ("reward_abduction_completeness",
                                                "reward_abduction_choice"),
    "reward_deduction_accuracy": ("reward_deduction_accuracy",),
    "reward_gated_depth": ("reward_gated_depth",),
    "reward_suspicion_consistency_or_calibration": ("reward_suspicion_consistency",
                                                    "reward_suspicion_calibration"),
    "reward_tactic_accuracy": ("reward_tactic_accuracy",),
}


def reward_mean(h: dict, *fn_names: str):
    """Mean of one reward at one step under whichever of `fn_names` the run logged; None if none."""
    for fn in fn_names:
        v = h.get(f"rewards/{fn}/mean")
        if v is not None:
            return v
    return None

# ----------------------------------------------------------------- metric helpers
def prf(tp: int, fp: int, fn: int):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    return p, r, (2 * p * r / (p + r) if (p + r) else 0.0)


def _sets(row):
    pred = set(str(row["predicted"]).split()) if pd.notna(row["predicted"]) and str(row["predicted"]).strip() else set()
    gt = set(str(row["ground_truth"]).split()) if pd.notna(row["ground_truth"]) and str(row["ground_truth"]).strip() else set()
    return pred, gt


def score_frame(df: pd.DataFrame) -> dict:
    """Micro P/R/F1, macro F1 and exact-match over whatever rows are given."""
    tp = fp = fn = 0
    exact = 0
    per_row_f1 = []
    sizes = []
    for _, row in df.iterrows():
        pred, gt = _sets(row)
        tp += len(pred & gt); fp += len(pred - gt); fn += len(gt - pred)
        sizes.append(len(pred))
        if pred == gt:
            exact += 1
        p, r, f = prf(len(pred & gt), len(pred - gt), len(gt - pred))
        per_row_f1.append(f)
    p, r, f1 = prf(tp, fp, fn)
    n = len(df)
    return {"n": n, "precision": p, "recall": r, "f1": f1,
            "macro_f1": sum(per_row_f1) / n if n else 0.0,
            "exact_match": exact / n if n else 0.0,
            "mean_pred_size": sum(sizes) / n if n else 0.0}


def game_end_frame(df: pd.DataFrame) -> pd.DataFrame:
    """One row per game: that game's highest round_id."""
    if df.empty:
        return df
    return df.loc[df.groupby("game_id")["round_id"].idxmax()]


def baselines(df: pd.DataFrame) -> list[dict]:
    """Reference scores on the same rows (name everyone, random pair)."""
    out = []

    # accuse-all-4: predict every candidate other than the investigator.
    tp = fp = fn = 0
    exact = 0
    for _, row in df.iterrows():
        _, gt = _sets(row)
        others = ALL_PLAYERS - {row["role_id"]}
        pred = others
        tp += len(pred & gt); fp += len(pred - gt); fn += len(gt - pred)
        exact += int(pred == gt)
    p, r, f = prf(tp, fp, fn)
    out.append({"baseline": "accuse_all_4", "n": len(df), "precision": p, "recall": r,
                "f1": f, "exact_match": exact / len(df) if len(df) else 0.0})

    # random |gt|-of-|others|, expectation rather than a sampled draw.
    e_tp = 0.0
    e_exact = 0.0
    n_slots = 0
    for _, row in df.iterrows():
        _, gt = _sets(row)
        k = len(gt)
        m = len(ALL_PLAYERS - {row["role_id"]})
        if k == 0 or m == 0:
            continue
        e_tp += k * (k / m)
        n_slots += k
        e_exact += 1.0 / math.comb(m, k)
    p = e_tp / n_slots if n_slots else 0.0
    out.append({"baseline": f"random_k_of_4", "n": len(df), "precision": p, "recall": p,
                "f1": p, "exact_match": e_exact / len(df) if len(df) else 0.0})

    out.append({"baseline": "predict_nothing", "n": len(df), "precision": 0.0,
                "recall": 0.0, "f1": 0.0, "exact_match": 0.0})
    return out


# ---- game-end detail: one row per (stage, ending round), plus ALL and the gold traces on the same games
GOLD_SELFTEST = Path(__file__).resolve().parent.parent / "Extra" / "trace-gold-analysis" / "eval_predictions_golden_selftest.csv"


def game_end_rows(tag: str, stage: str, ge_df: pd.DataFrame) -> list[dict]:
    """One row per ending round for the given one-row-per-game frame, plus an ALL row."""
    out = []
    if ge_df.empty:
        return out
    buckets = sorted(int(r) for r in ge_df["round_id"].unique())
    for r in buckets + ["ALL"]:
        sub = ge_df if r == "ALL" else ge_df[ge_df["round_id"].astype(int) == r]
        if sub.empty:
            continue
        s = score_frame(sub)
        out.append({"run_tag": tag, "stage": stage,
                    "end_round": "ALL" if r == "ALL" else f"R{r}",
                    "n_games": s["n"], "precision": s["precision"], "recall": s["recall"],
                    "f1": s["f1"], "exact_match": s["exact_match"],
                    "n_exact": int(round(s["exact_match"] * s["n"]))})
    return out


def gold_game_end_rows(tag: str, game_ids, path: Path = GOLD_SELFTEST):
    """Gold-trace verdicts on this run's validation games. Returns (rows, caveat_or_None)."""
    path = Path(path)
    if not path.exists():
        return [], f"gold self-test not found at {path.name}; no ceiling row written"
    g = pd.read_csv(path)
    need = {"game_id", "is_final_round", "game_end_round", "predicted_evil", "true_evil"}
    if not need.issubset(g.columns):
        return [], f"gold self-test {path.name} is missing columns; no ceiling row written"
    g = g[g["game_id"].isin(list(game_ids)) & g["is_final_round"].astype(bool)].copy()
    if g.empty:
        return [], "gold self-test covers none of this run's games; no ceiling row written"
    g = g.rename(columns={"predicted_evil": "predicted", "true_evil": "ground_truth"})
    g["round_id"] = g["game_end_round"].astype(int)
    missing = set(game_ids) - set(g["game_id"])
    note = f"gold ceiling covers {len(g)}/{len(set(game_ids))} val games" if missing else None
    return game_end_rows(tag, "gold_trace", g), note


# ----------------------------------------------------------------- run-dir loading
def latest_checkpoint(grpo_dir: Path):
    cks = sorted(grpo_dir.glob("checkpoint-*"), key=lambda q: int(q.name.split("-")[-1])) if grpo_dir.is_dir() else []
    return cks[-1] if cks else None


def load_grpo_history(run_dir: Path):
    """The notebook's end-of-training dump, else the newest checkpoint."""
    direct = run_dir / "grpo_log_history.json"
    if direct.exists():
        # max_steps is not in the dump, so read it from the newest checkpoint
        ck = latest_checkpoint(run_dir / "grpo-qwen3-avalon")
        ms = None
        if ck and (ck / "trainer_state.json").exists():
            try:
                ms = json.loads((ck / "trainer_state.json").read_text()).get("max_steps")
            except Exception:
                ms = None
        return json.loads(direct.read_text()), "grpo_log_history.json", ms
    ck = latest_checkpoint(run_dir / "grpo-qwen3-avalon")
    if ck and (ck / "trainer_state.json").exists():
        st = json.loads((ck / "trainer_state.json").read_text())
        return st.get("log_history", []), f"{ck.name}/trainer_state.json", st.get("max_steps")
    return [], None, None


def guess_grpo_recipe(g, max_steps):
    for name, gg, steps in GRPO_RECIPES:
        if gg == g and steps == max_steps:
            return name
    return None


def analyse(run_dir: Path, run_tag: str | None, notes: str, gold_selftest: Path = GOLD_SELFTEST):
    meta_p = run_dir / "run_metadata.json"
    meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}

    # strip only "base-": trace and tactic runs share RUN_TAG values and differ by prefix
    tag = run_tag or run_dir.name
    if tag.startswith("base-"):
        tag = tag[len("base-"):]

    caveats = []
    if not meta:
        caveats.append("no run_metadata.json (job died before the Compute Budget cell)")

    # ---- GRPO reward history
    hist_raw, hist_src, ck_max_steps = load_grpo_history(run_dir)
    hist = [h for h in hist_raw if "reward" in h]
    sft_hist = []
    sft_p = run_dir / "sft_log_history.json"
    if sft_p.exists():
        sft_hist = [h for h in json.loads(sft_p.read_text()) if "loss" in h]
    else:
        caveats.append("no sft_log_history.json (run predates the SFT dump patch)")
    if hist_src and hist_src != "grpo_log_history.json":
        caveats.append(f"GRPO curve from {hist_src}; steps after the last checkpoint are not covered")
    if not hist:
        caveats.append("no GRPO reward history found")

    steps_done = hist[-1]["step"] if hist else None
    max_steps = meta.get("grpo_max_steps") or ck_max_steps

    # ---- evaluation
    stage_scores, per_round_rows, baseline_rows, ge_rows = {}, [], [], []
    all_frames = []
    parse_failures = 0
    for stage in STAGES:
        f = run_dir / f"eval_predictions_{stage}.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f)
        if df.empty:
            continue
        df["round_id"] = df["round_id"].astype(int)
        if "parsed" in df.columns:
            parse_failures += int((~df["parsed"].astype(bool)).sum())
        all_frames.append(df)
        row = score_frame(df)
        ge = score_frame(game_end_frame(df))
        row["game_end_exact"] = ge["exact_match"]
        row["game_end_f1"] = ge["f1"]
        row["games"] = len(game_end_frame(df))
        stage_scores[stage] = row
        ge_rows.extend(game_end_rows(tag, stage, game_end_frame(df)))
        for rnd, sub in df.groupby("round_id"):
            s = score_frame(sub)
            per_round_rows.append({"run_tag": tag, "stage": stage, "round_id": int(rnd),
                                   "n_rows": s["n"], "precision": s["precision"],
                                   "recall": s["recall"], "f1": s["f1"],
                                   "exact_match": s["exact_match"]})

    missing = [s for s in STAGES if s not in stage_scores]
    if missing:
        caveats.append(f"no eval_predictions for: {', '.join(missing)}")
    if not stage_scores and (run_dir / "eval_summary.csv").exists():
        caveats.append("eval_summary.csv exists but no eval_predictions_*.csv; metrics not recomputable")

    if all_frames:
        ref = all_frames[0]
        for scope, frame in (("row_level", ref), ("game_end", game_end_frame(ref))):
            for b in baselines(frame):
                baseline_rows.append({"run_tag": tag, "scope": scope, **b})
        gold_rows, gold_note = gold_game_end_rows(tag, game_end_frame(ref)["game_id"].tolist(),
                                                  gold_selftest)
        ge_rows = gold_rows + ge_rows
        if gold_note:
            caveats.append(gold_note)
        rounds = sorted(ref["round_id"].unique().tolist())
        eval_rounds = ",".join(f"R{r}" for r in rounds)
        eval_rows, eval_games = len(ref), len(game_end_frame(ref))
        if rounds == [1]:
            caveats.append("EVAL IS ROUND-1 ONLY -- game_end columns equal row-level; not a comparable score")
    else:
        eval_rounds, eval_rows, eval_games = None, None, None

    if meta.get("eval_limit") not in (None, "None") and meta:
        caveats.append(f"eval_limit={meta.get('eval_limit')} (< full 100-row val set)")

    complete = bool(stage_scores) and len(stage_scores) == 3 and steps_done and max_steps and steps_done >= max_steps
    status = "complete" if complete else "partial"

    def g(stage, key):
        return stage_scores.get(stage, {}).get(key)

    sft_s, grpo_s = meta.get("sft_seconds"), meta.get("grpo_seconds")
    row = {
        "run_tag": tag,
        "status": status,
        "date": (meta.get("recorded_utc") or "")[:10] or None,
        "sft_recipe": "sft-h100" if "H100" in (meta.get("gpu_name") or "") else ("sft-l40" if "L40" in (meta.get("gpu_name") or "") else None),
        "grpo_recipe": guess_grpo_recipe(meta.get("grpo_num_generations"), max_steps),
        "gpu": meta.get("gpu_name"),
        "grpo_num_generations": meta.get("grpo_num_generations"),
        "grpo_max_steps": max_steps,
        "grpo_steps_done": steps_done,
        "sft_min": round(sft_s / 60, 2) if sft_s else None,
        "sft_peak_mem_gb": meta.get("sft_peak_mem_gb"),
        "grpo_min": round(grpo_s / 60, 2) if grpo_s else None,
        "grpo_peak_mem_gb": meta.get("grpo_peak_mem_gb"),
        "total_train_hours": round(sum(x for x in (sft_s, grpo_s) if x) / 3600, 2) if (sft_s or grpo_s) else None,
        "eval_limit": meta.get("eval_limit", "n/a") if meta else None,
        "eval_min": round(meta["eval_seconds_total"] / 60, 2) if meta.get("eval_seconds_total") else None,
        "eval_peak_mem_gb": meta.get("eval_peak_mem_gb"),
        "eval_rows": eval_rows, "eval_games": eval_games, "eval_rounds_covered": eval_rounds,
        "base_f1": g("base", "f1"), "sft_f1": g("sft", "f1"), "grpo_f1": g("grpo", "f1"),
        "delta_f1_grpo_vs_sft": (g("grpo", "f1") - g("sft", "f1")) if (g("grpo", "f1") is not None and g("sft", "f1") is not None) else None,
        "base_game_end_exact": g("base", "game_end_exact"),
        "sft_game_end_exact": g("sft", "game_end_exact"),
        "grpo_game_end_exact": g("grpo", "game_end_exact"),
        "base_game_end_f1": g("base", "game_end_f1"),
        "sft_game_end_f1": g("sft", "game_end_f1"),
        "grpo_game_end_f1": g("grpo", "game_end_f1"),
        "grpo_mean_pred_size": round(g("grpo", "mean_pred_size"), 2) if g("grpo", "mean_pred_size") is not None else None,
        "sft_mean_pred_size": round(g("sft", "mean_pred_size"), 2) if g("sft", "mean_pred_size") is not None else None,
        "reward_first": hist[0]["reward"] if hist else None,
        "reward_last": hist[-1]["reward"] if hist else None,
        "parse_failures_total": parse_failures if stage_scores else None,
        "source_run_dir": run_dir.name,
        "caveats": "; ".join(caveats) if caveats else "",
        "notes": notes or "",
    }

    reward_rows = [{
        "run_tag": tag, "step": h.get("step"), "epoch": h.get("epoch"), "reward": h.get("reward"),
        **{col: reward_mean(h, *names) for col, names in REWARD_ALIASES.items()},
        "loss": h.get("loss"), "grad_norm": h.get("grad_norm"),
        "completions_mean_length": h.get("completions/mean_length"),
        "frac_reward_zero_std": h.get("frac_reward_zero_std"),
    } for h in hist]

    return tag, row, reward_rows, per_round_rows, ge_rows, baseline_rows, sft_hist


# ----------------------------------------------------------------- workbook I/O
def write_readme(ws):
    """(Re)write the README sheet from README_LINES."""
    for i in range(1, max(ws.max_row, len(README_LINES)) + 1):
        ws.cell(row=i, column=1).value = None
    for i, line in enumerate(README_LINES, start=1):
        c = ws.cell(row=i, column=1, value=line)
        c.font = Font(name=FONT, size=10, bold=(i == 1))
    ws.column_dimensions["A"].width = 100
    return ws


def build_workbook(path: Path):
    wb = Workbook()
    ws = wb.active
    ws.title = "README"
    write_readme(ws)
    for name, cols in SHEETS.items():
        _format_header(wb.create_sheet(name), cols)
    wb.save(path)
    return wb


def _format_header(s, cols):
    for j, (col, width) in enumerate(cols, start=1):
        c = s.cell(row=1, column=j, value=col)
        c.font = HEADER_FONT
        c.fill = HEADER_FILL
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        c.border = BORDER
        s.column_dimensions[get_column_letter(j)].width = width
    s.freeze_panes = "B2"
    s.row_dimensions[1].height = 34
    return s


def ensure_sheets(wb) -> list[str]:
    """Add any missing sheet; existing rows are never touched."""
    added = []
    for name, cols in SHEETS.items():
        if name not in wb.sheetnames:
            _format_header(wb.create_sheet(name), cols)
            added.append(name)
        else:
            # columns are addressed by position, so a short header is rewritten in place
            ws = wb[name]
            if [ws.cell(row=1, column=i + 1).value for i in range(len(cols))] != [c[0] for c in cols]:
                _format_header(ws, cols)
    order = ["README"] + [n for n in SHEETS] + [n for n in wb.sheetnames if n != "README" and n not in SHEETS]
    wb._sheets = [wb[n] for n in order if n in wb.sheetnames]
    return added


def read_rows(ws, cols) -> list[dict]:
    names = [c[0] for c in cols]
    out = []
    r = 2
    while ws.cell(row=r, column=1).value not in (None, ""):
        out.append({n: ws.cell(row=r, column=i + 1).value for i, n in enumerate(names)})
        r += 1
    return out


def write_rows(ws, cols, rows: list[dict]):
    names = [c[0] for c in cols]
    r = 2
    while ws.cell(row=r, column=1).value not in (None, ""):
        r += 1
    for rr in range(2, r):
        for cc in range(1, len(names) + 1):
            ws.cell(row=rr, column=cc).value = None
    for i, d in enumerate(rows, start=2):
        for j, n in enumerate(names, start=1):
            c = ws.cell(row=i, column=j, value=d.get(n))
            c.font = BODY
            c.border = BORDER
            c.alignment = Alignment(vertical="top", wrap_text=n in ("caveats", "notes"))
            if n in PCT and isinstance(d.get(n), float):
                c.number_format = "0.0%"
            elif isinstance(d.get(n), float):
                c.number_format = "0.0000"


def replace_rows(ws, cols, tag, new_rows):
    existing = [d for d in read_rows(ws, cols) if d["run_tag"] != tag]
    write_rows(ws, cols, existing + new_rows)


# run 4 (three seeds), run 5 (three seeds), run 6, run 7
RUN_ORDER = ["grpo-h100-g16", "grpo-h100-g16-r2", "grpo-h100-g16-r3",
             "reward-grpo-h100-g16", "reward-grpo-h100-g16-r2", "reward-grpo-h100-g16-r3",
             "tactic-grpo-h100-g16", "tactic-reward-grpo-h100-g16"]


def sort_runs(wb):
    """Order every sheet as runs 4, 5, 6, 7; any other run follows, by date."""
    runs = read_rows(wb["Runs"], RUN_COLS)
    rank = lambda t: RUN_ORDER.index(t) if t in RUN_ORDER else len(RUN_ORDER)
    runs.sort(key=lambda d: (rank(d.get("run_tag")), str(d.get("date") or "9999"), str(d.get("run_tag") or "")))
    write_rows(wb["Runs"], RUN_COLS, runs)
    order = {d["run_tag"]: i for i, d in enumerate(runs)}
    for name, cols in SHEETS.items():
        if name == "Runs" or name not in wb.sheetnames:
            continue
        rows = read_rows(wb[name], cols)
        rows.sort(key=lambda d: order.get(d.get("run_tag"), len(order)))
        write_rows(wb[name], cols, rows)


def _census(wb) -> dict:
    """{(sheet, run_tag): hash of that run's rows}, to catch overwritten runs."""
    buckets: dict = {}
    for name, cols in SHEETS.items():
        if name not in wb.sheetnames:
            continue
        for d in read_rows(wb[name], cols):
            buckets.setdefault((name, d.get("run_tag")), []).append(
                repr([d.get(c) for c in cols]))
    return {k: (len(v), hashlib.md5("\n".join(sorted(v)).encode()).hexdigest()[:12])
            for k, v in buckets.items()}


def backup(path: Path):
    """Timestamped copy into results/Legacy/ before any save."""
    if not path.exists():
        return None
    legacy = path.parent / "Legacy"
    legacy.mkdir(exist_ok=True)
    dest = legacy / f"{path.stem}.{datetime.now():%Y%m%d-%H%M%S}{path.suffix}"
    shutil.copy2(path, dest)
    return dest


def write(path: Path, tag, run_row, reward_rows, per_round_rows, ge_rows, baseline_rows):
    if not path.exists():
        print(f"{path.name} not found -- creating it.")
        build_workbook(path)
    bak = backup(path)
    wb = load_workbook(path)
    added = ensure_sheets(wb)
    if added:
        print(f"  added sheet(s): {', '.join(added)} (existing rows untouched)")
    before = _census(wb)

    # a tag must come from one run directory, otherwise two runs would overwrite each other
    src = run_row.get("source_run_dir")
    for d in read_rows(wb["Runs"], RUN_COLS):
        # keep an existing note unless --notes is given
        if d.get("run_tag") == tag and d.get("notes") and not run_row.get("notes"):
            run_row["notes"] = d["notes"]
        prior = d.get("source_run_dir")
        if d.get("run_tag") == tag and not prior:
            print(f"  note: the existing '{tag}' row predates source_run_dir tracking, so a"
                  f" collision with it cannot be detected. Re-analysing every run once binds them all.")
        if d.get("run_tag") == tag and prior and src and prior != src:
            print(f"\n  REFUSING TO SAVE -- tag collision on '{tag}'.")
            print(f"    already in the workbook, written from: {prior}")
            print(f"    this run comes from:                   {src}")
            print("  Give one of them a distinct RUN_TAG and re-run. Nothing was changed.")
            print(f"  Backup: {bak}")
            raise SystemExit(2)

    replace_rows(wb["Runs"], RUN_COLS, tag, [run_row])
    replace_rows(wb["Reward Detail"], RD_COLS, tag, reward_rows)
    replace_rows(wb["Per Round"], PR_COLS, tag, per_round_rows)
    replace_rows(wb["Game End"], GE_COLS, tag, ge_rows)
    replace_rows(wb["Baselines"], BL_COLS, tag, baseline_rows)
    sort_runs(wb)

    # refuse to save if any other run lost rows
    after = _census(wb)
    harmed = {k: (before[k], after.get(k)) for k in before
              if k[1] != tag and after.get(k) != before[k]}
    if harmed:
        print("\n  REFUSING TO SAVE -- writing this run would change another run's rows:")
        for (sheet, other), (b, a) in sorted(harmed.items(), key=lambda kv: str(kv[0])):
            a_txt = "GONE" if a is None else f"{a[0]} rows / {a[1]}"
            print(f"    {sheet:<14} {str(other):<30} {b[0]} rows / {b[1]}  ->  {a_txt}")
        print("  Most likely cause: this run's tag collides with one already in the")
        print("  workbook, or a run_tag in the Runs sheet was edited by hand.")
        print(f"  {path.name} on disk is unchanged. Backup: {bak}")
        raise SystemExit(2)

    if "README" in wb.sheetnames:
        write_readme(wb["README"])
    wb.save(path)
    if bak:
        print(f"  backup of the previous workbook: Legacy/{bak.name}")


def report(tag, row, reward_rows, per_round_rows, ge_rows, baseline_rows, sft_hist):
    print(f"\n=== {tag}  [{row['status']}] ===")
    print(f"  GPU {row['gpu']} | GRPO {row['grpo_steps_done']}/{row['grpo_max_steps']} steps "
          f"(G={row['grpo_num_generations']}) | train {row['total_train_hours']} h")
    if sft_hist:
        print(f"  SFT loss {sft_hist[0]['loss']:.4f} -> {sft_hist[-1]['loss']:.4f} over {len(sft_hist)} logged steps")
    if reward_rows:
        print(f"  GRPO reward {row['reward_first']:.4f} -> {row['reward_last']:.4f} over {len(reward_rows)} logged steps")
    if row["eval_rows"]:
        print(f"  Eval: {row['eval_rows']} rows / {row['eval_games']} games, rounds {row['eval_rounds_covered']}")
        print(f"  {'stage':<8}{'row F1':>9}{'game-end exact':>16}{'game-end F1':>13}{'mean pred':>11}")
        for s in STAGES:
            f1, ex, gf = row[f"{s}_f1"], row[f"{s}_game_end_exact"], row[f"{s}_game_end_f1"]
            mp = row.get(f"{s}_mean_pred_size")
            if f1 is None:
                print(f"  {s:<8}{'--':>9}"); continue
            print(f"  {s:<8}{f1:>9.3f}{ex:>15.1%}{gf:>13.3f}{('' if mp is None else f'{mp:>11.2f}')}")
        if ge_rows:
            ends = [r["end_round"] for r in ge_rows if r["stage"] == ge_rows[-1]["stage"]]
            print(f"  game-ending rows only, by the round each game ended")
            print("  " + f"{'stage':<11}" + "".join(f"{e:>14}" for e in ends))
            for st in ("gold_trace",) + STAGES:
                sub = {r["end_round"]: r for r in ge_rows if r["stage"] == st}
                if not sub:
                    continue
                cells = []
                for e in ends:
                    r = sub.get(e)
                    cells.append("--".rjust(14) if not r
                                 else f"{r['f1']:.3f}/{r['n_exact']}of{r['n_games']}".rjust(14))
                print(f"  {st:<11}" + "".join(cells))
            print("  " + " " * 11 + "(F1 / exact games)")
        print(f"  {'baselines (row level)':<30}")
        for b in baseline_rows:
            if b["scope"] == "row_level":
                print(f"    {b['baseline']:<18} F1={b['f1']:.3f}  exact={b['exact_match']:.1%}")
    if row["caveats"]:
        print("  CAVEATS: " + row["caveats"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", nargs="+", help="Run output folder(s)")
    ap.add_argument("--xlsx", default=str(Path(__file__).parent / "grpo_results.xlsx"))
    ap.add_argument("--run-tag", help="Override the tag (only valid with a single run_dir)")
    ap.add_argument("--notes", default="")
    ap.add_argument("--gold-selftest", default=str(GOLD_SELFTEST),
                    help="Gold-trace self-test CSV used for the Game End ceiling row")
    ap.add_argument("--rebuild", action="store_true", help="Delete the workbook first and rebuild from the given run dirs")
    ap.add_argument("--dry-run", action="store_true", help="Print only; write nothing")
    args = ap.parse_args()

    if args.run_tag and len(args.run_dir) > 1:
        sys.exit("--run-tag only makes sense with a single run_dir")

    xlsx = Path(args.xlsx)
    if args.rebuild and xlsx.exists() and not args.dry_run:
        xlsx.unlink()
        print(f"Removed {xlsx.name} (--rebuild)")

    for rd in args.run_dir:
        run_dir = Path(rd)
        if not run_dir.is_dir():
            sys.exit(f"FATAL: {run_dir} is not a directory")
        tag, row, rw, pr, ge, bl, sft_hist = analyse(run_dir, args.run_tag, args.notes,
                                                     Path(args.gold_selftest))
        report(tag, row, rw, pr, ge, bl, sft_hist)
        if not args.dry_run:
            write(xlsx, tag, row, rw, pr, ge, bl)
            print(f"  -> {xlsx}")


if __name__ == "__main__":
    main()
