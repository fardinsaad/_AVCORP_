#!/usr/bin/env python3
"""Writes the MT tabs of grpo_results.xlsx from 8_eval-test-with-model-trace outputs; every other tab is left as it is.

Usage:
    python add_model_trace.py                      # every run folder in ../trained/eval-test-model-trace/self-chain/
    python add_model_trace.py <run_dir> [...]      # only these folders (e.g. the runs that have finished so far)
    python add_model_trace.py --dry-run            # print the numbers, write nothing
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Alignment

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_results as ar  # noqa: E402

HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = HERE.parent / "trained" / "eval-test-model-trace" / "self-chain"
KNOWN_FOLDERS = {  # output folder -> prompt variant
    "base-grpo-h100-g16": "trace", "base-grpo-h100-g16-r2": "trace", "base-grpo-h100-g16-r3": "trace",
    "base-reward-grpo-h100-g16": "trace", "base-reward-grpo-h100-g16-r2": "trace",
    "base-reward-grpo-h100-g16-r3": "trace",
    "tactic-grpo-h100-g16": "tactic", "tactic-reward-grpo-h100-g16": "tactic",
}
BASE_TAG = {"trace": "base (trace prompt)", "tactic": "base (tactic prompt)"}
STAGE_CAVEATS = {("reward-grpo-h100-g16", "sft"): "sft-warmup is not the SFT this run's GRPO started from; never pair sft with grpo"}

MT_RUN_COLS = [
    ("run_tag", 24), ("stage", 8), ("n_rows", 8), ("n_games", 8), ("parse_failures", 9),
    ("hit_max_new_tokens", 10), ("f1", 9), ("macro_f1", 9), ("exact_match", 10), ("mean_pred_size", 10),
    ("game_end_exact", 11), ("game_end_n_exact", 10), ("game_end_f1", 11),
    ("gold_f1", 9), ("gold_game_end_exact", 11), ("gold_game_end_f1", 11),
    ("delta_f1", 9), ("delta_game_end_exact", 11), ("delta_game_end_f1", 11),
    ("gpu", 17), ("caveats", 52), ("source_dir", 30),
]
MT_SHEETS = {"MT Runs": MT_RUN_COLS, "MT Per Round": ar.PR_COLS, "MT Game End": ar.GE_COLS}
MT_PCT = {"f1", "macro_f1", "exact_match", "game_end_exact", "game_end_f1", "gold_f1",
          "gold_game_end_exact", "gold_game_end_f1", "delta_f1", "delta_game_end_exact",
          "delta_game_end_f1", "precision", "recall"}


def tag_of(folder: str) -> str:
    return folder[len("base-"):] if folder.startswith("base-") else folder  # same rule as analyze_results


def load_stage(run_dir: Path, stage: str) -> pd.DataFrame | None:
    p = run_dir / f"eval_predictions_{stage}.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    if len(df) != 100 or df["game_id"].nunique() != 25:
        raise SystemExit(f"FATAL: {p} has {len(df)} rows / {df['game_id'].nunique()} games, expected 100 / 25")
    if "chain_mode" not in df.columns or set(df["chain_mode"]) != {"self"}:
        raise SystemExit(f"FATAL: {p} is not a model-trace (self-chain) evaluation")
    return df


def gold_values(runs_rows: list[dict], tag: str, stage: str, variant: str) -> tuple:
    """(f1, game_end_exact, game_end_f1) of the gold-trace evaluation, from the Runs tab."""
    keys = (f"{stage}_f1", f"{stage}_game_end_exact", f"{stage}_game_end_f1")
    if stage != "base":
        row = next((d for d in runs_rows if d.get("run_tag") == tag), None)
        if row is None:
            raise SystemExit(f"FATAL: run_tag '{tag}' is not in the Runs tab")
        return tuple(row.get(k) for k in keys)
    prefix = "tactic-" if variant == "tactic" else "base-"
    vals = {tuple(d.get(k) for k in keys) for d in runs_rows
            if str(d.get("source_run_dir") or "").startswith(prefix) and d.get("eval_rows") == 100}
    if len(vals) != 1:
        raise SystemExit(f"FATAL: gold-trace base values for the {variant} prompt are not unique in Runs: {vals}")
    return vals.pop()


def analyse_folder(run_dir: Path, runs_rows: list[dict]):
    folder = run_dir.name
    if folder not in KNOWN_FOLDERS:
        raise SystemExit(f"FATAL: {folder} is not one of the 8 model-trace run folders")
    variant, tag = KNOWN_FOLDERS[folder], tag_of(folder)
    summ = run_dir / "eval_summary.csv"
    gpu = {r["stage"]: r.get("gpu") for r in pd.read_csv(summ).to_dict("records")} if summ.exists() else {}
    runs, per_round, game_end = [], [], []
    for stage in ar.STAGES:
        df = load_stage(run_dir, stage)
        if df is None:
            continue
        row_tag = BASE_TAG[variant] if stage == "base" else tag
        s = ar.score_frame(df)
        ge_df = ar.game_end_frame(df)
        g = ar.score_frame(ge_df)
        gold = gold_values(runs_rows, tag, stage, variant)
        new = (s["f1"], g["exact_match"], g["f1"])
        runs.append({
            "run_tag": row_tag, "stage": stage, "n_rows": s["n"], "n_games": g["n"],
            "parse_failures": int((~df["parsed"].astype(bool)).sum()),
            "hit_max_new_tokens": int(df["hit_max_new_tokens"].astype(bool).sum()),
            "f1": s["f1"], "macro_f1": s["macro_f1"], "exact_match": s["exact_match"],
            "mean_pred_size": s["mean_pred_size"],
            "game_end_exact": g["exact_match"], "game_end_n_exact": int(round(g["exact_match"] * g["n"])),
            "game_end_f1": g["f1"],
            "gold_f1": gold[0], "gold_game_end_exact": gold[1], "gold_game_end_f1": gold[2],
            "delta_f1": None if gold[0] is None else new[0] - gold[0],
            "delta_game_end_exact": None if gold[1] is None else new[1] - gold[1],
            "delta_game_end_f1": None if gold[2] is None else new[2] - gold[2],
            "gpu": gpu.get(stage), "caveats": STAGE_CAVEATS.get((tag, stage)), "source_dir": folder,
        })
        for rnd in sorted(int(r) for r in df["round_id"].unique()):
            sub = ar.score_frame(df[df["round_id"].astype(int) == rnd])
            per_round.append({"run_tag": row_tag, "stage": stage, "round_id": rnd, "n_rows": sub["n"],
                              "precision": sub["precision"], "recall": sub["recall"], "f1": sub["f1"],
                              "exact_match": sub["exact_match"]})
        game_end += ar.game_end_rows(row_tag, stage, ge_df)
        if not any(r["run_tag"] == row_tag and r["stage"] == "gold_trace" for r in game_end):
            game_end += ar.gold_game_end_rows(row_tag, set(ge_df["game_id"]))[0]  # ceiling on the same games
    return runs, per_round, game_end


def write_mt(ws, cols, rows):
    names = [c[0] for c in cols]
    for rr in range(2, ws.max_row + 1):
        for cc in range(1, len(names) + 1):
            ws.cell(row=rr, column=cc).value = None
    for i, d in enumerate(rows, start=2):
        for j, n in enumerate(names, start=1):
            v = d.get(n)
            c = ws.cell(row=i, column=j, value=v)
            c.font, c.border = ar.BODY, ar.BORDER
            c.alignment = Alignment(vertical="top", wrap_text=n == "caveats")
            if n in MT_PCT and isinstance(v, float):
                c.number_format = "0.0%"
            elif isinstance(v, float):
                c.number_format = "0.0000"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", nargs="*", help=f"model-trace run folders (default: every folder in {DEFAULT_ROOT})")
    ap.add_argument("--xlsx", default=str(HERE / "grpo_results.xlsx"))
    ap.add_argument("--dry-run", action="store_true", help="print, write nothing")
    args = ap.parse_args()
    dirs = [Path(d) for d in args.run_dir] or sorted(p for p in DEFAULT_ROOT.iterdir() if p.is_dir())
    xlsx = Path(args.xlsx)
    if not xlsx.exists():
        raise SystemExit(f"FATAL: {xlsx} not found")

    wb = load_workbook(xlsx)
    runs_rows = ar.read_rows(wb["Runs"], ar.RUN_COLS)
    new = {name: [] for name in MT_SHEETS}
    for d in dirs:
        r, pr, ge = analyse_folder(d, runs_rows)
        new["MT Runs"] += r; new["MT Per Round"] += pr; new["MT Game End"] += ge
        for x in r:
            print(f"  {x['run_tag']:<26} {x['stage']:<5} f1 {x['f1']:.3f} (gold {x['gold_f1'] or 0:.3f})  "
                  f"game_end_exact {x['game_end_n_exact']}/{x['n_games']} = {x['game_end_exact']:.1%} "
                  f"(gold {x['gold_game_end_exact'] or 0:.1%})  parse_fail {x['parse_failures']}")
    keys = [(x["run_tag"], x["stage"]) for x in new["MT Runs"]]
    if len(keys) != len(set(keys)):
        raise SystemExit(f"FATAL: the same (run_tag, stage) comes from two folders: {keys}")
    if args.dry_run:
        print("dry run: nothing written")
        return

    bak = ar.backup(xlsx)
    before = ar._census(wb)
    order = list(BASE_TAG.values()) + [d.get("run_tag") for d in runs_rows]
    for name, cols in MT_SHEETS.items():
        if name not in wb.sheetnames:
            ar._format_header(wb.create_sheet(name), cols)
        ws = wb[name]
        if [ws.cell(row=1, column=i + 1).value for i in range(len(cols))] != [c[0] for c in cols]:
            raise SystemExit(f"FATAL: {name} header differs from this script's columns; nothing written")
        written = {x["run_tag"] for x in new[name]}
        rows = [x for x in ar.read_rows(ws, cols) if x.get("run_tag") not in written] + new[name]
        rows.sort(key=lambda x: (order.index(x["run_tag"]) if x["run_tag"] in order else len(order),
                                 ar.STAGES.index(x["stage"]) if x["stage"] in ar.STAGES else 3))
        write_mt(ws, cols, rows)
    if ar._census(wb) != before:
        raise SystemExit(f"FATAL: a non-MT tab changed; {xlsx.name} on disk is unchanged. Backup: {bak}")
    ar.write_readme(wb["README"])
    wb.save(xlsx)
    print(f"  wrote {sum(len(v) for v in new.values())} rows to the MT tabs of {xlsx.name}; backup: Legacy/{bak.name}")


if __name__ == "__main__":
    main()
