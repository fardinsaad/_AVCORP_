#!/usr/bin/env python3
"""Writes crossdomain_results.xlsx: Avalon-NLU results with the old prompt (9_) and the domain-context prompt (10_).

Usage:
    python add_crossdomain_results.py
"""
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
TRAINED = HERE.parent / "trained"
SOURCES = {"old (9_)": TRAINED / "crossdomain" / "avalon-nlu",
           "new (10_)": TRAINED / "crossdomain-prompt" / "avalon-nlu"}
OUT = HERE / "crossdomain_results.xlsx"
FLAG = "NLU-3YjxjN"  # never solved by any model; the /19 columns leave it out
RUNS = {
    "base-grpo-h100-g16": ("run4_1", "trace"), "base-grpo-h100-g16-r2": ("run4_2", "trace"),
    "base-grpo-h100-g16-r3": ("run4_3", "trace"), "base-reward-grpo-h100-g16": ("run5_1", "trace"),
    "base-reward-grpo-h100-g16-r2": ("run5_2", "trace"), "base-reward-grpo-h100-g16-r3": ("run5_3", "trace"),
    "tactic-grpo-h100-g16": ("run6", "tactic"), "tactic-reward-grpo-h100-g16": ("run7", "tactic"),
}
CAVEATS = {("run5_1", "sft"): "sft-warmup is not the SFT this run's GRPO started from; not paired with grpo"}


def f1(pairs):
    pairs = list(pairs)
    tp = sum(len(p & t) for p, t in pairs)
    fp = sum(len(p - t) for p, t in pairs)
    fn = sum(len(t - p) for p, t in pairs)
    return 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0


def score(csv):
    d = pd.read_csv(csv)
    d["pred"] = d.predicted.fillna("").astype(str).str.split().apply(set)
    d["true"] = d.ground_truth.astype(str).str.split().apply(set)
    end = d[d.round_id == d.groupby("game_id").round_id.transform("max")]
    out = {}
    for tag, rows, ends in (("20", d, end), ("19", d[d.game_id != FLAG], end[end.game_id != FLAG])):
        pairs = list(zip(ends.pred, ends.true))
        exact = sum(p == t for p, t in pairs)
        out[f"game_end_exact_{tag}"] = exact
        out[f"game_end_pct_{tag}"] = 100 * exact / len(pairs)
        out[f"game_f1_{tag}"] = f1(pairs)
        out[f"f1_{tag}"] = f1(zip(rows.pred, rows.true))
    out["game_end_exactly_two"] = int((end[end.game_id != FLAG].pred.apply(len) == 2).sum())
    out["n_rows"] = len(d)
    out["parse_failures"] = int((~d.parsed.astype(bool)).sum())
    out["hit_max_new_tokens"] = int(d.hit_max_new_tokens.astype(bool).sum())
    return out


def collect():
    recs = []
    for prompt, root in SOURCES.items():
        for folder, (run, variant) in RUNS.items():
            for stage in ("base", "sft", "grpo"):
                csv = root / folder / f"eval_predictions_{stage}.csv"
                if not csv.exists():
                    continue
                label = f"base ({variant} prompt)" if stage == "base" else run
                recs.append({"prompt": prompt, "run": label, "stage": stage, **score(csv),
                             "caveat": CAVEATS.get((run, stage)), "source": str(csv.relative_to(TRAINED))})
    return pd.DataFrame(recs)


def family(run):
    return run if run.startswith("base") else run.split("_")[0]


def stage_means(df, cols):
    """Per run family first (run4 and run5 over their seeds, run6 and run7 single runs), then the overall means per prompt."""
    df = df.assign(family=df.run.map(family), n_runs=1)
    order = {f: i for i, f in enumerate(["base (trace prompt)", "base (tactic prompt)", "run4", "run5", "run6", "run7"])}
    per = df.groupby(["prompt", "family", "stage"]).agg({**{c: "mean" for c in cols}, "n_runs": "sum"}).reset_index()
    # the paper's SFT mean for run 5 leaves out run5_1 (see CAVEATS)
    paired = df[(df.family == "run5") & (df.stage == "sft") & (df.run != "run5_1")]
    paired = paired.groupby(["prompt", "family", "stage"]).agg({**{c: "mean" for c in cols}, "n_runs": "sum"}).reset_index()
    paired["family"] = "run5 (without run5_1)"
    per = pd.concat([per, paired])
    order["run5 (without run5_1)"] = order["run5"] + 0.5
    per["_o"] = per.family.map(order)
    per["_s"] = per.stage.map({"base": 0, "sft": 1, "grpo": 2})
    per = per.sort_values(["prompt", "_o", "_s"]).drop(columns=["_o", "_s"])
    overall = df.groupby(["prompt", "stage"]).agg({**{c: "mean" for c in cols}, "n_runs": "sum"}).reset_index()
    overall["family"] = "all runs"
    return pd.concat([per, overall])[["prompt", "family", "stage", "n_runs"] + cols]


def main():
    df = collect()
    keys = ["run", "stage"]
    cols = ["game_end_exact_19", "game_end_pct_19", "game_f1_19", "f1_19"]
    old = df[df.prompt == "old (9_)"].set_index(keys)[cols]
    new = df[df.prompt == "new (10_)"].set_index(keys)[cols]
    cmp_ = old.join(new, lsuffix="_old", rsuffix="_new", how="outer")
    for c in cols:
        cmp_[f"{c}_delta"] = cmp_[f"{c}_new"] - cmp_[f"{c}_old"]
    cmp_ = cmp_[[f"{c}_{s}" for c in cols for s in ("old", "new", "delta")]].reset_index()

    means = stage_means(df, cols)

    with pd.ExcelWriter(OUT, engine="openpyxl") as xw:
        for prompt in SOURCES:
            df[df.prompt == prompt].drop(columns="prompt").to_excel(xw, sheet_name=prompt.split()[0] + " prompt", index=False)
        cmp_.to_excel(xw, sheet_name="old vs new", index=False)
        means.to_excel(xw, sheet_name="stage means", index=False)
        for ws in xw.book.worksheets:
            for col in ws.columns:
                ws.column_dimensions[col[0].column_letter].width = max(10, min(45, len(str(col[0].value)) + 2))
            ws.freeze_panes = "C2"
    print(f"wrote {OUT.name}: {len(df)} rows")
    print(means.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
