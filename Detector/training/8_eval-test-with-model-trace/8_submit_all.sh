#!/bin/bash
# Submits the smoke test and the 8 evaluation jobs; the 8 start only if the smoke test passes, and Slurm cancels them if it fails.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

SLURM=8_submit_eval-test.slurm
[ -f "$SLURM" ] && [ -f 8_eval_model_trace.py ] || { echo "FATAL: run this from the 8_eval-test-with-model-trace folder"; exit 1; }

smoke=$(sbatch --parsable --job-name=evt-smoke --time=00:45:00 "$SLURM" run4_1 base,sft,grpo gold --limit-rows 2)
smoke=${smoke%%;*}
echo "evt-smoke   $smoke"

ids=()
while read -r run stages; do
  id=$(sbatch --parsable --dependency=afterok:"$smoke" --kill-on-invalid-dep=yes \
         --job-name="evt-$run" "$SLURM" "$run" "$stages" self < /dev/null)
  id=${id%%;*}
  ids+=("$id")
  printf "evt-%-7s %s  (%s, waits for %s)\n" "$run" "$id" "$stages" "$smoke"
done <<'RUNS'
run4_1 base,sft,grpo
run4_2 sft,grpo
run4_3 sft,grpo
run5_1 sft,grpo
run5_2 sft,grpo
run5_3 sft,grpo
run6 base,sft,grpo
run7 sft,grpo
RUNS

echo "watch:      squeue -u $USER -O \"JobID:10,Name:14,State:10,Reason:26,TimeUsed:12\""
echo "smoke log:  tail -f /share/mpsingh/fsaad/avcorp-runs/eval-test-model-trace/gold-chain/base-grpo-h100-g16/progress.log"
echo "cancel all: scancel $smoke ${ids[*]}"
