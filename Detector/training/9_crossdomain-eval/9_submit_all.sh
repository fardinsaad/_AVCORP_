#!/bin/bash
# Submits the 8 Avalon-NLU evaluation jobs (base only in run4_1 and run6); no smoke test, there are no reference outputs.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

SLURM=9_submit_crossdomain-eval.slurm
[ -f "$SLURM" ] && [ -f 9_crossdomain_eval.py ] && [ -f avalon_nlu.csv ] || { echo "FATAL: run this from the 9_crossdomain-eval folder"; exit 1; }

ids=()
while read -r run stages; do
  id=$(sbatch --parsable --job-name="nlu-$run" "$SLURM" "$run" "$stages" < /dev/null)
  id=${id%%;*}
  ids+=("$id")
  printf "nlu-%-7s %s  (%s)\n" "$run" "$id" "$stages"
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
echo "logs:       /share/mpsingh/fsaad/avcorp-runs/crossdomain/avalon-nlu/<run folder>/progress.log"
echo "cancel all: scancel ${ids[*]}"
