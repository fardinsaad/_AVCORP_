#!/bin/bash
# LSF version of 8_submit_all.sh: the smoke test is held until the 8 are queued, the 8 wait for it, and a failed smoke test kills them.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

JOB=8_submit_eval-test.lsf
LOG=/share/mpsingh/fsaad/lsf-logs
[ -f "$JOB" ] && [ -f 8_eval_model_trace.py ] || { echo "FATAL: run this from the 8_eval-test-with-model-trace folder"; exit 1; }
jid() { sed -n 's/^Job <\([0-9]*\)> is submitted.*/\1/p'; }

smoke=$(RUN=run4_1 STAGES=base,sft,grpo CHAIN=gold EXTRA="--limit-rows 2" \
        bsub -H -J evt-smoke -W 45 -o "$LOG/evt-smoke.%J.out" -e "$LOG/evt-smoke.%J.err" < "$JOB" | jid)
[ -n "$smoke" ] || { echo "FATAL: smoke test submission failed"; exit 1; }
echo "evt-smoke   $smoke  (held until the 8 are queued)"

: > 8_lsf_dependents.txt
while read -r run stages; do
  id=$(RUN="$run" STAGES="$stages" CHAIN=self EXTRA="" \
       bsub -w "done($smoke)" -J "evt-$run" -o "$LOG/evt-$run.%J.out" -e "$LOG/evt-$run.%J.err" < "$JOB" | jid)
  [ -n "$id" ] || { echo "FATAL: submission failed for $run. Clean up with: bkill $smoke $(tr '\n' ' ' < 8_lsf_dependents.txt)"; exit 1; }
  echo "$id" >> 8_lsf_dependents.txt
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

bresume "$smoke"
echo "released:   evt-smoke $smoke"
echo "watch:      bjobs -w -J 'evt-*'"
echo "banner:     bpeek $smoke"
echo "smoke log:  tail -f /share/mpsingh/fsaad/avcorp-runs/eval-test-model-trace/gold-chain/base-grpo-h100-g16/progress.log"
echo "cancel all: bkill $smoke $(tr '\n' ' ' < 8_lsf_dependents.txt)"
