#!/bin/bash
# Run 5, seed 2 (LSF): SFT + GRPO with the revised rewards, then evaluation. Submit: bsub < 5_2_submit_grpo-h100-g16.sh

#BSUB -q gpu
#BSUB -m gpu_h100
#BSUB -gpu "num=1:mode=exclusive_process:mps=no"
#BSUB -J avcorp-reward-grpo-h100-g16-r2
#BSUB -o /share/mpsingh/fsaad/lsf-logs/avcorp-reward-grpo-h100-g16-r2.%J.out
#BSUB -e /share/mpsingh/fsaad/lsf-logs/avcorp-reward-grpo-h100-g16-r2.%J.err
#BSUB -n 8
#BSUB -R "span[hosts=1] rusage[mem=64.00]"
#BSUB -W 2400

# batch shells are non-login, so load apptainer through the module system
source /usr/share/Modules/init/bash 2>/dev/null || true
module load apptainer/1.4.2-1 2>/dev/null || module load apptainer 2>/dev/null || true

set -euo pipefail

REPO=/home/fsaad/avcorp
CONTAINER=/usr/local/apps/ood/images/jupyter-pytorch.sif
NOTEBOOK=5_2_sft-h100-grpo-h100-g16_AVCORP-training-trace-labels-HPC.ipynb
RUN_DIR=/share/mpsingh/fsaad/avcorp-runs/base-reward-grpo-h100-g16-r2

echo "=== $(hostname) · $(date) ==="

[ -d "$REPO" ]                            || { echo "FATAL: repo missing: $REPO"; exit 1; }
[ -f "$REPO/$NOTEBOOK" ]                  || { echo "FATAL: notebook missing"; exit 1; }
[ -f "$REPO/Deception-Dataset.csv" ]      || { echo "FATAL: corpus CSV missing (cell 6 reads it relatively)"; exit 1; }
[ -f "$CONTAINER" ]                       || { echo "FATAL: container missing: $CONTAINER"; exit 1; }
[ -d /share/mpsingh/fsaad/hf_cache/hub ]  || { echo "FATAL: Qwen3-8B not staged in HF cache"; exit 1; }
command -v apptainer >/dev/null             || { echo "FATAL: apptainer not on PATH after module load"; exit 1; }
apptainer --version

nvidia-smi || echo "WARNING: nvidia-smi failed on host"

cd "$REPO"

# touch the autofs path first, then bind it into the container
ls /share/mpsingh/fsaad > /dev/null

# --nv passes the GPU in; --timeout=-1 disables nbconvert's cell timeout
apptainer exec --nv --bind /share/mpsingh/fsaad "$CONTAINER" \
  jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=-1 \
    "$NOTEBOOK"

echo "=== finished $(date) ==="
ls -la "$RUN_DIR" 2>/dev/null || echo "(run dir not created)"
