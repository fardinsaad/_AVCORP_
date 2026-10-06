#!/bin/bash
# Run 7 (Slurm): tactic labels, revised rewards; RUN_DIR uses the tactic- prefix. Submit: sbatch 7_submit_grpo-h100-g16.slurm

#SBATCH --job-name=avcorp-tactic-reward-grpo-h100-g16
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=/share/mpsingh/fsaad/lsf-logs/avcorp-tactic-reward-grpo-h100-g16.%j.out
#SBATCH --error=/share/mpsingh/fsaad/lsf-logs/avcorp-tactic-reward-grpo-h100-g16.%j.err

# --time is HH:MM:SS; --gres must name the GPU type

source /usr/share/Modules/init/bash 2>/dev/null || true
module load apptainer/1.4.2-1 2>/dev/null || module load apptainer 2>/dev/null || true

set -euo pipefail

REPO=/home/fsaad/avcorp
CONTAINER=/usr/local/apps/ood/images/jupyter-pytorch.sif
NOTEBOOK=7_sft-h100-grpo-h100-g16_AVCORP-training-trace-tactic-labels-HPC.ipynb
RUN_DIR=/share/mpsingh/fsaad/avcorp-runs/tactic-reward-grpo-h100-g16

# this banner should appear within a minute of the job starting
echo "=== $(hostname) · $(date) ==="
echo "SLURM_JOB_ID=${SLURM_JOB_ID}  NODELIST=${SLURM_JOB_NODELIST}  GPUS=${CUDA_VISIBLE_DEVICES:-unset}"
echo "RUN_TAG=reward-grpo-h100-g16  RUN_DIR=$RUN_DIR"

[ -d "$REPO" ]                            || { echo "FATAL: repo missing: $REPO"; exit 1; }
[ -f "$REPO/$NOTEBOOK" ]                  || { echo "FATAL: notebook missing: $NOTEBOOK"; exit 1; }
[ -f "$REPO/Deception-Dataset.csv" ]      || { echo "FATAL: corpus CSV missing (cell 6 reads it relatively)"; exit 1; }
[ -f "$CONTAINER" ]                       || { echo "FATAL: container missing: $CONTAINER"; exit 1; }
[ -d /share/mpsingh/fsaad/hf_cache/hub ]  || { echo "FATAL: Qwen3-8B not staged in HF cache"; exit 1; }
command -v apptainer >/dev/null           || { echo "FATAL: apptainer not on PATH after module load"; exit 1; }
apptainer --version

nvidia-smi || echo "WARNING: nvidia-smi failed on host"

cd "$REPO"

# touch the autofs path first, then bind it into the container
ls /share/mpsingh/fsaad > /dev/null

apptainer exec --nv --bind /share/mpsingh/fsaad "$CONTAINER" \
  jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=-1 \
    "$NOTEBOOK"

echo "=== finished $(date) ==="
ls -la "$RUN_DIR" 2>/dev/null || echo "(run dir not created)"
