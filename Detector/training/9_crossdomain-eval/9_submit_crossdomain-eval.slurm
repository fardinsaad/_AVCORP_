#!/bin/bash
# Usage: sbatch --job-name=<name> 9_submit_crossdomain-eval.slurm <run> <stages> [extra args]

#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=/share/mpsingh/fsaad/lsf-logs/%x.%j.out
#SBATCH --error=/share/mpsingh/fsaad/lsf-logs/%x.%j.err

source /usr/share/Modules/init/bash 2>/dev/null || true
module load apptainer/1.4.2-1 2>/dev/null || module load apptainer 2>/dev/null || true

set -euo pipefail

[ $# -ge 2 ] || { echo "usage: sbatch 9_submit_crossdomain-eval.slurm <run> <stages> [extra args]"; exit 1; }
RUN=$1; STAGES=$2; shift 2

WORK=${SLURM_SUBMIT_DIR:-$PWD}
CONTAINER=/usr/local/apps/ood/images/jupyter-pytorch.sif

echo "=== $(hostname) · $(date) ==="
echo "SLURM_JOB_ID=${SLURM_JOB_ID}  NODELIST=${SLURM_JOB_NODELIST}  GPUS=${CUDA_VISIBLE_DEVICES:-unset}"
echo "RUN=$RUN  STAGES=$STAGES  EXTRA=$*"

[ -f "$WORK/9_crossdomain_eval.py" ]      || { echo "FATAL: 9_crossdomain_eval.py not in $WORK"; exit 1; }
[ -f "$WORK/avalon_nlu.csv" ]            || { echo "FATAL: avalon_nlu.csv not in $WORK"; exit 1; }
[ -f "$CONTAINER" ]                       || { echo "FATAL: container missing: $CONTAINER"; exit 1; }
[ -d /share/mpsingh/fsaad/hf_cache/hub ]  || { echo "FATAL: Qwen3-8B not staged in HF cache"; exit 1; }
command -v apptainer >/dev/null           || { echo "FATAL: apptainer not on PATH after module load"; exit 1; }
nvidia-smi || echo "WARNING: nvidia-smi failed on host"

cd "$WORK"
ls /share/mpsingh/fsaad > /dev/null  # autofs: mount the path before binding it

apptainer exec --nv --bind /share/mpsingh/fsaad "$CONTAINER" \
  python3 -u 9_crossdomain_eval.py --run "$RUN" --stages "$STAGES" "$@"

echo "=== finished $(date) ==="
