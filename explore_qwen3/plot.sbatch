#!/bin/bash

#SBATCH --job-name=plot_heatmaps-1gpu
#SBATCH --partition=gpu
#SBATCH --qos=qos_nmi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=17
#SBATCH --mem-per-cpu=8G
#SBATCH --time=00:30:00
#SBATCH --array=0-6
#SBATCH --output=slurm_out/plot_heatmaps-1gpu-%A_%a.out
#SBATCH --error=slurm_out/plot_heatmaps-1gpu-%A_%a.err

echo "Job array ID: $SLURM_ARRAY_JOB_ID"
echo "Job array index: $SLURM_ARRAY_TASK_ID"

echo "Setting up environment variables and paths..."
# --- User-Specific Paths ---
# Please set your scratch directory before running
if [ -z "${CONTAINER_PATH}" ]; then
  echo "ERROR: Please set the CONTAINER_PATH environment variable."
  exit 1
fi

if [ -z "${TRITON_CACHE_DIR}" ]; then
  echo "ERROR: Please set the TRITON_CACHE_DIR environment variable."
  exit 1
fi

if [ -z "${HOST_CA_CERT_PATH}" ]; then
  echo "ERROR: Please set the HOST_CA_CERT_PATH environment variable."
  exit 1
fi

if [ -z "${CONTAINER_CA_CERT_PATH}" ]; then
  echo "ERROR: Please set the CONTAINER_CA_CERT_PATH environment variable."
  exit 1
fi

# Calculate session number (cycle through available sessions)
SESSION_NUM=$((SLURM_ARRAY_TASK_ID % 20 + 1))
export OVERLAY_PATH="/gpfs/radev/home/xs272/scratch/overlay_llama_factory_session${SESSION_NUM}.img"

echo "Using overlay session: $SESSION_NUM"
echo "Overlay path: $OVERLAY_PATH"

export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

echo "=== Generating Qwen3 Heatmaps ==="

# Define the list of models
MODEL_LIST=(
    "Qwen/Qwen3-0.6B-Base"
    "Qwen/Qwen3-1.7B-Base"
    "Qwen/Qwen3-4B-Base"
    "Qwen/Qwen3-8B-Base"
    "Qwen/Qwen3-14B-Base"
    "Qwen/Qwen3-30B-A3B"
    "Qwen/Qwen3-32B"
)

# Get the model name for this array task
MODEL_NAME="${MODEL_LIST[$SLURM_ARRAY_TASK_ID]}"

echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Processing model: $MODEL_NAME"

PLOT_COMMAND="python plot_heatmap_checkpoints.py --model_name \"${MODEL_NAME}\""

echo "Starting Apptainer container and executing plot script..."
srun apptainer exec \
    --nv \
    --bind ${HOST_CA_CERT_PATH}:${CONTAINER_CA_CERT_PATH} \
    --overlay ${OVERLAY_PATH} \
    ${CONTAINER_PATH} \
    bash -c "${PLOT_COMMAND}"

echo "Plot generation completed for $MODEL_NAME!"
echo "Check ./figures/ for results"

