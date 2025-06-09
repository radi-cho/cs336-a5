#!/bin/bash

# SLURM configuration
#SBATCH --job-name=grpo_experiments
#SBATCH --output=logs/grpo_%A_%a.out
#SBATCH --error=logs/grpo_%A_%a.err
#SBATCH --partition=a5-batch
#SBATCH --qos=a5-batch-qos
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --array=0-8

mkdir -p logs

epochs=(1 2 3)
batch_sizes=(256 128 64)

epoch_idx=$((SLURM_ARRAY_TASK_ID / 3))
batch_idx=$((SLURM_ARRAY_TASK_ID % 3))

epochs_per_rollout_batch=${epochs[$epoch_idx]}
train_batch_size=${batch_sizes[$batch_idx]}

echo "Running experiment with:"
echo "epochs_per_rollout_batch: $epochs_per_rollout_batch"
echo "train_batch_size: $train_batch_size"
echo "job_id: $SLURM_ARRAY_TASK_ID"

python cs336_alignment/grpo.py \
    --epochs $epochs_per_rollout_batch \
    --batch_size $train_batch_size \
    --job_id $SLURM_ARRAY_TASK_ID 