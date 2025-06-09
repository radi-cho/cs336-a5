#!/bin/bash

epochs=(1 2 3)
batch_sizes=(256 128 64)

for epoch in "${epochs[@]}"; do
    for batch_size in "${batch_sizes[@]}"; do
        echo "Submitting job with:"
        echo "epochs_per_rollout_batch: $epoch"
        echo "train_batch_size: $batch_size"
        
        sbatch --partition=a5-batch \
               --qos=a5-batch-qos \
               --gpus=1 \
               --output=logs/grpo_${epoch}_${batch_size}.out \
               --error=logs/grpo_${epoch}_${batch_size}.err \
               --wrap="uv run cs336_alignment/grpo.py --epochs $epoch --batch_size $batch_size"
    done
done