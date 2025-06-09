#!/bin/bash

loss_types=("grpo_no_clip" "grpo_clip")

for loss_type in "${loss_types[@]}"; do
    echo "Submitting job with:"
    echo "loss_type: $loss_type"
    
    sbatch --partition=a5-batch \
           --qos=a5-batch-qos \
           --gpus=1 \
           --output=logs/grpo_${loss_type}.out \
           --error=logs/grpo_${loss_type}.err \
           --wrap="uv run cs336_alignment/grpo.py --loss_type $loss_type"
done