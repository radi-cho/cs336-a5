#!/bin/bash

# SLURM configuration
#SBATCH --partition=a5-batch
#SBATCH --qos=a5-batch-qos
#SBATCH --gpus=1

uv run cs336_alignment/grpo.py