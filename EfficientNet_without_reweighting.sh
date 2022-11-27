#!/usr/bin/env bash
#SBATCH -A SNIC2022-22-1091 -p alvis
#SBATCH -t 1:00:00
#SBATCH --gpus-per-node=A100:4
#SBATCH --nodes 1

python3 train_basic_model_efficientnet_without_reweighting.py

