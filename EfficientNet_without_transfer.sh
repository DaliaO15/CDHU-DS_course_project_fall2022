#!/usr/bin/env bash
#SBATCH -A SNIC2022-22-1091 -p alvis
#SBATCH -t 1:00:00
#SBATCH --gpus-per-node=A100:2
#SBATCH --nodes 1

module load TensorFlow/2.5.0-fosscuda-2020b matplotlib/3.3.3-fosscuda-2020b JupyterLab/2.2.8-GCCcore-10.2.0
pip install split_folders
pip install scikit-learn

jupyter lab