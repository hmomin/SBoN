#!/bin/bash
#SBATCH --job-name=H100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=32G
#SBATCH --gres=gpu:1
#SBATCH --time=1:30:00
#SBATCH --partition=pli
#SBATCH --account=mengdigroup

nvidia-smi
module load anaconda3/2024.2
conda init
conda activate hf
python -u script.py
