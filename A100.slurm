#!/bin/bash
#SBATCH --job-name=A100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --time=1:00:00

### salloc --job-name=A100 --nodes=1 --ntasks=1 --cpus-per-task=8 --mem-per-cpu=32G --gres=gpu:1 --constraint=gpu80 --time=1:00:00

nvidia-smi
module load anaconda3/2024.2
conda init
conda activate hf
python -u script.py