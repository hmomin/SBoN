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

# Minimal Generation
#python -u minimal_generator.py --llm_name dpo-sft10k --max_tokens 2048 --seed 0 --top_k 50 --top_p 1.0
#python -u minimal_generator.py --llm_name ppo-human --max_tokens 2048 --seed 0 --top_k 50 --top_p 1.0 --temperature 0.7

# Rescoring regenerated trajectories
#python -u visualize_3_rescore.py

# Regeneration of dead trajectories
#python -u visualize_2_regenerate.py

# Speculative Rejection (alpha=?)
#accelerate launch --num_processes 1 --num_machines 1 --gpu_ids 0 --machine_rank 0 --mixed_precision no --dynamo_backend no visualize_1_store.py --llm_name sft10k --reward_model_name RM-Mistral-7B --max_tokens 2048 --batch_size 0 --seed 20000 --top_k 50 --top_p 1.0 --speculative_rejection --alpha 0.2

# Best-of-20
#accelerate launch --num_processes 1 --num_machines 1 --gpu_ids 0 --machine_rank 0 --mixed_precision no --dynamo_backend no demo.py --llm_name Meta-Llama-3-8B-Instruct --reward_model_name reward-model-human --max_tokens 2048 --batch_size 1 --seed 0 --top_k 50 --top_p 1.0

# Best-of-40
#accelerate launch --multi_gpu --num_processes 2 --num_machines 1 --gpu_ids 0,1 --machine_rank 0 --mixed_precision no --dynamo_backend no demo.py --llm_name sft10k --reward_model_name ArmoRM-Llama3-8B-v0.1 --max_tokens 2048 --batch_size 20 --seed 0 --top_k 50 --top_p 1.0 --record_memory

# Best-of-80
#accelerate launch --multi_gpu --num_processes 4 --num_machines 1 --gpu_ids 0,1,2,3 --machine_rank 0 --mixed_precision no --dynamo_backend no demo.py --llm_name sft10k --reward_model_name RM-Mistral-7B --max_tokens 2048 --batch_size 20 --seed 0 --top_k 50 --top_p 1.0 --record_memory

# Generation Profiling
#python -u generation_profiling_test.py

# Batch Size Benchmarking
#python -u benchmark_batch_size.py
