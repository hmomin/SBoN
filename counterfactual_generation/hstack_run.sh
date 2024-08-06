#!/bin/bash

source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate hf
cd /home/ubuntu/SBoN

data_folders=(
    # "output_AF_gpt2-xl__20_1000_seed_0"
    "output_HH_gpt-j-6b__20_1000_seed_0"
    # "output_HH_Meta-Llama-3-8B__20_1000_seed_0"
    # "output_AF_Mistral-7B-v0.3__20_1000_seed_0"
)
RM_names=(
    "reward-model-deberta-v3-large-v2"
    "RM-Mistral-7B"
    "FsfairX-LLaMA3-RM-v0.1"
    "ArmoRM-Llama3-8B-v0.1"
)
device_ids=(
    0
    1
    2
    3
    # 4
    # 5
    # 6
    # 7
)

index=0

# for data_folder in "${data_folders[@]}"
# do
#     for RM_name in "${RM_names[@]}"
#     do
#         device_id=${device_ids[$index]}
        
#         python -m counterfactual_generation.score \
#         --data_folder $data_folder \
#         --reward_model_name $RM_name \
#         --device_id $device_id &
        
#         index=$((index + 1))
#     done
# done

# ("./datasets/alpaca_farm_100.json", "gpt2-xl", "ArmoRM-Llama3-8B-v0.1", 100, 20, 128, 0.8),
# ("./datasets/alpaca_farm_100.json", "gpt2-xl", "FsfairX-LLaMA3-RM-v0.1", 100, 20, 128, 0.7),
# ("./datasets/alpaca_farm_100.json", "gpt2-xl", "reward-model-deberta-v3-large-v2", 100, 20, 256, 0.8),
# ("./datasets/alpaca_farm_100.json", "gpt2-xl", "RM-Mistral-7B", 100, 20, 128, 0.8),

python -m algorithm.main \
--data_filename ./datasets/alpaca_farm_100.json \
--output_folder output_A100_SR_AF_gpt2-xl_ArmoRM-Llama3-8B-v0.1_128_0.8 \
--llm_name gpt2-xl \
--reward_model_name ArmoRM-Llama3-8B-v0.1 \
--num_trajectories 100 \
--speculative_rejection \
--decision_token 128 \
--rejection_rate 0.8 \
--max_tokens 1024 \
--batch_size 20 \
--seed 0 \
--top_k 50 \
--top_p 1.0 \
--temperature 1.0 \
--device_id 4 &

python -m algorithm.main \
--data_filename ./datasets/alpaca_farm_100.json \
--output_folder output_A100_SR_AF_gpt2-xl_FsfairX-LLaMA3-RM-v0.1_128_0.7 \
--llm_name gpt2-xl \
--reward_model_name FsfairX-LLaMA3-RM-v0.1 \
--num_trajectories 100 \
--speculative_rejection \
--decision_token 128 \
--rejection_rate 0.7 \
--max_tokens 1024 \
--batch_size 20 \
--seed 0 \
--top_k 50 \
--top_p 1.0 \
--temperature 1.0 \
--device_id 5 &

python -m algorithm.main \
--data_filename ./datasets/alpaca_farm_100.json \
--output_folder output_A100_SR_AF_gpt2-xl_reward-model-deberta-v3-large-v2_256_0.8 \
--llm_name gpt2-xl \
--reward_model_name reward-model-deberta-v3-large-v2 \
--num_trajectories 100 \
--speculative_rejection \
--decision_token 256 \
--rejection_rate 0.8 \
--max_tokens 1024 \
--batch_size 20 \
--seed 0 \
--top_k 50 \
--top_p 1.0 \
--temperature 1.0 \
--device_id 6 &

python -m algorithm.main \
--data_filename ./datasets/alpaca_farm_100.json \
--output_folder output_A100_SR_AF_gpt2-xl_RM-Mistral-7B_128_0.8 \
--llm_name gpt2-xl \
--reward_model_name RM-Mistral-7B \
--num_trajectories 100 \
--speculative_rejection \
--decision_token 128 \
--rejection_rate 0.8 \
--max_tokens 1024 \
--batch_size 20 \
--seed 0 \
--top_k 50 \
--top_p 1.0 \
--temperature 1.0 \
--device_id 7 &

# wait
