#!/bin/bash

source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate hf
cd /home/ubuntu/SBoN

data_folders=(
    # "output_AF_gpt2-xl__20_1000_seed_0"
    # "output_AF_gpt-j-6b__20_1000_seed_0"
    "output_AF_Meta-Llama-3-8B__20_1000_seed_0"
    # "output_AF_Mistral-7B-v0.3__20_1000_seed_0"
)
RM_names=(
    "reward-model-deberta-v3-large-v2"
    "reward-model-deberta-v3-large-v2"
    "reward-model-deberta-v3-large-v2"
    "reward-model-deberta-v3-large-v2"
    # "RM-Mistral-7B"
    # "FsfairX-LLaMA3-RM-v0.1"
    # "ArmoRM-Llama3-8B-v0.1"
)
device_ids=(
    # 0
    # 1
    # 2
    # 3
    4
    5
    6
    7
)

index=0

for data_folder in "${data_folders[@]}"
do
    for RM_name in "${RM_names[@]}"
    do
        device_id=${device_ids[$index]}
        
        python -m counterfactual_generation.score \
        --data_folder $data_folder \
        --reward_model_name $RM_name \
        --device_id $device_id &
        
        index=$((index + 1))
    done
done

# wait
