#!/bin/bash

source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate hf
cd /home/ubuntu/SBoN

data_folder=(
    "output_AF_gpt2-xl__20_1000_seed_0"
)
RM_names=(
    "reward-model-deberta-v3-large-v2"
    "RM-Mistral-7B"
    "FsfairX-LLaMA3-RM-v0.1"
    "ArmoRM-Llama3-8B-v0.1"
    "reward-model-deberta-v3-large-v2"
    "RM-Mistral-7B"
    "FsfairX-LLaMA3-RM-v0.1"
    "ArmoRM-Llama3-8B-v0.1"
)
device_ids=(0 1 2 3 4 5 6 7)

for i in "${!device_ids[@]}"
do
    device_id=${device_ids[$i]}
    RM_name=${RM_names[$i]}
    
    python -m counterfactual_generation.score \
    --data_folder $data_folder \
    --reward_model_name $RM_name \
    --device_id $device_id &
done

# wait