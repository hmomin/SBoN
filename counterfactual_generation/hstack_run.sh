#!/bin/bash

nvidia-smi
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate hf
cd /home/ubuntu/SBoN

data_filename="./datasets/alpaca_farm_100.json"
output_folder="output_AF_gpt2-xl__20_1000_seed_0"
llm_name="gpt2-xl"
num_trajectories=1000
max_length=1024
batch_size=20
seed=0
top_k=50
top_p=1.0
device_ids=(0,1,2,3,4,5,6,7)

for device_id in "${device_ids[@]}"
do
    python -m counterfactual_generation.generate \
    --data_filename $data_filename \
    --output_folder $output_folder \
    --llm_name $llm_name \
    --num_trajectories $num_trajectories \
    --max_length $max_length \
    --batch_size $batch_size \
    --seed $seed \
    --top_k $top_k \
    --top_p $top_p \
    --device_id $device_id &
done

# wait
