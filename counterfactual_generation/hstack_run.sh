nvidia-smi
conda activate hf
cd ~/SBoN

python -m counterfactual_generation.generate \
--data_filename ./datasets/alpaca_farm_100.json \
--output_folder output_AF_gpt-xl__20_1000_seed_0 \
--llm_name gpt-xl \
--num_trajectories 1000 \
--max_length 1024 \
--batch_size 20 \
--seed 0 \
--top_k 50 \
--top_p 1.0
