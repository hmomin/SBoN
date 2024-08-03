nvidia-smi
conda activate hf
cd ~/SBoN

accelerate launch \
--num_processes 1 \
--num_machines 1 \
--machine_rank 0 \
--mixed_precision no \
--dynamo_backend no \
main.py \
--output_folder output_H100_BoN_AF_sft10k_ArmoRM-Llama3-8B-v0.1_20_8_seed_56 \
--llm_name sft10k \
--reward_model_name ArmoRM-Llama3-8B-v0.1 \
--max_tokens 2048 \
--max_gen_tokens 2048 --data_filename ./datasets/alpaca_farm_100.json --batch_size 20 --seed 56 --top_k 50 --top_p 1.0 --temperature 1.0
