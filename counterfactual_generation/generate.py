# NOTE: this file can be used to do generation with any model supported by vLLM,
# including:
# gpt2
# sft10k (from Alpaca Farm)
# Llama-2-7b-hf
# Llama-2-7b-chat-hf
# Meta-Llama-3-8B
# Meta-Llama-3-8B-Instruct
# Mistral-7B-v0.1
# Mistral-7B-Instruct-v0.1

import argparse
import gc
import numpy as np
import os
import secrets
import torch
from accelerate import PartialState
from best_of_n import BestOfN
from collections import namedtuple
from pprint import pprint
from time import time
from typing import Any
from utils.read_write_utils import (
    create_output_folder,
    get_generation_prompts,
    write_to_disk,
)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_filename",
        help="relative filename containing sample prompts",
        type=str,
        default="./datasets/alpaca_farm_100.json",
    )
    parser.add_argument(
        "--output_folder",
        help="folder name of output files",
        type=str,
        default="./output_test",
    )
    parser.add_argument(
        "--model_dir",
        help="directory containing model files - leave as '' to instantiate from huggingface",
        type=str,
        default="./../../../scratch/gpfs/my0049",
    )
    parser.add_argument(
        "--llm_name", help="model basename for generation", type=str, required=True
    )
    parser.add_argument(
        "--reward_model_name",
        help="model basename for scoring",
        type=str,
        default="",
    )
    parser.add_argument(
        "--num_trajectories",
        help="total amount of trajectories to generate per prompt",
        type=int,
        default=1_000,
    )
    parser.add_argument(
        "--max_length",
        help="max length per generation (including input prompt)",
        type=int,
        default=2_048,
    )
    parser.add_argument(
        "--batch_size",
        help="how many trajectories to generate in parallel per batch",
        type=int,
        default=40,
    )
    parser.add_argument(
        "--seed",
        help="random seed for transformers",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--top_k",
        help="top-k parameter for generation model",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--top_p",
        help="top-p parameter for generation model",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--temperature",
        help="temperature parameter for generation model",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--local_files_only",
        help="whether to use local_files_only for HF models",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--record_memory",
        help="whether to profile memory usage during execution",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--pretty_print_output",
        help="should output file be easily human-readable?",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--device_id",
        help="which GPU to use",
        type=int,
        default=0,
    )
    args = parser.parse_args()
    return args


def main() -> None:
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id
    PseudoState = namedtuple(
        "PseudoState", ["device", "local_process_index", "is_main_process"]
    )
    distributed_state = PseudoState(f"cuda:{args.device_id}", args.device_id, True)
    state_device = str(distributed_state.device)
    print(f"DEVICE: {state_device}")
    pprint(vars(args))

    num_batches = int(np.ceil(args.num_trajectories / args.batch_size))

    generator = BestOfN(args, distributed_state)

    generation_prompts = get_generation_prompts(args)
    output_folder = create_output_folder(args)

    start_time = time()
    while len(generation_prompts) > 0:
        full_data: list[dict[str, Any]] = []
        print(f"Number of prompts remaining: {len(generation_prompts)}", flush=True)
        prompt_dict = secrets.choice(generation_prompts)
        pprint(prompt_dict)
        prompt: str = prompt_dict["prompt"]
        for _ in range(num_batches):
            generator.generate(prompt, prompt_dict=prompt_dict)
            full_data.extend(generator.all_data)

            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        write_to_disk(
            full_data,
            output_folder,
            generator.initial_memory,
            args.pretty_print_output,
            args.record_memory,
        )
        if state_device == "cuda":
            break  # NOTE: just stop after a single prompt - this prevents wasted computation
            # elapsed_hours = (time() - start_time) / 3600
            # if elapsed_hours > 2.5:
            #     break
        generation_prompts = get_generation_prompts(args)
    print("DONE")


if __name__ == "__main__":
    with torch.no_grad():
        main()
