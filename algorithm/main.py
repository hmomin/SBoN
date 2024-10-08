import argparse
import gc
import os
import secrets
import torch
from accelerate import PartialState
from accelerate.utils import gather_object
from algorithm.best_of_n import BestOfN
from algorithm.speculative_rejection import SpeculativeRejection
from collections import namedtuple
from pprint import pprint
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
        required=True,
    )
    parser.add_argument(
        "--num_trajectories",
        help="how many trajectories to generate/score per prompt",
        type=int,
        default=1_000,
    )
    parser.add_argument(
        "--speculative_rejection",
        help="use speculative rejection for generation?",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--decision_token",
        help="what 'score before' token index to use for speculative rejection",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "--rejection_rate",
        help="what percentage of trajectories to reject after scoring them at decision token",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--max_tokens",
        help="maximum number of tokens to generate per trajectory",
        type=int,
        default=2_048,
    )
    parser.add_argument(
        "--batch_size",
        help="batch size to use for best-of-N - ignored when using speculative rejection",
        type=int,
        default=20,
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
        "--pretty_print_output",
        help="should output file be easily human-readable?",
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
        "--local_files_only",
        help="whether to use local_files_only for HF models",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--temperature",
        help="temperature parameter for generation model",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--device_id",
        help="which GPU to use",
        type=int,
        default=-1,
    )
    args = parser.parse_args()
    return args


def main() -> None:
    args = get_args()
    PseudoState = namedtuple(
        "PseudoState", ["device", "local_process_index", "is_main_process"]
    )
    distributed_state = PseudoState(
        f"cuda{(':' + str(args.device_id)) if args.device_id > 0 else ''}", 0, True
    )
    state_device = str(distributed_state.device)
    print(f"DEVICE: {state_device}")
    pprint(vars(args))

    generator = (
        SpeculativeRejection(args, distributed_state)
        if args.speculative_rejection
        else BestOfN(args, distributed_state)
    )

    generation_prompts = get_generation_prompts(args)
    output_folder = create_output_folder(args)

    while len(generation_prompts) > 0:
        print(f"Number of prompts remaining: {len(generation_prompts)}", flush=True)
        prompt_dict = secrets.choice(generation_prompts)
        pprint(prompt_dict)
        prompt: str = prompt_dict["prompt"]

        generator.generate(prompt, prompt_dict=prompt_dict)

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        all_data_gather = gather_object(generator.all_data)
        write_to_disk(
            all_data_gather,
            output_folder,
            generator.initial_memory,
            args.pretty_print_output,
            args.record_memory,
        )
        generation_prompts = get_generation_prompts(args)
    print("DONE")


if __name__ == "__main__":
    with torch.no_grad():
        main()
