# NOTE: this file is used to do scoring with the following reward models:
# reward-model-deberta-v3-large-v2
# RM-Mistral-7B
# FsfairX-LLaMA3-RM-v0.1
# ArmoRM-Llama3-8B-v0.1

import argparse
import json
import numpy as np
import os
import secrets
import torch
import transformers
from collections import namedtuple
from copy import deepcopy
from pprint import pprint
from time import sleep
from tqdm import tqdm
from typing import Any
from utils.reward_utils import (
    compute_scores,
    get_reward_model,
    get_reward_tokenizer,
)
from utils.validation_utils import get_full_model_name

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_folder",
        help="folder to retrieve generated trajectories",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--reward_model_name",
        help="model name for scoring",
        type=str,
        default="reward-model-deberta-v3-large-v2",
    )
    parser.add_argument(
        "--device_id",
        help="which GPU to use",
        type=int,
        default=-1,
    )
    args = parser.parse_args()
    return args


def create_output_folder(args: argparse.Namespace) -> str:
    output_folder_name = f"{args.data_folder}_{args.reward_model_name}"
    if not os.path.exists(output_folder_name):
        os.mkdir(output_folder_name)
    return output_folder_name


def get_generation_filepaths(args: argparse.Namespace, output_folder: str) -> list[str]:
    all_basenames = os.listdir(args.data_folder)
    generation_basenames = remove_scored_generations(all_basenames, output_folder)
    generation_filepaths = [
        os.path.join(args.data_folder, basename) for basename in generation_basenames
    ]
    return generation_filepaths


def remove_scored_generations(
    generation_basenames: list[str], output_folder: str
) -> list[str]:
    scored_basenames = os.listdir(output_folder)
    scored_prompt_indices = [
        basename.split("prompt_")[-1] for basename in scored_basenames
    ]
    generation_prompt_indices = [
        generation.split("prompt_")[-1] for generation in generation_basenames
    ]
    remaining_basenames = [
        basename
        for basename, idx in zip(generation_basenames, generation_prompt_indices)
        if idx not in scored_prompt_indices
    ]
    return remaining_basenames


def get_other_args(args: argparse.Namespace, filepath: str) -> None:
    generation_data = load_generation_data(filepath)
    sample_batch = generation_data[0]
    sample_batch.pop("trajectories")
    sample_batch.pop("elapsed_sec")
    sample_batch.pop("clock")
    args.llm_name = sample_batch["llm_name"]
    args.max_length = sample_batch["max_length"]
    args.model_dir = sample_batch["model_dir"]
    args.pretty_print_output = sample_batch["pretty_print_output"]


def load_generation_data(filepath: str) -> list[dict[str, Any]]:
    print(f"Loading data from {filepath}...")
    with open(filepath, "r") as f:
        generation_data: list[dict[str, Any]] = json.load(f)
    return generation_data


def rebatch_generation_data(
    generation_data: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rebatched_data: list[dict[str, Any]] = []
    args_batch: dict[str, Any] = deepcopy(generation_data[0])
    args_batch.pop("trajectories")
    rebatched_data.append(args_batch)

    total_objects: dict[str, list[Any]] = {}
    total_elements = get_packed_data(generation_data, "trajectories")
    total_objects["trajectories"] = total_elements
    total_length = len(total_elements)
    assert total_length == generation_data[0]["num_trajectories"]
    rebatched_data.append(total_objects)
    return rebatched_data


def get_packed_data(generation_data: list[dict[str, Any]], data_key: str) -> list[Any]:
    all_generation_elements: list[Any] = []
    for batch_data in generation_data:
        generation_elements: list[Any] = batch_data[data_key]
        all_generation_elements.extend(generation_elements)
    return all_generation_elements


def get_input_length(
    generation_data: list[dict[str, Any]],
    generation_tokenizer: transformers.PreTrainedTokenizerFast,
) -> None:
    trajectories = generation_data[1]["trajectories"]
    prompt = generation_tokenizer.bos_token + trajectories[0]["prompt"]
    input_encoding = generation_tokenizer(
        [prompt], padding=True, add_special_tokens=False, return_tensors="pt"
    )
    input_length = input_encoding["input_ids"].shape[-1]
    generation_data[0]["input_length"] = input_length


def compute_iterative_rewards(
    trajectory: dict[str, Any],
    generation_tokenizer,
    reward_model_name: str,
    reward_tokenizer,
    reward_model,
    device: str,
) -> None:
    question = trajectory["prompt"]
    output_encoding = generation_tokenizer(
        [trajectory["output"]],
        padding=True,
        add_special_tokens=False,
        return_tensors="pt",
    ).to(device)
    output_tokens = output_encoding["input_ids"]
    partial_scores: list[float] = []
    generation_length = output_tokens.shape[-1]
    score_indices = compute_score_indices(generation_length)
    for token_idx in score_indices:
        partial_sequence = output_tokens[:, :token_idx]
        output_text = generation_tokenizer.decode(
            partial_sequence[0], skip_special_tokens=True
        )
        scores = compute_scores(
            question, [output_text], reward_model_name, reward_tokenizer, reward_model
        )
        score = scores[0]
        partial_scores.append(score)
    trajectory["score_before_indices"] = score_indices
    trajectory["partial_scores"] = partial_scores
    trajectory.pop("score")


def compute_score_indices(generation_length: int) -> list[int]:
    score_indices = [2 ** i for i in range(int(np.log2(generation_length)) + 1)]
    if score_indices[-1] < generation_length:
        score_indices.append(generation_length)
    return score_indices


def write_to_disk(
    all_data: list[dict[str, Any]],
    output_folder: str,
    pretty_print_output: bool = False,
) -> None:
    prompt_idx: int = all_data[0]["prompt"]["JSON_idx"]
    write_filename = f"{get_filename()}_prompt_{prompt_idx:04d}.json"
    write_path = os.path.join(output_folder, write_filename)
    with open(write_path, "w") as fp:
        if pretty_print_output:
            json.dump(all_data, fp, indent=4)
        else:
            json.dump(all_data, fp)
        print(f"Wrote data to {write_filename}")


def get_filename() -> str:
    filename = os.path.basename(__file__)
    split_without_extension = filename.split(".")[:-1]
    name_without_extension = ".".join(split_without_extension)
    return name_without_extension


def main() -> None:
    args = get_args()
    PseudoState = namedtuple(
        "PseudoState", ["device", "local_process_index", "is_main_process"]
    )
    distributed_state = PseudoState(
        f"cuda{(':' + str(args.device_id)) if args.device_id > 0 else ''}", 0, True
    )
    output_folder = create_output_folder(args)
    generation_filepaths = get_generation_filepaths(args, output_folder)
    get_other_args(args, generation_filepaths[0])
    print(f"Generation Model: {args.llm_name} - Reward Model: {args.reward_model_name}")
    LLM_name = get_full_model_name(args.model_dir, args.llm_name)
    reward_model_name = get_full_model_name(args.model_dir, args.reward_model_name)

    generation_tokenizer = transformers.AutoTokenizer.from_pretrained(LLM_name)
    generation_tokenizer.pad_token = generation_tokenizer.eos_token
    generation_tokenizer.padding_side = "right"

    reward_tokenizer = get_reward_tokenizer(reward_model_name)
    reward_model = get_reward_model(
        reward_model_name, reward_tokenizer, distributed_state.device
    )

    while len(generation_filepaths) > 0:
        generation_filepath = secrets.choice(generation_filepaths)
        generation_data = load_generation_data(generation_filepath)
        prompt_dict: dict[str, Any] = generation_data[0]["prompt"]
        question: str = prompt_dict["prompt"]
        pprint(question)
        generation_data = rebatch_generation_data(generation_data)
        get_input_length(generation_data, generation_tokenizer)
        print(f"input_token_length: {generation_data[0]['input_length']}", flush=True)
        num_trajectories_scored = 0
        trajectories = generation_data[1]["trajectories"]
        for trajectory in tqdm(trajectories):
            compute_iterative_rewards(
                trajectory,
                generation_tokenizer,
                reward_model_name,
                reward_tokenizer,
                reward_model,
                distributed_state.device,
            )
            # NOTE: write to disk after the first batch to "hog" this prompt
            if num_trajectories_scored == 0:
                write_to_disk(
                    [{"prompt": generation_data[0]["prompt"]}],
                    output_folder,
                    args.pretty_print_output,
                )
            num_trajectories_scored += 1
        write_to_disk(generation_data, output_folder, args.pretty_print_output)
        print(f"Scored {num_trajectories_scored} trajectories.", flush=True)
        generation_filepaths = get_generation_filepaths(args, output_folder)
    print("DONE")


if __name__ == "__main__":
    with torch.no_grad():
        main()
