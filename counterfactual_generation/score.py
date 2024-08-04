# NOTE: this file is used to do scoring with the following reward models:
# reward-model-deberta-v3-large-v2
# RM-Mistral-7B
# FsfairX-LLaMA3-RM-v0.1
# ArmoRM-Llama3-8B-v0.1

import argparse
import json
import numpy as np
import os
import random
import secrets
import time
import torch
import transformers
from collections import namedtuple
from copy import deepcopy
from pprint import pprint
from time import sleep
from tqdm import tqdm
from typing import Any
from utils.reward_utils import (
    get_reward_model,
    get_reward_tokenizer,
    get_rewards,
    get_reward_tokens,
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
        "--batch_size",
        help="how many trajectories to score in parallel per batch",
        type=int,
        default=1,
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
    batch_size: int,
    tokenizer,
) -> list[dict[str, Any]]:
    rebatched_data: list[dict[str, Any]] = []
    relevant_batch_keys = ["trajectories"]
    args_batch: dict[str, Any] = deepcopy(generation_data[0])
    args_batch.pop("trajectories")
    rebatched_data.append(args_batch)

    input_token_ids = generation_data[1]["input_token_ids"]
    total_objects: dict[str, list[Any]] = {}
    total_length = 0
    for relevant_key in relevant_batch_keys:
        total_elements = get_packed_data(generation_data, relevant_key)
        total_objects[relevant_key] = total_elements
        total_length = len(total_elements)
    batch_counter = 0
    for idx in range(0, total_length, batch_size):
        new_batch: dict[str, Any] = {
            "batch_idx": batch_counter,
            "input_token_ids": input_token_ids,
        }
        for relevant_key in relevant_batch_keys:
            new_batch[relevant_key] = total_objects[relevant_key][
                idx : idx + batch_size
            ]
            if relevant_key == "output_token_ids":
                pad_generation_tokens(new_batch[relevant_key], tokenizer)
        rebatched_data.append(new_batch)
        batch_counter += 1
    return rebatched_data


def get_packed_data(generation_data: list[dict[str, Any]], data_key: str) -> list[Any]:
    all_generation_elements: list[Any] = []
    for batch_data in generation_data[1:]:
        generation_elements: list[Any] = batch_data[data_key]
        all_generation_elements.extend(generation_elements)
    return all_generation_elements


def pad_generation_tokens(
    generation_tokens: list[list[int]],
    tokenizer,
) -> None:
    tokenization_lengths = [len(tokenization) for tokenization in generation_tokens]
    max_length = max(tokenization_lengths)
    min_length = min(tokenization_lengths)
    if min_length < max_length:
        for tokenization in generation_tokens:
            while len(tokenization) < max_length:
                tokenization.append(tokenizer.pad_token_id)


def compute_iterative_rewards(
    question: str,
    output_tokens: torch.Tensor,
    generation_tokenizer,
    reward_model_name: str,
    reward_tokenizer,
    reward_model,
    batch_data: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    previous_rewards = [[0.0]] * args.batch_size
    running_indices = list(range(args.batch_size))
    batch_rewards: list[list[list[float]]] = []
    generation_length = output_tokens.shape[-1]
    # print(f"output_length: {generation_length}")
    score_indices = compute_score_indices(generation_length)
    for token_idx in score_indices:
        partial_sequences, finished_indices = get_partial_sequences(
            output_tokens, token_idx, generation_tokenizer, running_indices
        )
        if partial_sequences.numel() > 0:
            output_texts = generation_tokenizer.batch_decode(
                partial_sequences, skip_special_tokens=True
            )
            reward_tokens = get_reward_tokens(
                question,
                output_texts,
                reward_model_name,
                reward_tokenizer,
                reward_model.device,
            )
            start_time = time.time()
            reward_list = get_rewards(reward_model_name, reward_model, reward_tokens)
            end_time = time.time()
            print(f"token_idx: {token_idx} - time: {end_time - start_time}")
            if reward_list is None:
                return
            list_rewards: list[list[float]] = reward_list
        else:
            list_rewards = []
        previous_rewards = augment_rewards(
            list_rewards, running_indices, previous_rewards
        )
        running_indices = list(set(range(args.batch_size)) - set(finished_indices))
        assert len(list_rewards) == args.batch_size
        batch_rewards.append(list_rewards)
    # print(f"scored batch {batch_data['batch_idx']} in {(end_time - start_time):.2f}s")
    batch_scores = np.array(batch_rewards).T.tolist()
    batch_data["reward_trajectories"] = batch_scores
    batch_data["score_before_indices"] = score_indices


def compute_score_indices(generation_length: int) -> list[int]:
    score_indices = [2 ** i for i in range(int(np.log2(generation_length)) + 1)]
    if score_indices[-1] < generation_length:
        score_indices.append(generation_length)
    return score_indices


def get_partial_sequences(
    output_tokens: torch.Tensor,
    token_idx: int,
    generation_tokenizer,
    running_indices: list[int],
) -> tuple[torch.Tensor, list[int]]:
    partial_sequences = output_tokens[running_indices, :token_idx]
    finished_sequences = (
        output_tokens[:, token_idx - 1] == generation_tokenizer.eos_token_id
    )
    finished_indices = torch.nonzero(finished_sequences).squeeze(-1).tolist()
    return partial_sequences, finished_indices


def augment_rewards(
    rewards_list: list[list[float]],
    running_indices: list[int],
    previous_rewards: list[list[float]],
) -> list[list[float]]:
    previous_finished_indices = list(
        set(range(len(previous_rewards))) - set(running_indices)
    )
    previous_finished_indices.sort()
    for finished_idx in previous_finished_indices:
        rewards_list.insert(finished_idx, previous_rewards[finished_idx])
    return rewards_list


def write_to_disk(
    all_data: list[dict[str, Any]],
    output_folder: str,
    pretty_print_output: bool = False,
) -> None:
    prompt_idx: int = all_data[0]["PROMPT_IDX"]
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
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)
    PseudoState = namedtuple(
        "PseudoState", ["device", "local_process_index", "is_main_process"]
    )
    distributed_state = PseudoState(
        f"cuda{(':' + str(args.device_id)) if args.device_id > 0 else ''}", 0, True
    )
    output_folder = create_output_folder(args)
    generation_filepaths = get_generation_filepaths(args, output_folder)
    get_other_args(args, generation_filepaths[0])
    pprint(args)
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
        pprint(list(generation_data[0].keys()))
        prompt_dict: dict[str, Any] = generation_data[0]["prompt"]
        pprint(prompt_dict)
        question: str = prompt_dict["prompt"]
        pprint(question)
        generation_data = rebatch_generation_data(
            generation_data, args.batch_size, generation_tokenizer
        )
        raise
        print(f"input_token_length: {len(generation_data[1]['input_token_ids'])}")
        num_trajectories_scored = 0
        for batch_data in tqdm(generation_data[1:]):
            output_tokens = torch.tensor(
                batch_data["output_token_ids"],
                dtype=torch.int64,
                device=distributed_state.device,
            )
            num_trajectories = output_tokens.shape[0]
            compute_iterative_rewards(
                question,
                output_tokens,
                generation_tokenizer,
                reward_model_name,
                reward_tokenizer,
                reward_model,
                batch_data,
                args,
            )
            # NOTE: write to disk after the first batch to "hog" this prompt
            if num_trajectories_scored == 0:
                write_to_disk(generation_data, output_folder, args.pretty_print_output)
            if "reward_trajectories" in batch_data:
                num_trajectories_scored += num_trajectories
        write_to_disk(generation_data, output_folder, args.pretty_print_output)
        print(f"Scored {num_trajectories_scored} trajectories.", flush=True)
        generation_filepaths = get_generation_filepaths(args, output_folder)
    print("DONE")


if __name__ == "__main__":
    with torch.no_grad():
        main()
