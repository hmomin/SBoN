# Checks speedup and suboptimality of speculative rejection

import json
import numpy as np
import os
from glob import glob
from matplotlib import pyplot as plt
from pprint import pprint
from time import sleep
from tqdm import tqdm
from typing import Any

GENERATION_FOLDER_PATHS = [
    "./output_AF_gpt2-xl__20_1000_seed_0",
    "./output_AF_gpt2-xl__20_1000_seed_0",
    "./output_AF_gpt2-xl__20_1000_seed_0",
    "./output_AF_gpt2-xl__20_1000_seed_0",
    "./output_AF_gpt-j-6b__20_1000_seed_0",
    "./output_AF_gpt-j-6b__20_1000_seed_0",
    "./output_AF_gpt-j-6b__20_1000_seed_0",
    "./output_AF_gpt-j-6b__20_1000_seed_0",
    "./output_AF_Meta-Llama-3-8B__20_1000_seed_0",
    "./output_AF_Meta-Llama-3-8B__20_1000_seed_0",
    "./output_AF_Meta-Llama-3-8B__20_1000_seed_0",
]
SCORE_FOLDER_PATHS = [
    "./output_AF_gpt2-xl__20_1000_seed_0_ArmoRM-Llama3-8B-v0.1",
    "./output_AF_gpt2-xl__20_1000_seed_0_FsfairX-LLaMA3-RM-v0.1",
    "./output_AF_gpt2-xl__20_1000_seed_0_reward-model-deberta-v3-large-v2",
    "./output_AF_gpt2-xl__20_1000_seed_0_RM-Mistral-7B",
    "./output_AF_gpt-j-6b__20_1000_seed_0_ArmoRM-Llama3-8B-v0.1",
    "./output_AF_gpt-j-6b__20_1000_seed_0_FsfairX-LLaMA3-RM-v0.1",
    "./output_AF_gpt-j-6b__20_1000_seed_0_reward-model-deberta-v3-large-v2",
    "./output_AF_gpt-j-6b__20_1000_seed_0_RM-Mistral-7B",
]
SBON_FOLDER_PATHS = [
    "./output_A100_SR_AF_gpt2-xl_ArmoRM-Llama3-8B-v0.1_128_0.8",
    "./output_A100_SR_AF_gpt2-xl_FsfairX-LLaMA3-RM-v0.1_128_0.7",
    "./output_A100_SR_AF_gpt2-xl_reward-model-deberta-v3-large-v2_256_0.8",
    "./output_A100_SR_AF_gpt2-xl_RM-Mistral-7B_128_0.8",
    "./output_H100_SR_AF_gpt-j-6b_ArmoRM-Llama3-8B-v0.1_128_0.8",
    "./output_H100_SR_AF_gpt-j-6b_FsfairX-LLaMA3-RM-v0.1_128_0.7",
    "./output_H100_SR_AF_gpt-j-6b_reward-model-deberta-v3-large-v2_128_0.7",
    "./output_H100_SR_AF_gpt-j-6b_RM-Mistral-7B_128_0.8",
]


def get_json_filepaths(JSON_FOLDER_PATH) -> list[str]:
    return glob(os.path.join(JSON_FOLDER_PATH, "*.json"))


def get_data_from_filepath(filepath: str) -> list[dict[str, Any]]:
    with open(filepath, "r") as f:
        data: list[dict[str, Any]] = json.load(f)
        assert type(data) == list
        assert type(data[0]) == dict
    return data


def get_full_bon_data(
    generation_data: list[dict[str, Any]], score_data: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    scored_trajectories = score_data[1]["trajectories"]
    score_idx = 0
    assert len(scored_trajectories) == 1000
    assert len(generation_data) == 50
    for batch in generation_data:
        trajectories = batch["trajectories"]
        assert len(trajectories) == 20
        for trajectory in trajectories:
            scored_trajectory = scored_trajectories[score_idx]
            trajectory["score"] = scored_trajectory["partial_scores"][-1]
            score_idx += 1
    return generation_data


def nesting_check(data: list[dict[str, Any]]) -> None:
    for element in data:
        pprint(list(element.keys()))


def compute_suboptimality_score(
    bon_data: list[dict[str, Any]], spec_rej_data: list[dict[str, Any]]
) -> float:
    bon_scores = [t["score"] for batch in bon_data for t in batch["trajectories"]]
    assert len(bon_scores) == 100
    spec_rej_scores = [
        t["score"] for batch in spec_rej_data for t in batch["trajectories"]
    ]
    absolute_difference = max(bon_scores) - min(bon_scores)
    sbon_difference = max(spec_rej_scores) - min(bon_scores)
    suboptimality_score = sbon_difference / absolute_difference * 100
    return suboptimality_score


def main() -> None:
    assert (
        len(GENERATION_FOLDER_PATHS)
        == len(SCORE_FOLDER_PATHS)
        == len(SBON_FOLDER_PATHS)
    ), f"{len(GENERATION_FOLDER_PATHS)} != {len(SCORE_FOLDER_PATHS)} != {len(SBON_FOLDER_PATHS)}"
    for generation_folder, score_folder, sbon_folder in zip(
        GENERATION_FOLDER_PATHS, SCORE_FOLDER_PATHS, SBON_FOLDER_PATHS
    ):
        print(sbon_folder)
        print("****************************************************")
        generation_filepaths = sorted(get_json_filepaths(generation_folder))
        score_filepaths = sorted(get_json_filepaths(score_folder))
        spec_rej_filepaths = sorted(get_json_filepaths(sbon_folder))
        assert (
            len(generation_filepaths) == len(score_filepaths) == len(spec_rej_filepaths)
        ), f"{len(generation_filepaths)} != {len(score_filepaths)} != {len(spec_rej_filepaths)}"

        suboptimality_scores: list[float] = []
        total_bon_time = 0.0
        total_spec_rej_time = 0.0

        for generation_filepath, score_filepath, spec_rej_filepath in zip(
            generation_filepaths, score_filepaths, spec_rej_filepaths
        ):
            generation_filepath_ending = generation_filepath.split("_")[-1]
            score_filepath_ending = score_filepath.split("_")[-1]
            spec_rej_filepath_ending = spec_rej_filepath.split("_")[-1]
            assert (
                generation_filepath_ending
                == score_filepath_ending
                == spec_rej_filepath_ending
            ), f"{generation_filepath_ending} != {score_filepath_ending} != {spec_rej_filepath_ending}"
            generation_data = get_data_from_filepath(generation_filepath)
            score_data = get_data_from_filepath(score_filepath)
            spec_rej_data = get_data_from_filepath(spec_rej_filepath)
            bo1000_data = get_full_bon_data(generation_data, score_data)
            bon_data = bo1000_data[:5]
            bon_time = sum([d["elapsed_sec"] for d in bon_data])
            spec_rej_time = sum([d["elapsed_sec"] for d in spec_rej_data])
            total_bon_time += bon_time
            total_spec_rej_time += spec_rej_time
            suboptimality_score = compute_suboptimality_score(bon_data, spec_rej_data)
            suboptimality_scores.append(suboptimality_score)
            # print("****************************************************")
        # plot histogram of suboptimality scores
        # plt.hist(suboptimality_scores, bins=100)
        # plt.title("Suboptimality Scores")
        # plt.xlabel("Suboptimality Score")
        # plt.ylabel("Frequency")
        # plt.show()
        mean_suboptimality_score = np.mean(suboptimality_scores)
        # median_suboptimality_score = np.median(suboptimality_scores)
        mean_speedup = total_bon_time / total_spec_rej_time
        print(f"mean speedup: {mean_speedup:.3f}")
        # print(f"relative compute time: {(1/mean_speedup):.3f}")
        print(f"mean score: {(mean_suboptimality_score):.1f}")
        # print(f"median score: {(median_suboptimality_score):.1f}")
        # print(f"effective N: {int(round(num_trajectories/mean_speedup))}")
        print("****************************************************")


if __name__ == "__main__":
    main()
