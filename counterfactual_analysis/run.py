import argparse
import json
import numpy as np
import random
from counterfactual_analysis.plotter import plot_data
from counterfactual_analysis.simple_rejection import SimpleRejection
from counterfactual_analysis.trajectory import (
    Trajectory,
    get_batch_stats,
    get_total_tokens,
)
from counterfactual_analysis.trial import Trial, TrialCollector
from copy import deepcopy
from glob import glob
from pprint import pprint
from tqdm import tqdm
from typing import Any

rejection_rates = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.98]
# rejection_rates = [0.10, 0.50, 0.90]  # FIXME!!!


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Counterfactual Analysis")
    parser.add_argument(
        "--data_folder",
        help="folder containing all scoring data",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--num_trials",
        help="number of trials to perform for each prompt",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--num_samples",
        help="number of samples to draw for each trial",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--seed",
        help="random seed for transformers",
        type=int,
        default=0,
    )
    args = parser.parse_args()
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed + 1)


def get_filepaths(data_folder: str) -> list[str]:
    filepaths = glob(f"{data_folder}/*.json")
    return filepaths


def get_all_trajectories(
    filepaths: list[str],
) -> tuple[list[list[Trajectory]], list[int]]:
    all_trajectories: list[list[Trajectory]] = []
    for filepath in filepaths:
        full_data: list[dict[str, Any]] = load_data(filepath)
        decision_tokens = get_decision_tokens(full_data[0]["max_length"])
        assert len(full_data) == 2
        trajectory_dicts: list[dict[str, Any]] = full_data[1]["trajectories"]
        trajectories = [
            Trajectory(trajectory_dict) for trajectory_dict in trajectory_dicts
        ]
        assert len(trajectories) == 1_000
        all_trajectories.append(trajectories)
    return all_trajectories, decision_tokens


def load_data(filepath: str) -> list[dict[str, Any]]:
    with open(filepath, "r") as f:
        data = json.load(f)
    return data


def get_decision_tokens(max_length: int) -> list[int]:
    log_final_token = int(np.log2(max_length))
    decision_tokens = [2 ** i for i in range(log_final_token + 1)]
    if decision_tokens[-1] == max_length:
        decision_tokens.pop()
    return decision_tokens


def main() -> None:
    args = get_args()
    set_seed(args.seed)
    data_folder: str = args.data_folder
    filepaths = get_filepaths(data_folder)
    filed_trajectories, decision_tokens = get_all_trajectories(filepaths)

    trial_collector = TrialCollector(rejection_rates, decision_tokens)
    for _ in tqdm(range(args.num_trials)):
        for trajectories in filed_trajectories:
            sampled_trajectories = random.choices(trajectories, k=args.num_samples)
            bon_tokens = get_total_tokens(sampled_trajectories)
            absolute_difference, bon_max_score = get_batch_stats(sampled_trajectories)
            for rejection_rate in rejection_rates:
                for decision_token in decision_tokens:
                    trial = Trial(
                        bon_tokens,
                        bon_max_score,
                        absolute_difference,
                        rejection_rate,
                        decision_token,
                    )
                    sbon_trajectories = deepcopy(sampled_trajectories)
                    counterfactual_test = SimpleRejection(
                        sbon_trajectories, rejection_rate, decision_token
                    )
                    sbon_tokens, sbon_max_score = counterfactual_test.run()
                    trial.update(sbon_tokens, sbon_max_score)
                    trial_collector.add_trial(trial)
    trial_collector.consolidate_stats()
    plot_data(trial_collector.stats)


if __name__ == "__main__":
    main()
