import argparse
import json
import os
from counterfactual_analysis.run import get_filepaths
from glob import glob
from pprint import pprint
from typing import Any


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute Hyperparameters")
    parser.add_argument(
        "--data_folder",
        help="folder containing all scoring data",
        type=str,
        default="./counterfactual_analysis/processed",
    )
    parser.add_argument(
        "--score_threshold",
        help="minimum threshold for score - compute lowest token rate above this threshold",
        type=float,
        default=99.0,
    )
    args = parser.parse_args()
    return args


def load_data(filepath: str) -> list[tuple[float, int, float, float]]:
    with open(filepath, "r") as f:
        data = json.load(f)
    return data


def get_model_names_from_filename(filepath: str) -> tuple[str, str]:
    basename = os.path.basename(filepath)
    split_basename = basename.split("_")
    lm_name = split_basename[2]
    rm_name = split_basename[8]
    return lm_name, rm_name


def main() -> None:
    args = get_args()
    filepaths = get_filepaths(args.data_folder)
    for filepath in filepaths:
        lowest_token_rate = 2.0
        best_rejection_rate = -1.0
        best_decision_token = -1
        associated_score = -1.0
        lm_name, rm_name = get_model_names_from_filename(filepath)
        data = load_data(filepath)
        for rejection_rate, decision_token, token_rate, score in data:
            if score >= args.score_threshold and token_rate < lowest_token_rate:
                lowest_token_rate = token_rate
                best_rejection_rate = rejection_rate
                best_decision_token = decision_token
                associated_score = score
        print(f"{lm_name} + {rm_name}")
        print(f"Rejection Rate: {best_rejection_rate}")
        print(f"Decision Token: {best_decision_token}")
        print(f"Token Rate:     {lowest_token_rate:.3f}")
        print(f"Score:          {associated_score:.1f}")
        print("*" * 80)


if __name__ == "__main__":
    main()
