import argparse
import json
import os
from counterfactual_analysis.run import get_filepaths
from pprint import pprint


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
    hyperparameter_print = ["*" * 80]
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
        hyperparameter_print.append(f"{lm_name} + {rm_name}")
        hyperparameter_print.append(f"Rejection Rate: {best_rejection_rate}")
        hyperparameter_print.append(f"Decision Token: {best_decision_token}")
        hyperparameter_print.append(f"Token Rate:     {lowest_token_rate:.3f}")
        hyperparameter_print.append(f"Score:          {associated_score:.1f}")
        hyperparameter_print.append("*" * 80)
    full_print = "\n".join(hyperparameter_print)
    print(full_print)
    with open(os.path.join(args.data_folder, "hyperparameters.txt"), "w") as f:
        f.write(full_print)


if __name__ == "__main__":
    main()
