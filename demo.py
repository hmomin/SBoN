import argparse
import torch
from accelerate import PartialState
from accelerate.utils import gather_object
from best_of_n import BestOfN
from pprint import pprint
from utils.read_write_utils import get_best_response
from speculative_rejection import SpeculativeRejection


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        help="directory containing model files - leave as '' to instantiate from huggingface",
        type=str,
        default="./../../../../scratch/gpfs/my0049",
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
        "--speculative_rejection",
        help="use speculative rejection for generation?",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--alpha",
        help="fraction of trajectories (finished or generating) to reject on each speculative rejection pass",
        type=float,
        default=-1.0,
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
        "--record_memory",
        help="whether to profile memory usage during execution",
        action="store_true",
        default=False,
    )
    args = parser.parse_args()
    return args


def main() -> None:
    distributed_state = PartialState()
    args = get_args()
    pprint(vars(args))

    generator = (
        SpeculativeRejection(args, distributed_state)
        if args.speculative_rejection
        else BestOfN(args, distributed_state)
    )

    prompt = "What's the best way to cook scrambled eggs?"
    generator.generate(prompt)

    distributed_state.wait_for_everyone()
    all_data_gather = gather_object(generator.all_data)
    if distributed_state.is_main_process:
        best_response, best_score = get_best_response(all_data_gather)
        print("-" * 80)
        print(prompt)
        print("-" * 80)
        print(best_response)
        print("-" * 80)
        print(f"Reward of best response: {best_score}")


if __name__ == "__main__":
    with torch.no_grad():
        main()
