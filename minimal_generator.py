import argparse
import json
import torch
import transformers
from accelerate import PartialState
from argparse import Namespace
from pprint import pprint
from utils.generation_utils import (
    get_generation_model,
    get_generation_tokenizer,
    get_terminators,
    get_input_encoding,
    get_output_texts,
    get_templated_prompt,
    unpad_output_texts,
)
from utils.trajectory import Trajectory
from utils.validation_utils import (
    get_full_model_name,
    validate_llm_name,
)
from utils.read_write_utils import (
    get_generation_prompts,
)


class MinimalGenerator:
    def __init__(
        self,
        args: Namespace,
        distributed_state: PartialState,
    ) -> None:
        validate_llm_name(args.llm_name)
        llm_name = get_full_model_name(args.model_dir, args.llm_name)

        self.llm_name = llm_name
        self.args = args
        self.distributed_state = distributed_state

        self.process_seed = args.seed + distributed_state.local_process_index
        print(f"DEVICE: {distributed_state.device}")
        transformers.set_seed(self.process_seed)

        self.generation_tokenizer = get_generation_tokenizer(llm_name)
        self.stop_tokens = ["</s>", "<|end_of_text|>", "<|eot_id|>"]
        self.terminators = get_terminators(llm_name, self.generation_tokenizer)
        self.generation_model = get_generation_model(llm_name, distributed_state.device)

        self.templated_prompt = ""

    def generate(self, prompt: str):
        self.prompt = prompt
        self.templated_prompt = get_templated_prompt(
            prompt, self.args.llm_name, self.generation_tokenizer
        )
        batch_encoding = get_input_encoding(
            [self.templated_prompt],
            self.generation_model,
            self.generation_tokenizer,
        )
        full_generation: torch.LongTensor = self.generation_model.generate(
            input_ids=batch_encoding.input_ids,
            attention_mask=batch_encoding.attention_mask,
            max_length=self.args.max_tokens,
            eos_token_id=self.terminators,
            pad_token_id=self.generation_tokenizer.pad_token_id,
            do_sample=True,
            top_p=self.args.top_p,
            top_k=self.args.top_k,
            temperature=self.args.temperature,
        )
        print(f"full_generation shape: {full_generation.shape}")
        padded_output_texts = get_output_texts(
            full_generation,
            self.templated_prompt,
            self.generation_tokenizer,
            skip_special_tokens=False,
        )
        unpadded_output_texts = unpad_output_texts(
            padded_output_texts, self.stop_tokens
        )
        self.trajectory = Trajectory(
            self.prompt,
            self.templated_prompt,
            padded_output_texts[0],
            unpadded_output_texts[0],
            0.0,
        )


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_filename",
        help="relative filename containing sample prompts",
        type=str,
        default="./datasets/alpaca_farm_eval.json",
    )
    parser.add_argument(
        "--model_dir",
        help="directory containing model files - leave as '' to instantiate from huggingface",
        type=str,
        default="./../../../../scratch/gpfs/my0049",
    )
    parser.add_argument(
        "--output_folder",
        help="dummy directory needed by remove_generated_prompts",
        type=str,
        default="./output",
    )
    parser.add_argument(
        "--llm_name", help="model basename for generation", type=str, required=True
    )
    parser.add_argument(
        "--max_tokens",
        help="maximum number of tokens to generate per trajectory",
        type=int,
        default=2_048,
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
    args = parser.parse_args()
    return args


def write_to_json(generation_list: list[dict[str, str]], args: Namespace) -> None:
    output_filename = f"{args.llm_name}_generation.json"
    with open(output_filename, "w") as f:
        json.dump(generation_list, f, indent=4)


def main() -> None:
    distributed_state = PartialState()
    args = get_args()
    pprint(vars(args))

    generator = MinimalGenerator(args, distributed_state)

    generation_prompts = get_generation_prompts(args)

    generation_list: list[dict[str, str]] = []
    for prompt_dict in generation_prompts:
        pprint(prompt_dict)
        prompt: str = prompt_dict["prompt"]
        generator.generate(prompt)
        pprint(generator.trajectory.get_json_representation(sparse=False))
        generation_list.append(
            generator.trajectory.get_alpaca_representation(args.llm_name)
        )
    write_to_json(generation_list, args)
    print("DONE")


if __name__ == "__main__":
    main()
