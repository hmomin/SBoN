from algorithm.generator import Generator
from pprint import pprint
from utils.batch_utils import calculate_batch_sizes
from utils.generation_utils import (
    get_input_encoding,
    get_output_texts,
    get_templated_prompt,
    unpad_output_texts,
)
from utils.reward_utils import compute_scores
from utils.trajectory import Trajectory
from utils.validation_utils import validate_alpha


class SpeculativeRejection(Generator):
    def generate(self, prompt: str, prompt_dict: dict | None = None) -> None:
        if prompt_dict is None:
            prompt_dict = prompt
        self.prepare_generation(prompt_dict)
        self.clock.reset()
        self.clock.start()
        self.prompt = prompt
        self.templated_prompt = get_templated_prompt(
            prompt, self.args.llm_name, self.generation_tokenizer
        )
        rejection_rate: float = self.args.rejection_rate
        decision_token: float = self.args.decision_token
        validate_alpha(rejection_rate)
        batch_encoding = get_input_encoding(
            [self.templated_prompt],
            self.generation_model,
            self.generation_tokenizer,
        )
        input_length = batch_encoding.input_ids.shape[-1]
        batch_size: int = self.args.batch_size
        num_trajectories: int = self.args.num_trajectories
        batch_sizes = calculate_batch_sizes(num_trajectories, batch_size)
        first_stop = input_length + decision_token
        assert first_stop < self.args.max_tokens
        self.clock.stop("hyperparameter selection")
        print(f"input_length: {input_length}", flush=True)
        self.clock.start()
        current_trajectories: list[Trajectory] = []
        for batch_size in batch_sizes:
            current_generations = [self.templated_prompt] * batch_size
            batch_trajectories = self.generate_batch_trajectories(
                prompt, current_generations, first_stop
            )
            current_trajectories.extend(batch_trajectories)
        current_generations = self.perform_speculative_rejection(
            current_trajectories, rejection_rate
        )
        self.clock.stop(f"speculative rejection - current_length {first_stop}")
        self.clock.start()
        if len(current_generations) > 0:
            current_trajectories = []
            batch_sizes = calculate_batch_sizes(len(current_generations), batch_size)
            for batch_size in batch_sizes:
                batch_trajectories = self.generate_batch_trajectories(
                    prompt, current_generations, self.args.max_tokens
                )
                current_trajectories.extend(batch_trajectories)
            self.trajectories.extend(current_trajectories)
        self.clock.stop("finish")
        self.post_generation()

    def generate_batch_trajectories(
        self, prompt: str, current_generations: list[str], max_length: int
    ) -> list[Trajectory]:
        batch_encoding = get_input_encoding(
            current_generations,
            self.generation_model,
            self.generation_tokenizer,
        )
        self.clock.stop("tokenization")
        self.clock.start()
        partial_generation = self.generation_model.generate(
            batch_encoding.input_ids,
            pad_token_id=self.generation_tokenizer.pad_token_id,
            max_length=max_length,
            eos_token_id=self.terminators,
            do_sample=True,
            top_p=self.args.top_p,
            top_k=self.args.top_k,
            temperature=self.args.temperature,
        )
        current_length = partial_generation.shape[-1]
        self.clock.stop(
            f"generation - partial_generation.shape {partial_generation.shape}"
        )
        print(f"partial_generation shape: {partial_generation.shape}", flush=True)
        self.clock.start()
        padded_output_texts = get_output_texts(
            partial_generation,
            self.templated_prompt,
            self.generation_tokenizer,
            skip_special_tokens=False,
        )
        unpadded_output_texts = unpad_output_texts(
            padded_output_texts, self.stop_tokens
        )
        self.clock.stop(f"decoding - current_length {current_length}")
        self.clock.start()
        reward_list = compute_scores(
            prompt,
            unpadded_output_texts,
            self.reward_model_name,
            self.reward_tokenizer,
            self.reward_model,
        )
        self.clock.stop(f"reward - current_length {current_length}")
        self.clock.start()
        current_trajectories: list[Trajectory] = [
            Trajectory(
                self.prompt,
                self.templated_prompt,
                padded_output_text,
                unpadded_output_text,
                score,
            )
            for padded_output_text, unpadded_output_text, score in zip(
                padded_output_texts, unpadded_output_texts, reward_list
            )
        ]
        return current_trajectories

    def perform_speculative_rejection(
        self,
        current_trajectories: list[Trajectory],
        alpha: float,
    ) -> list[str]:
        current_trajectories.sort(key=lambda trajectory: trajectory.score, reverse=True)
        keep_fraction = 1.0 - alpha
        keep_amount = int(round(keep_fraction * len(current_trajectories)))
        self.trajectories = current_trajectories[:keep_amount]
        # NOTE: might as well keep the finished trajectories...
        self.trajectories += [
            trajectory for trajectory in current_trajectories if trajectory.finished
        ]
        generating_trajectories = [
            trajectory for trajectory in self.trajectories if not trajectory.finished
        ]
        current_generations = [
            trajectory.templated_prompt + trajectory.unpadded_output_text
            for trajectory in generating_trajectories
        ]
        return current_generations
