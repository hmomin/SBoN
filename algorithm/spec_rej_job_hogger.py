import os
import subprocess
from pprint import pprint
from time import sleep

USERNAME = "my0049"

"""
DATASET in (
    "./datasets/hh_rlhf_100.json",
    "./datasets/alpaca_farm_100.json",
)
LLM in (
    "gpt2-xl",
    "gpt-j-6b",
    "Mistral-7B-v0.3",
    "Meta-Llama-3-8B",
)
RM in (
    "reward-model-deberta-v3-large-v2",
    "RM-Mistral-7B",
    "FsfairX-LLaMA3-RM-v0.1",
    "ArmoRM-Llama3-8B-v0.1",
)
"""

MAX_H100_GPUS = 8
MAX_A100_JOBS = 0
# JOB FORMAT: (DATASET, LLM, RM, NUM_TRAJECTORIES, BATCH_SIZE, DECISION_TOKEN, REJECTION_RATE)
JOBS = {
    "A100": [],
    "H100": [
        (
            "./datasets/alpaca_farm_100.json",
            "gpt-j-6b",
            "ArmoRM-Llama3-8B-v0.1",
            100,
            20,
            128,
            0.8,
        ),
        (
            "./datasets/alpaca_farm_100.json",
            "gpt-j-6b",
            "FsfairX-LLaMA3-RM-v0.1",
            100,
            20,
            128,
            0.7,
        ),
        (
            "./datasets/alpaca_farm_100.json",
            "gpt-j-6b",
            "reward-model-deberta-v3-large-v2",
            100,
            20,
            128,
            0.7,
        ),
        (
            "./datasets/alpaca_farm_100.json",
            "gpt-j-6b",
            "RM-Mistral-7B",
            100,
            20,
            128,
            0.8,
        ),
        # ("./datasets/alpaca_farm_100.json", "gpt2-xl", "ArmoRM-Llama3-8B-v0.1", 100, 20, 128, 0.8),
        # ("./datasets/alpaca_farm_100.json", "gpt2-xl", "FsfairX-LLaMA3-RM-v0.1", 100, 20, 128, 0.7),
        # ("./datasets/alpaca_farm_100.json", "gpt2-xl", "reward-model-deberta-v3-large-v2", 100, 20, 256, 0.8),
        # ("./datasets/alpaca_farm_100.json", "gpt2-xl", "RM-Mistral-7B", 100, 20, 128, 0.8),
        (
            "./datasets/alpaca_farm_100.json",
            "Meta-Llama-3-8B",
            "ArmoRM-Llama3-8B-v0.1",
            100,
            20,
            256,
            0.5,
        ),
        (
            "./datasets/alpaca_farm_100.json",
            "Meta-Llama-3-8B",
            "FsfairX-LLaMA3-RM-v0.1",
            100,
            20,
            1024,
            0.6,
        ),
        (
            "./datasets/alpaca_farm_100.json",
            "Meta-Llama-3-8B",
            "RM-Mistral-7B",
            100,
            20,
            128,
            0.8,
        ),
    ],
}


def get_queue_output() -> str:
    queue_output = ""
    while len(queue_output) == 0:
        try:
            result = subprocess.run(
                ["squeue", "-u", USERNAME], capture_output=True, text=True
            )
            if result.returncode != 0:
                print(f"WARNING: error occurred in calling squeue - retrying...")
                sleep(10)
                continue
            queue_output = result.stdout
        except:
            sleep(10)
    return queue_output


def get_gpu_count(queue_output: str) -> int:
    gpu_count = 0
    split_output = queue_output.split()
    for item in split_output:
        if "SpReGPU" in item:
            gpu_count += int(item[7:])
    return gpu_count


def create_new_job(cluster: str, idx: int) -> bool:
    job_list = JOBS[cluster]
    job_to_run = job_list[idx]
    output_folder = get_output_folder_from_tuple(job_to_run, cluster)
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    if len(os.listdir(output_folder)) >= 100:
        return False
    job_command = get_job_command_from_tuple(job_to_run, output_folder)
    slurm_filename = f"{cluster}.slurm"
    try:
        slurm_contents = read_file(slurm_filename)
        new_lines = switch_job(cluster, job_command, slurm_contents, 1)
        write_to_file(slurm_filename, new_lines)
        subprocess.run(["sbatch", slurm_filename])
        return True
    except Exception as e:
        print(e)
        return False


def get_output_folder_from_tuple(job_tuple: tuple, cluster: str) -> str:
    (
        dataset_name,
        LLM_name,
        RM_name,
        num_trajectories,
        batch_size,
        decision_token,
        rejection_rate,
    ) = job_tuple
    dataset_code = "AF" if "alpaca" in dataset_name else "HH"
    return f"output_{cluster}_SR_{dataset_code}_{LLM_name}_{RM_name}_{decision_token}_{rejection_rate}"


def get_job_command_from_tuple(job_tuple: tuple, output_folder: str) -> str:
    (
        dataset_name,
        LLM_name,
        RM_name,
        num_trajectories,
        batch_size,
        decision_token,
        rejection_rate,
    ) = job_tuple
    if "gpt2-xl" in LLM_name:
        max_length = 1_024
    elif "gpt-j-6b" in LLM_name:
        max_length = 2_048
    else:
        max_length = 8_000
    job_command = (
        f"python -m algorithm.main "
        + f"--data_filename {dataset_name} "
        + f"--output_folder {output_folder} "
        + f"--llm_name {LLM_name} "
        + f"--reward_model_name {RM_name} "
        + f"--num_trajectories {num_trajectories} "
        + f"--speculative_rejection "
        + f"--decision_token {decision_token} "
        + f"--rejection_rate {rejection_rate} "
        + f"--max_tokens {max_length} "
        + f"--batch_size {batch_size} "
        + f"--seed {0} "
        + f"--top_k {50} "
        + f"--top_p {1.0} "
        + f"--temperature {1.0} "
    )
    return job_command


def read_file(filename: str) -> list[str]:
    with open(filename, "r") as f:
        contents = f.readlines()
    return contents


def switch_job(
    cluster: str, job_to_run: str, slurm_contents: list[str], num_gpus: int
) -> list[str]:
    new_lines: list[str] = []
    for line in slurm_contents:
        if "python" in line or "accelerate" in line:
            new_lines.append(f"{job_to_run}\n")
        elif "#SBATCH --job-name=" in line:
            new_lines.append(f"#SBATCH --job-name=SpReGPU{num_gpus}\n")
        elif "#SBATCH --gres=gpu:" in line:
            new_lines.append(f"#SBATCH --gres=gpu:{num_gpus}\n")
        else:
            new_lines.append(line)
    return new_lines


def write_to_file(filename: str, new_lines: list[str]) -> None:
    os.remove(filename)
    with open(filename, "w") as f:
        f.writelines(new_lines)


def main() -> None:
    queue_output = get_queue_output()
    A100_running_jobs = queue_output.count("A100")
    gpu_count = get_gpu_count(queue_output)
    A100_index = H100_index = 0

    while len(JOBS["A100"]) + len(JOBS["H100"]) > 0:
        if len(JOBS["A100"]) > 0 and A100_running_jobs < MAX_A100_JOBS:
            job_created = create_new_job("A100", A100_index)
            if not job_created:
                JOBS["A100"].pop(A100_index)
                if A100_index >= len(JOBS["A100"]):
                    A100_index = 0
            else:
                A100_index = (
                    (A100_index + 1) % len(JOBS["A100"]) if len(JOBS["A100"]) > 0 else 0
                )
        if len(JOBS["H100"]) > 0 and gpu_count < MAX_H100_GPUS:
            job_created = create_new_job("H100", H100_index)
            if not job_created:
                JOBS["H100"].pop(H100_index)
                if H100_index >= len(JOBS["H100"]):
                    H100_index = 0
            else:
                # FIXME: here, we just keep running the same job over and over
                # H100_index = (
                #     (H100_index + 1) % len(JOBS["H100"]) if len(JOBS["H100"]) > 0 else 0
                # )
                pass
        sleep(10)
        queue_output = get_queue_output()
        A100_running_jobs = queue_output.count("A100")
        gpu_count = get_gpu_count(queue_output)


if __name__ == "__main__":
    main()
