import os
import subprocess
from pprint import pprint
from time import sleep

USERNAME = "my0049"

# DATASET in ("./datasets/hh_rlhf_100.json", "./datasets/alpaca_farm_100.json")
# LLM in ("gpt-j-6b", "Mistral-7B-v0.3", "Meta-Llama-3-8B")
# RM in ("RM-Mistral-7B", "FsfairX-LLaMA3-RM-v0.1", "ArmoRM-Llama3-8B-v0.1")

MAX_H100_GPUS = 16
MAX_A100_JOBS = 0
# JOB FORMAT: (DATA_FILENAME, LLM, RM, BATCH_SIZE, NUM_TRAJECTORIES, SEED)
JOBS = {
    "A100": [
        # ("./datasets/alpaca_farm_100.json", "gpt-j-6b", "", 20, 1_000, 0),
        # ("./datasets/alpaca_farm_100.json", "Mistral-7B-v0.3", "", 20, 1_000, 0),
        # ("./datasets/alpaca_farm_100.json", "Meta-Llama-3-8B", "", 20, 1_000, 0),
    ],
    "H100": [
        ("./datasets/alpaca_farm_100.json", "gpt-j-6b", "", 20, 1_000, 0),
        ("./datasets/alpaca_farm_100.json", "Mistral-7B-v0.3", "", 20, 1_000, 0),
        ("./datasets/alpaca_farm_100.json", "Meta-Llama-3-8B", "", 20, 1_000, 0),
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
        if "H100GPU" in item:
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
        data_filename,
        LLM_name,
        RM_name,
        batch_size,
        num_trajectories,
        seed,
    ) = job_tuple
    data_code = "AF" if "alpaca" in data_filename else "HH"
    return f"output_{data_code}_{LLM_name}_{RM_name}_{batch_size}_{num_trajectories}_seed_{seed}"


def get_job_command_from_tuple(job_tuple: tuple, output_folder: str) -> str:
    (
        data_filename,
        LLM_name,
        RM_name,
        batch_size,
        num_trajectories,
        seed,
    ) = job_tuple
    max_length = 2_048 if any(s in LLM_name for s in ["sft10k", "gpt-j-6b"]) else 8_000
    job_command = (
        f"python -m counterfactual_generation.generate "
        + f"--data_filename {data_filename} "
        + f"--output_folder {output_folder} "
        + f"--llm_name {LLM_name} "
        + (f"--reward_model_name {RM_name} " if len(RM_name) > 0 else "")
        + f"--num_trajectories {num_trajectories} "
        + f"--max_length {max_length} "
        + f"--batch_size {batch_size} "
        + f"--seed {seed} "
        + f"--top_k {50} "
        + f"--top_p {1.0} "
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
            new_lines.append(f"#SBATCH --job-name={cluster}GPU{num_gpus}\n")
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
                H100_index = (
                    (H100_index + 1) % len(JOBS["H100"]) if len(JOBS["H100"]) > 0 else 0
                )
        sleep(10)
        queue_output = get_queue_output()
        A100_running_jobs = queue_output.count("A100")
        gpu_count = get_gpu_count(queue_output)


if __name__ == "__main__":
    main()
