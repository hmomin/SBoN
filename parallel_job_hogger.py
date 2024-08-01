import os
import random
import subprocess
from pprint import pprint
from time import sleep

USERNAME = "my0049"

# DATASET in ("hh_rlhf_100.json", "alpaca_farm_100.json")
# LLM in ("sft10k", "Meta-Llama-3-8B", "Meta-Llama-3-8B-Instruct")
# RM in ("RM-Mistral-7B", "FsfairX-LLaMA3-RM-v0.1", "ArmoRM-Llama3-8B-v0.1")

MAX_GPUS = 8
# JOB FORMAT: (LLM, RM, BATCH_SIZE, NUM_GPUS, SPECULATIVE_REJECTION, SEED, ALPHA)
JOBS = {
    "A100": [],
    "H100": [
        ("sft10k", "reward-model-human", 45, 8, False, 0, -1.0),
        ("sft10k", "reward-model-human", 45, 8, False, 8, -1.0),
        ("sft10k", "reward-model-human", 45, 8, False, 16, -1.0),
        ("sft10k", "reward-model-human", 45, 8, False, 24, -1.0),
        ("sft10k", "reward-model-human", 45, 4, False, 0, -1.0),
        ("sft10k", "reward-model-human", 45, 2, False, 0, -1.0),
        ("sft10k", "reward-model-human", 45, 1, False, 0, -1.0),
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
    global MAX_GPUS
    gpu_count = 0
    hanshi_found = False
    split_output = queue_output.split()
    for item in split_output:
        if "H100GPU" in item:
            gpu_count += int(item[7:])
        elif "hanshi" in item:
            hanshi_found = True
    MAX_GPUS = 8 if hanshi_found else 16
    return gpu_count


def create_new_job(cluster: str, idx: int) -> bool:
    job_list = JOBS[cluster]
    job_to_run = job_list[idx]
    output_folder = get_output_folder_from_tuple(job_to_run, cluster)
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    if len(os.listdir(output_folder)) >= 805:
        return False
    job_command = get_job_command_from_tuple(job_to_run, output_folder)
    slurm_filename = f"{cluster}.slurm"
    try:
        slurm_contents = read_file(slurm_filename)
        new_lines = switch_job(job_command, slurm_contents, job_to_run[3])
        write_to_file(slurm_filename, new_lines)
        subprocess.run(["sbatch", slurm_filename])
        return True
    except Exception as e:
        print(e)
        return False


def get_output_folder_from_tuple(job_tuple: tuple, cluster: str) -> str:
    (
        LLM_name,
        RM_name,
        batch_size,
        num_gpus,
        speculative_rejection,
        seed,
        alpha,
    ) = job_tuple
    type_code = f"SpR_alpha_{alpha}" if speculative_rejection else "BoN"
    return f"output_{cluster}_{type_code}_AF_{LLM_name}_{RM_name}_{batch_size}_{num_gpus}_seed_{seed}"


def get_job_command_from_tuple(job_tuple: tuple, output_folder: str) -> str:
    (
        LLM_name,
        RM_name,
        batch_size,
        num_gpus,
        speculative_rejection,
        seed,
        alpha,
    ) = job_tuple
    max_tokens = 2_048 if "sft10k" in LLM_name else 8_000
    top_p = 1.0
    gpu_ids = get_gpu_ids(num_gpus)
    port = 42_000 + random.randint(0, 1_000)
    multiple_status = f"--multi_gpu --main_process_port {port} " if num_gpus > 1 else ""
    job_command = (
        f"accelerate launch "
        + multiple_status
        + f"--num_processes {num_gpus} --num_machines 1 "
        + f"--gpu_ids {gpu_ids} --machine_rank 0 --mixed_precision no --dynamo_backend no "
        + f"main.py "
        + f"--output_folder {output_folder} "
        + f"--llm_name {LLM_name} "
        + f"--reward_model_name {RM_name} "
        + f"--max_tokens {max_tokens} "
        + f"--max_gen_tokens {max_tokens} "
        + f"--data_filename ./datasets/alpaca_farm_eval.json "  # FIXME: can switch between 100 and eval here
        + f"--batch_size {batch_size} "
        + f"--seed {seed} "
        + f"--top_k {50} "
        + f"--top_p {top_p} "
        + f"--temperature {1.0} "
        # + f"--record_memory "
    )
    if speculative_rejection:
        job_command += f"--speculative_rejection --alpha {alpha}"
    return job_command


def get_gpu_ids(num_gpus: int) -> str:
    return ",".join([str(i) for i in range(num_gpus)])


def read_file(filename: str) -> list[str]:
    with open(filename, "r") as f:
        contents = f.readlines()
    return contents


def switch_job(job_to_run: str, slurm_contents: list[str], num_gpus: int) -> list[str]:
    new_lines: list[str] = []
    for line in slurm_contents:
        if "python" in line or "accelerate" in line:
            new_lines.append(f"{job_to_run}\n")
        elif "#SBATCH --job-name=" in line:
            new_lines.append(f"#SBATCH --job-name=H100GPU{num_gpus}\n")
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
    H100_index = 0

    while len(JOBS["A100"]) + len(JOBS["H100"]) > 0:
        if len(JOBS["A100"]) > 0 and A100_running_jobs < 2:
            job_created = create_new_job("A100", 0)
            if not job_created:
                JOBS["A100"].pop(0)
        if len(JOBS["H100"]) > 0 and gpu_count < MAX_GPUS:
            job_created = create_new_job("H100", H100_index)
            if not job_created:
                JOBS["H100"].pop(H100_index)
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
