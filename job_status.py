import argparse
import os
import subprocess
import time
from datetime import datetime
from counterfactual_generation.generate_job_hogger import (
    JOBS as generate_jobs,
    get_output_folder_from_tuple as generate_get_output_folder_from_tuple,
)
from algorithm.spec_rej_job_hogger import (
    JOBS as sr_jobs,
    get_output_folder_from_tuple as sr_get_output_folder_from_tuple,
)


USERNAME = "my0049"


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no_empty",
        action="store_true",
        default=False,
        help="Don't show empty folders",
    )
    parser.add_argument(
        "--no_full", action="store_true", default=False, help="Don't show full folders"
    )
    args = parser.parse_args()
    return args


def main() -> None:
    args = get_args()
    no_empty: bool = args.no_empty
    no_full: bool = args.no_full
    while True:
        os.system("clear")
        schedule_output = subprocess.run(
            ["squeue", "-u", USERNAME, "-S", "P,i", "--start"],
            capture_output=True,
            text=True,
        )
        queue_output = subprocess.run(
            ["squeue", "-u", USERNAME, "-S", "P,i"], capture_output=True, text=True
        )
        print(schedule_output.stdout[:-1])
        print(queue_output.stdout[:-1])
        print(datetime.now().strftime("%H:%M:%S"))
        for key, job_list in generate_jobs.items():
            for job_tuple in job_list:
                output_folder = generate_get_output_folder_from_tuple(job_tuple, key)
                if not os.path.exists(output_folder):
                    os.mkdir(output_folder)
                num_files = len(os.listdir(output_folder))
                if (no_empty and num_files == 0) or (no_full and num_files >= 100):
                    continue
                print(f"    {num_files} files in {output_folder}")
        for key, job_list in sr_jobs.items():
            for job_tuple in job_list:
                output_folder = sr_get_output_folder_from_tuple(job_tuple, key)
                if not os.path.exists(output_folder):
                    os.mkdir(output_folder)
                num_files = len(os.listdir(output_folder))
                if (no_empty and num_files == 0) or (no_full and num_files >= 100):
                    continue
                print(f"    {num_files} files in {output_folder}")
        time.sleep(60)


if __name__ == "__main__":
    main()
