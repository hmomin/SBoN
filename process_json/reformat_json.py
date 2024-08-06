# Reformats JSON output files to be more human-readable
import json
from glob import glob
from tqdm import tqdm


def get_og_paths() -> list[str]:
    json_paths = glob("./output_*/*.json")
    return json_paths


def main() -> None:
    og_paths = get_og_paths()
    for og_path in tqdm(og_paths):
        with open(og_path, "r") as f:
            some_data = json.load(f)
        with open(og_path, "w") as f:
            json.dump(some_data, f, indent=4)


if __name__ == "__main__":
    main()
