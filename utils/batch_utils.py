import math


def calculate_batch_sizes(num_trajectories: int, batch_size: int) -> list[int]:
    num_batches = math.ceil(num_trajectories / batch_size)
    approximate_batch_size = num_trajectories / num_batches
    floor_batch_size = math.floor(approximate_batch_size)
    remainder = num_trajectories - floor_batch_size * num_batches
    batch_sizes = [floor_batch_size] * num_batches
    for idx in range(remainder):
        batch_sizes[idx] += 1
    return batch_sizes


def get_batches(num_trajectories: int, batch_size: int) -> list[int]:
    full_batches = num_trajectories // batch_size
    batches: list[int] = [batch_size] * full_batches
    if num_trajectories % batch_size > 0:
        batches.append(num_trajectories % batch_size)
    return batches


def main() -> None:
    assert calculate_batch_sizes(100, 20) == [20, 20, 20, 20, 20]
    assert calculate_batch_sizes(100, 19) == [17, 17, 17, 17, 16, 16]
    assert calculate_batch_sizes(17, 20) == [17]
    assert calculate_batch_sizes(20, 20) == [20]
    assert calculate_batch_sizes(21, 20) == [11, 10]


if __name__ == "__main__":
    main()
