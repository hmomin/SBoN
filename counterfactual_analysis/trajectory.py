from typing import Any


class Trajectory(object):
    def __init__(self, trajectory: dict[str, Any]) -> None:
        self.indices: list[int] = trajectory["score_before_indices"]
        self.scores: list[float] = trajectory["partial_scores"]
        self.alive = True

    def __repr__(self) -> str:
        return f"Trajectory(\n    indices: {self.indices},\n    scores: {self.scores},\n    alive: {self.alive}\n)"


def get_total_tokens(trajectories: list[Trajectory]) -> int:
    total_tokens = 0
    for trajectory in trajectories:
        num_tokens = trajectory.indices[-1]
        total_tokens += num_tokens
    return total_tokens


def get_batch_stats(trajectories: list[Trajectory]) -> tuple[float, float]:
    min_score = trajectories[0].scores[-1]
    max_score = trajectories[0].scores[-1]
    for trajectory in trajectories:
        if not trajectory.alive:
            continue
        min_score = min(min_score, trajectory.scores[-1])
        max_score = max(max_score, trajectory.scores[-1])
    return max_score - min_score, max_score
