import bisect
from counterfactual_analysis.trajectory import (
    Trajectory,
    get_batch_stats,
    get_total_tokens,
)
from pprint import pprint


class SimpleRejection(object):
    def __init__(
        self, trajectories: list[Trajectory], rejection_rate: float, decision_token: int
    ) -> None:
        self.trajectories = trajectories
        self.rejection_rate = rejection_rate
        self.decision_token = decision_token

    def run(self) -> tuple[int, float]:
        decision_scores = self.get_decision_scores()
        N = len(self.trajectories)
        num_to_reject = int(round(self.rejection_rate * N))
        bottom_k_indices = get_bottom_k_indices(decision_scores, num_to_reject)
        self.reject_bottom_trajectories(bottom_k_indices)
        new_num_tokens = get_total_tokens(self.trajectories)
        _, new_max_score = get_batch_stats(self.trajectories)
        return new_num_tokens, new_max_score

    def get_decision_scores(self) -> list[float]:
        decision_scores: list[float] = []
        for trajectory in self.trajectories:
            score_before_indices = trajectory.indices
            partial_scores = trajectory.scores
            if score_before_indices[-1] < self.decision_token:
                decision_score = partial_scores[-1]
            else:
                score_index = score_before_indices.index(self.decision_token)
                decision_score = partial_scores[score_index]
            decision_scores.append(decision_score)
        assert len(decision_scores) == len(self.trajectories)
        return decision_scores

    def reject_bottom_trajectories(self, bottom_k_indices: list[int]) -> None:
        for rejection_index in bottom_k_indices:
            rejected_trajectory = self.trajectories[rejection_index]
            rejected_trajectory.alive = False
            truncate_index = get_truncate_index(
                rejected_trajectory.indices, self.decision_token
            )
            rejected_trajectory.indices = rejected_trajectory.indices[:truncate_index]
            rejected_trajectory.scores = rejected_trajectory.scores[:truncate_index]


def get_bottom_k_indices(scores: list[float], k: int) -> list[int]:
    assert 0 < k < len(scores)
    indexed_scores = [(index, value) for index, value in enumerate(scores)]
    ascending_indexed_scores = sorted(indexed_scores, key=lambda x: x[1])
    bottom_k_indices = [index for index, value in ascending_indexed_scores[:k]]
    return bottom_k_indices


def get_truncate_index(indices: list[int], decision_token: int) -> int:
    decision_token_index = bisect.bisect_right(indices, decision_token)
    return decision_token_index


def main() -> None:
    scores = [3.5, 2.1, 5.9, 7.3, 1.4]
    num_to_reject = 3
    bottom_k_indices = get_bottom_k_indices(scores, num_to_reject)
    assert bottom_k_indices == [4, 1, 0]


if __name__ == "__main__":
    main()
