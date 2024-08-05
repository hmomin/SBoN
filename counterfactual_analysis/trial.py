import numpy as np


class Trial(object):
    def __init__(
        self,
        bon_tokens: int,
        bon_max_score: float,
        absolute_difference: float,
        rejection_rate: float,
        decision_token: int,
    ) -> None:
        self.bon_tokens = bon_tokens
        self.bon_max_score = bon_max_score
        self.absolute_difference = absolute_difference
        self.rejection_rate = rejection_rate
        self.decision_token = decision_token
        self.sbon_tokens = -1
        self.sbon_max_score = -np.inf

    def __repr__(self) -> str:
        return f"Trial(\n    bon_tokens: {self.bon_tokens},\n    bon_max_score: {self.bon_max_score},\n    absolute_difference: {self.absolute_difference},\n    rejection_rate: {self.rejection_rate},\n    decision_token: {self.decision_token},\n    sbon_tokens: {self.sbon_tokens},\n    sbon_max_score: {self.sbon_max_score}\n)"

    def update(
        self,
        sbon_tokens: int,
        sbon_max_score: float,
    ) -> None:
        self.sbon_tokens = sbon_tokens
        self.sbon_max_score = sbon_max_score


class TrialCollector(object):
    def __init__(
        self, rejection_rates: list[float], decision_tokens: list[int]
    ) -> None:
        self.trial_collection: dict[str, list[Trial]] = {}
        for rejection_rate in rejection_rates:
            for decision_token in decision_tokens:
                key = f"{rejection_rate}_{decision_token}"
                self.trial_collection[key] = []

    def __repr__(self) -> str:
        return f"TrialCollector(\n    trial_collection: {self.trial_collection}\n)"

    def add_trial(self, trial: Trial) -> None:
        key = f"{trial.rejection_rate}_{trial.decision_token}"
        self.trial_collection[key].append(trial)

    def consolidate_stats(self) -> None:
        for key, trials in self.trial_collection.items():
            bon_tokens = 0
            sbon_tokens = 0
            suboptimalities: list[float] = []
            for trial in trials:
                print(trial)
                raise
                bon_tokens += trial.bon_tokens
                bon_max_score = max(bon_max_score, trial.bon_max_score)
                sbon_tokens += trial.sbon_tokens
                sbon_max_score = max(sbon_max_score, trial.sbon_max_score)
            print(
                f"key: {key}, bon_tokens: {bon_tokens}, bon_max_score: {bon_max_score}, sbon_tokens: {sbon_tokens}, sbon_max_score: {sbon_max_score}"
            )
