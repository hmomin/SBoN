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
