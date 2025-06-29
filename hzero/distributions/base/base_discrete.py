from typing import Optional
from abc import ABC, abstractmethod


class BaseDiscrete(ABC):
    def __init__(self, trials: Optional[int], estimated_param: int = 0):
        self.trials = trials
        self.estimated_param = estimated_param

    @abstractmethod
    def probability(self, k: int):
        pass

    @property
    @abstractmethod
    def mean(self):
        pass

    @property
    @abstractmethod
    def var(self):
        pass

    @abstractmethod
    def cumulative_prob(self, k: int):
        pass

    @abstractmethod
    def plot(self):
        pass

    @abstractmethod
    def cumulative_plot(self):
        pass

    @abstractmethod
    def __iter__(self):
        pass

    @abstractmethod
    def __next__(self):
        pass