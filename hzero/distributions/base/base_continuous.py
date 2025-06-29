from abc import ABC, abstractmethod
from typing import Optional


class BaseContinuous(ABC):
    def __init__(self, trials: Optional[int] = None):
        self.trials = trials

    @abstractmethod
    def density(self, x: float) -> float:
        pass

    @abstractmethod
    def cumulative_prob(self, x: float) -> float:
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
    def plot(self):
        pass

    @abstractmethod
    def cumulative_plot(self):
        pass