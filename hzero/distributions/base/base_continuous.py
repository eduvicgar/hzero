from abc import ABC, abstractmethod


class BaseContinuous(ABC):
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