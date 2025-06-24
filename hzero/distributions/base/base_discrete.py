from abc import ABC, abstractmethod


class BaseDiscrete(ABC):
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
    def plot(self):
        pass

    @abstractmethod
    def __iter__(self):
        pass

    @abstractmethod
    def __next__(self):
        pass