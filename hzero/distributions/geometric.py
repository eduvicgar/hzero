from typing import override, Iterator, Tuple, Optional, Sequence
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from .base import BaseDiscrete


class Geometric(BaseDiscrete):
    def __init__(self,
                 max_k: int,
                 p: Optional[float] = None,
                 data: Optional[Sequence[int]] = None,
                 estimated_param: int = 0) -> None:
        if p is None and data is None:
            raise ValueError("you must specify either p or data")
        if p is None:
            p = self._estimate_mle_p(data)
            estimated_param = 1
        self.p = p
        self.max_k = max_k
        self.estimated_param = estimated_param

    @staticmethod
    def _estimate_mle_p(data: Sequence[int]) -> float:
        """
        MLE estimation of p parameter
        """
        data = np.array(data)
        if np.any(data < 1):
            raise ValueError("All values must be ≥ 1.")
        return 1 / np.mean(data)

    @override
    def probability(self, k: int) -> float:
        if k < 1:
            raise ValueError("k must be greater than 1")
        return self.p * (1 - self.p)**(k-1)

    @property
    @override
    def mean(self) -> float:
        return 1 / self.p

    @property
    @override
    def var(self) -> float:
        return (1 - self.p) / self.p**2

    @override
    def cumulative_prob(self, k: int) -> float:
        if k < 1:
            return 0.0
        return sum(self.probability(i) for i in range(1, k + 1))

    @override
    def plot(self) -> None:
        x = np.arange(1, self.max_k + 1)
        y = np.array([self.probability(int(i)) for i in x])
        plt.stem(x, y, basefmt=" ")
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True)) # Fuerza a indicar enteros en el eje x del plot
        plt.show()

    @override
    def cumulative_plot(self) -> None:
        x = np.arange(1, self.max_k + 1)
        y = np.array([self.cumulative_prob(int(k)) for k in x])
        plt.step(x, y, where='post')
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.ylim(0, 1.05)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.show()

    def __iter__(self) -> Iterator[Tuple[int, float]]:
        self._k = 1
        return self

    def __next__(self) -> Tuple[int, float]:
        if self._k > self.max_k:
            raise StopIteration
        result = (self._k, self.probability(self._k))
        self._k += 1
        return result

if __name__ == '__main__':
    test = Geometric(p=1/3, max_k=50)
    print(test.mean)
    print(test.var)
    for prob in test:
        print(prob)
    print(test.cumulative_prob(50))
    test.plot()
    test.cumulative_plot()