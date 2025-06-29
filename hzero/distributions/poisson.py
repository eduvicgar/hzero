import math
from typing import override, Iterator, Tuple, Optional, Sequence
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from .base import BaseDiscrete


class Poisson(BaseDiscrete):
    def __init__(self,
                 max_k: int,
                 lmbda: Optional[float] = None,
                 data: Optional[Sequence[int]] = None,
                 estimated_param: int = 0,
                 trials: Optional[int] = None) -> None:
        super().__init__(trials)
        if lmbda is None and data is None:
            raise ValueError("you must specify either lambda or data")
        if lmbda is None:
            lmbda = self._estimate_mle_lmdba(data)
            estimated_param = 1
        self.lmbda = lmbda
        self.max_k = max_k
        self.estimated_param = estimated_param

    @staticmethod
    def _estimate_mle_lmdba(data: Sequence[int]) -> float:
        data = np.array(data)
        if np.any(data < 0):
            raise ValueError("Los datos deben ser ≥ 0.")
        return np.mean(data)

    @override
    def probability(self, k: int) -> float:
        if k < 0:
            raise ValueError("k must be greater than 0")
        return (self.lmbda**k / math.factorial(k)) * math.exp(-self.lmbda)

    @property
    @override
    def mean(self) -> float:
        return self.lmbda

    @property
    @override
    def var(self) -> float:
        return self.lmbda

    @override
    def cumulative_prob(self, k: int) -> float:
        if k < 0:
            return 0.0
        return sum(self.probability(i) for i in range(0, k + 1))

    @override
    def plot(self) -> None:
        x = np.arange(0, self.max_k + 1)
        y = np.array([self.probability(int(i)) for i in x])
        plt.stem(x, y, basefmt=" ")
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))  # Fuerza a indicar enteros en el eje x del plot
        plt.show()

    @override
    def cumulative_plot(self) -> None:
        x = np.arange(0, self.max_k + 1)
        y = np.array([self.cumulative_prob(int(k)) for k in x])
        plt.step(x, y, where='post')
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.ylim(0, 1.05)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.show()

    def __iter__(self) -> Iterator[Tuple[int, float]]:
        self._k = 0
        return self

    def __next__(self) -> Tuple[int, float]:
        if self._k > self.max_k:
            raise StopIteration
        result = (self._k, self.probability(self._k))
        self._k += 1
        return result

if __name__ == '__main__':
    test = Poisson(lmbda=10, max_k=20)
    print(test.mean)
    print(test.var)
    suma = 0
    for prob in test:
        print(prob)
        suma += prob[1]
    print(suma)
    test.plot()
    test.cumulative_plot()