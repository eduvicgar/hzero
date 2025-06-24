import math
from typing import override, Iterator, Tuple
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from base.base_discrete import BaseDiscrete


class NegativeBinomial(BaseDiscrete):
    def __init__(self, n: int, p: float, max_k: int):
        self.n = n
        self.p = p
        self.max_k = max_k

    @override
    def probability(self, k: int) -> float:
        if k < 0:
            raise ValueError("k must be greater than 0")
        return math.comb(self.n + k - 1, k) * self.p ** self.n * (1 - self.p) ** k

    @property
    @override
    def mean(self) -> float:
        return (self.n * (1 - self.p)) / self.p

    @property
    @override
    def var(self) -> float:
        return (self.n * (1 - self.p)) / self.p**2

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
        x = np.arange(1, self.max_k + 1)
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
    test = NegativeBinomial(n=5, p=0.3, max_k=30)
    print(test.mean)
    print(test.var)
    suma = 0
    for prob in test:
        print(prob)
        suma += prob[1]
    print(suma)
    test.plot()
    test.cumulative_plot()