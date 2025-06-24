import math
from typing import override, Iterator, Tuple
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from base.base_discrete import BaseDiscrete


class Binomial(BaseDiscrete):
    def __init__(self, n: int, p: float):
        self.n = n
        self.p = p

    @override
    def probability(self, k: int) -> float:
        if k < 0 or k > self.n:
            raise ValueError("k must be between 0 and n")
        return math.comb(self.n, k) * self.p**k * (1-self.p)**(self.n - k)

    @property
    @override
    def mean(self) -> float:
        return self.n * self.p

    @property
    @override
    def var(self) -> float:
        return self.n * self.p * (1 - self.p)

    @override
    def cumulative_prob(self, k: int) -> float:
        if k < 0:
            return 0.0
        return sum(self.probability(i) for i in range(0, k + 1))

    @override
    def plot(self) -> None:
        x = np.arange(0, self.n + 1)
        y = np.array([self.probability(int(i)) for i in x])
        plt.stem(x, y, basefmt=" ")
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))  # Fuerza a indicar enteros en el eje x del plot
        plt.show()

    @override
    def cumulative_plot(self) -> None:
        x = np.arange(0, self.n + 1)
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
        if self._k > self.n:
            raise StopIteration
        result = (self._k, self.probability(self._k))
        self._k += 1
        return result

if __name__ == '__main__':
    test = Binomial(n=10, p=0.5)
    print(test.mean)
    print(test.var)
    for prob in test:
        print(prob)
    print(test.cumulative_prob(10))
    test.plot()
    test.cumulative_plot()