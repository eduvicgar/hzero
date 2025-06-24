from typing import override, Iterator, Tuple
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from base.base_discrete import BaseDiscrete


class Geometric(BaseDiscrete):
    def __init__(self, p: float, max_k: int):
        self.p = p
        self.max_k = max_k

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
    def plot(self) -> None:
        x = np.arange(1, self.max_k + 1)
        y = np.array([self.probability(int(i)) for i in x])
        plt.stem(x, y, basefmt=" ")
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True)) # Fuerza a indicar enteros en el eje x del plot
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
    suma = 0
    for prob in test:
        print(prob)
        suma += prob[1]
    print(suma)
    test.plot()