import math
from typing import override, Optional, Sequence
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import expon
from base.base_continuous import BaseContinuous


class Exponential(BaseContinuous):
    def __init__(self, lmbda: Optional[float] = None, data: Optional[Sequence[int]] = None):
        if lmbda is None and data is None:
            raise ValueError("you must specify either lambda or data")
        if lmbda is None:
            lmbda = self._estimate_mle_lmdba(data)
        self.lmbda = lmbda

    @staticmethod
    def _estimate_mle_lmdba(data: Sequence[int]) -> float:
        data = np.array(data)
        if np.any(data < 0):
            raise ValueError("All values must be ≥ 0")
        return 1 / np.mean(data)

    @override
    def density(self, x: float) -> float:
        if x < 0:
            return 0
        return self.lmbda * math.exp(-self.lmbda * x)

    @override
    def cumulative_prob(self, x: float) -> float:
        return 1 - math.exp(-self.lmbda * x)

    @override
    @property
    def mean(self) -> float:
        return 1 / self.lmbda

    @override
    @property
    def var(self) -> float:
        return 1 / self.lmbda**2

    @override
    def plot(self) -> None:
        x = np.linspace(0, expon.ppf(0.999, self.lmbda), 100)
        y = np.array([self.density(i) for i in x])
        plt.plot(x, y)
        plt.show()

    @override
    def cumulative_plot(self) -> None:
        x = np.linspace(0, expon.ppf(0.999, self.lmbda), 100)
        y = np.array([self.cumulative_prob(i) for i in x])
        plt.plot(x, y)
        plt.show()

if __name__ == '__main__':
    test = Exponential(3)
    print(test.mean)
    print(test.var)
    test.plot()
    test.cumulative_plot()