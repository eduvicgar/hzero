from typing import override
import numpy as np
from matplotlib import pyplot as plt
from base.base_continuous import BaseContinuous


class Uniform(BaseContinuous):
    def __init__(self, low: float, high: float):
        if low > high:
            raise ValueError("low must be smaller than high")
        self.low = low
        self.high = high

    @override
    def density(self, x: float) -> float:
        if x < self.low or x > self.high:
            return 0
        return 1/(self.high - self.low)

    @override
    def cumulative_prob(self, x: float) -> float:
        if x < self.low:
            return 0
        if x > self.high:
            return 1
        return (x - self.low)/(self.high - self.low)

    @property
    @override
    def mean(self) -> float:
        return (self.high + self.low) / 2

    @property
    @override
    def var(self) -> float:
        return (self.high - self.low)**2 / 12

    @override
    def plot(self) -> None:
        margin = 0.2 * (self.high - self.low)
        x = np.linspace(self.low - margin, self.high + margin, 1000)
        y = np.array([self.density(i) for i in x])
        plt.plot(x, y)
        plt.show()

    @override
    def cumulative_plot(self):
        margin = 0.2 * (self.high - self.low)
        x = np.linspace(self.low - margin, self.high + margin, 1000)
        y = np.array([self.cumulative_prob(i) for i in x])
        plt.plot(x, y)
        plt.show()

if __name__ == '__main__':
    test = Uniform(2, 8)
    print(test.mean)
    print(test.var)
    test.plot()
    test.cumulative_plot()