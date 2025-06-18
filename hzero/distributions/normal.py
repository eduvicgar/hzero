"""
This module implements a class for the normal distribution.
With this class you can compute critic values, p-values and plot the distribution
using the methods provided to the class.
"""
from typing import Optional, Literal
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import norm
from ..validators import *

class Normal:
    """
    Defines a class of the normal distribution.
    """
    def __init__(self, mean: float, std: float):
        self.mean = mean
        self.std = std

    @validate_alpha
    def critical_value(self, alpha: float, two_tailed: bool) -> float:
        """
        Calculates the critical value from the normal distribution based on the given significance level.

        :param alpha: Significance level (a float between 0 and 1). Determines the probability
                      threshold.
        :param two_tailed: If True, performs a two-tailed test;
                           otherwise, performs a one-tailed test.
        :return: The critical value corresponding to the given alpha level and test type.

        :raises ValueError: If the alpha value is not in the range (0, 1).
        """
        if two_tailed:
            v = norm.ppf(alpha / 2, self.mean, self.std)
        else:
            v = norm.ppf(alpha, self.mean, self.std)
        return v

    @validate_tail(("left", "right", "bilateral"))
    def p_value(self, d: float, tail: Literal["left", "right", "bilateral"]) -> float:
        """
        Calculates the p-value from the normal distribution for a given statistic.

        :param d: The calculated statistic. Can be positive or negative.
        :param tail:
            Specifies the type of hypothesis test:
            - "right": one-tailed test (right side).
            - "left": one-tailed test (left side).
            - "bilateral": two-tailed test.
        :return: The p-value indicating the probability of observing a statistic
                 as extreme as `d` under the null hypothesis.
        """
        if tail == "bilateral":
            return 2 * (1 - norm.cdf(abs(d), self.mean, self.std))
        return norm.cdf(d, self.mean, self.std) if tail == "left" else 1 - norm.cdf(d, self.mean, self.std)

    @validate_alpha
    @validate_tail(("left", "right", "bilateral", None))
    def plot(self,
             d: Optional[float] = None,
             alpha: Optional[float] = None,
             tail: Optional[Literal["right", "left", "bilateral"]] = None) -> None:
        """
        Plots the normal distribution with optional rejection regions highlighted
        based on the given parameters.

        This function can visualize critical regions for hypothesis testing depending on
        the test statistic `d`, significance level `alpha`, and the type of test `tail`.

        :param d:
            If alpha is specified, the test statistic value.
            Otherwise, the critic value used to define the rejection region when
            `tail` is specified.
            If provided, it will be shown as a dashed red line on the plot.
        :param alpha:
            The significance level of the test, must be between 0 and 1. It is used to
            compute critical values and determine the rejection region(s).
        :param tail:
            Specifies the type of hypothesis test:
            - "right": one-tailed test (right side).
            - "left": one-tailed test (left side).
            - "bilateral": two-tailed test.
            If `d` is provided without `alpha`, "bilateral" is not allowed.
        :return:
            None. Displays a plot of the normal distribution and any relevant rejection regions.
        """

        if d and not alpha and tail == "bilateral":
            raise ValueError("Given critic value d, you can't choose a bilateral tail.")

        x = np.linspace(norm.ppf(0.001, self.mean, self.std), norm.ppf(0.999, self.mean, self.std), 1000)
        y = norm.pdf(x, self.mean, self.std)
        plt.plot(x, y, color='black')

        def fill_region(given_condition, given_label):
            x_fill = x[given_condition]
            y_fill = y[given_condition]
            plt.fill_between(x_fill, y_fill, color='lightgrey', hatch='///', edgecolor='black',
                             linewidth=0.0)
            plt.axvline(d, color='red', linestyle='--', label=given_label)
            patch = Patch(facecolor='lightgrey', hatch='///', edgecolor='black',
                          label='Rejection region')
            plt.legend(handles=[
                plt.Line2D([], [], color='red', linestyle='--', label=given_label),
                patch
            ])

        if d and tail in ("right", "left") and not alpha:
            condition = x >= d if tail == "right" else x <= d
            fill_region(condition, f'Critic value = {d}')

        if d and alpha and tail:
            label_statistic = f'Statistic = {d}'
            if tail == "right":
                cv = self.critical_value(alpha, two_tailed=False)
                fill_region(x >= cv, label_statistic)
            elif tail == "left":
                cv = self.critical_value(alpha, two_tailed=False)
                fill_region(x <= cv, label_statistic)
            else:
                cv = self.critical_value(alpha, two_tailed=True)
                fill_region(x <= -cv, label_statistic)
                fill_region(x >= cv, label_statistic)

        plt.grid(True)
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    n = Normal(0, 1)
    n.plot(d=1.45, tail="left")