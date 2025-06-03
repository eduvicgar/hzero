"""
This module implements a class for the chi square distribution.
With this class you can compute critic values, p-values and plot the distribution
using the methods provided to the class.
"""
from typing import Optional, Literal
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import chi2


class ChiSquare:
    """
    Defines a class of the chi square distribution.
    """
    def __init__(self, df: int):
        self.df = df

    def critic_value(self, alpha: float, tail: Literal["left", "right"]) -> float:
        """
        Calculates the critical value from the chi square distribution based on
        the given significance level.

        :param alpha: Significance level (a float between 0 and 1).
                      Determines the probability threshold.
        :param tail: Determines the type of hypothesis test:
                     "right": one-tailed test (right side),
                     "left": one-tailed test (left side).
        :return: The critical t-value corresponding to the given alpha level and test type.
        :raises ValueError: If the alpha value is not in the range (0, 1).
        """
        if not 0 < alpha < 1:
            raise ValueError("Alpha value must be between 0 and 1")
        if tail not in ("left", "right"):
            raise ValueError("tail must be one of 'left', 'right'")
        return chi2.ppf(alpha, self.df) if tail == "left" else chi2.ppf(1 - alpha, self.df)


    def p_value(self, d: float, tail: Literal["left", "right"]) -> float:
        """
        Calculates the p-value from the chi square distribution for a given chisq-statistic.

        :param d: The calculated chisq-statistic (difference measure). Should be positive.
        :param tail: If left, computes a one-tailed left p-value;
                     otherwise, computes a one-tailed right p-value.
        :return: The p-value indicating the probability of observing a chisq-statistic
                 as extreme as `d` under the null hypothesis.
        """
        if d < 0:
            raise ValueError("Chi-square statistic must be non-negative")
        if tail not in ("left", "right"):
            raise ValueError("tail must be one of 'left', 'right'")
        return 1 - chi2.cdf(d, self.df) if tail == "right" else chi2.cdf(d, self.df)

    def plot(self,
            d: Optional[float] = None,
            alpha: Optional[float] = None,
            tail: Optional[Literal["right", "left", "bilateral"]] = None) -> None:
        """
        Plots the chi square distribution with optional rejection regions highlighted
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
            None. Displays a plot of the chi square distribution and
            any relevant rejection regions.
        """
        if tail not in ("left", "right", "bilateral", None):
            raise ValueError("tail must be one of 'left', 'right', 'bilateral'")
        if d and not alpha and tail == "bilateral":
            raise ValueError("Given critic value d, you can't choose a bilateral tail.")
        if alpha is not None and not 0 < alpha < 1:
            raise ValueError("Alpha value must be between 0 and 1")

        x = np.linspace(chi2.ppf(0.001, self.df), chi2.ppf(0.999, self.df), 1000)
        y = chi2.pdf(x, self.df)
        plt.plot(x, y, color='black')

        def fill_region(given_condition, given_label):
            x_fill = x[given_condition]
            y_fill = y[given_condition]
            plt.fill_between(x_fill, y_fill, color='lightgrey', hatch='///',
                             edgecolor='black', linewidth=0.0)
            plt.axvline(d, color='red', linestyle='--', label=given_label)
            patch = Patch(facecolor='lightgrey', hatch='///',
                          edgecolor='black', label='Rejection region')
            plt.legend(handles=[
                plt.Line2D([], [], color='red', linestyle='--', label=given_label),
                patch
            ])

        if d and d < 0:
            raise ValueError("Chi-square statistic must be non-negative")
        if tail in ("right", "left") and not alpha:
            condition = x >= d if tail == "right" else x <= d
            fill_region(condition, f'Critic value = {d}')

        if alpha and tail:
            label_statistic = f'Statistic = {d}'
            if tail == "right":
                cv = self.critic_value(alpha, tail=tail)
                fill_region(x >= cv, label_statistic)
            elif tail == "left":
                cv = self.critic_value(alpha, tail=tail)
                fill_region(x <= cv, label_statistic)
            else:
                cv_left = self.critic_value(alpha / 2, tail="left")
                cv_right = self.critic_value(alpha / 2, tail="right")
                fill_region(x <= cv_left, label_statistic)
                fill_region(x >= cv_right, label_statistic)

        plt.grid(True)
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    chisq = ChiSquare(4)
    chisq.plot(d=3.25, alpha=0.05, tail="bilateral")
