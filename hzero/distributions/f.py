"""
This module implements a class for the F distribution.
With this class you can compute critic values, p-values and plot the distribution
using the methods provided to the class.
"""
from typing import Optional, Literal, Tuple
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import f as fdistribution
from ..validators import *


class FSnedecor:
    """
    Defines a class of the F distribution.
    """
    def __init__(self, df1: int, df2: int):
        if df1 < 0 or df2 < 0:
            raise ValueError("The degrees of freedom must be >= 0.")
        self.df1 = df1
        self.df2 = df2

    @validate_alpha
    @validate_tail(("left", "right", "bilateral"))
    def critical_value(self,
                       alpha: float,
                       tail: Literal["left", "right", "bilateral"]) -> float | Tuple[float, float]:
        """
        Calculates the critical value from the F distribution based on
        the given significance level.

        :param alpha: Significance level (a float between 0 and 1).
                      Determines the probability threshold.
        :param tail: Determines the type of hypothesis test:
                     "right": one-tailed test (right side),
                     "left": one-tailed test (left side),
                     "bilateral": two-tailed test.
        :return: The critical F-value corresponding to the given alpha level and test type.
        :raises ValueError: If the alpha value is not in the range (0, 1) or tail is not
                            left or right.
        """
        if tail == "bilateral":
            lower = fdistribution.ppf(alpha / 2, self.df1, self.df2)
            upper = fdistribution.ppf(1 - alpha / 2, self.df1, self.df2)
            return lower, upper
        return fdistribution.ppf(alpha, self.df1, self.df2) if tail == "left" \
            else fdistribution.ppf(1 - alpha, self.df1, self.df2)

    @validate_d_nonnegative
    @validate_tail(("left", "right", "bilateral"))
    def p_value(self, d: float, tail: Literal["left", "right", "bilateral"]) -> float:
        """
        Calculates the p-value from the F distribution for a given F-statistic.

        :param d: The calculated F-statistic (difference measure). Should be positive.
        :param tail: If left, computes a one-tailed left p-value;
                     otherwise, computes a one-tailed right p-value.
        :return: The p-value indicating the probability of observing an F-statistic
                 as extreme as `d` under the null hypothesis.
        """
        if tail == "bilateral":
            p_left = fdistribution.cdf(d, self.df1, self.df2)
            p_right = fdistribution.sf(d, self.df1, self.df2)
            return 2 * min(p_left, p_right)
        return fdistribution.cdf(d, self.df1, self.df2) if tail == "left" \
            else fdistribution.sf(d, self.df1, self.df2)

    @validate_alpha
    @validate_d_nonnegative
    @validate_tail(("left", "right", "bilateral", None))
    def plot(self,
             d: Optional[float] = None,
             alpha: Optional[float] = None,
             tail: Optional[Literal["right", "left", "bilateral"]] = None) -> None:
        """
        Plots the F distribution with optional rejection regions highlighted
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
            None. Displays a plot of the F distribution and
            any relevant rejection regions.
        """
        if d and not alpha and tail == "bilateral":
            raise ValueError("Given critic value d, you can't choose a bilateral tail.")
        x = np.linspace(fdistribution.ppf(0.001, self.df1, self.df2),
                        fdistribution.ppf(0.999, self.df1, self.df2), 1000)
        y = fdistribution.pdf(x, self.df1, self.df2)
        plt.plot(x, y, color="black")

        def fill_region(given_condition, given_label):
            x_fill = x[given_condition]
            y_fill = y[given_condition]
            plt.fill_between(x_fill, y_fill, color="lightgrey", hatch="///",
                             edgecolor="black", linewidth=0.0)
            plt.axvline(d, color="red", linestyle="--", label=given_label)
            patch = Patch(facecolor="lightgrey", hatch="///",
                          edgecolor="black", label="Rejection region")
            plt.legend(handles=[
                plt.Line2D([], [], color="red", linestyle="--", label=given_label),
                patch
            ])

        if d:
            if d < 0:
                raise ValueError("F statistic must be non-negative")
            if tail in ("left", "right") and not alpha:
                condition = x <= d if tail == "right" else x >= d
                fill_region(condition, f"Critic value = {d}")

            if alpha and tail:
                label_statistic = f'Statistic = {round(d, 3)}'
                if tail == "right":
                    cv = self.critical_value(alpha, tail=tail)
                    fill_region(x >= cv, label_statistic)
                elif tail == "left":
                    cv = self.critical_value(alpha, tail=tail)
                    fill_region(x <= cv, label_statistic)
                else:
                    cv_left = self.critical_value(alpha / 2, tail="left")
                    cv_right = self.critical_value(alpha / 2, tail="right")
                    fill_region(x <= cv_left, label_statistic)
                    fill_region(x >= cv_right, label_statistic)

        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def __str__(self) -> str:
        return f"Snedecor's F distribution with {self.df1}, {self.df2} degrees of freedom"

if __name__ == "__main__":
    f = FSnedecor(10, 12)
    f.plot(d=1.25, alpha=0.05, tail="bilateral")
