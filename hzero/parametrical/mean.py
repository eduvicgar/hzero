"""
Module for hypothesis testing of a population mean.

This module defines the `Mean` class, which performs hypothesis testing on a single sample mean.

It supports one-tailed and two-tailed tests, automatic calculation of the test statistic,
critical value, and p-value, and can display a plot of the distribution with relevant highlights.
"""
from typing import Optional, Literal
from dataclasses import dataclass
import math
import numpy as np
import numpy.typing as npt
from hzero.distributions import Normal, TStudent

@dataclass
class HypothesisParam:
    """


    :param tail: Type of test tail - "left", "right", or "bilateral".
    :param hzero_mean: Null hypothesis mean value.
    :param std: Known population standard deviation. If None, uses sample standard deviation.
    :param significance: Significance level for the test. If None, p-value is used instead.
    """
    hzero_mean: float
    std: Optional[float] = None
    significance: Optional[float] = None
    tail: Literal["left", "right", "bilateral"] = "bilateral"

class Mean:
    """
    Performs a hypothesis test for the population mean.

    The test can be done assuming known population standard deviation
    or unknown standard deviation, depending on the parameters.

    :param pob_data: Sample data as a NumPy array.
    :param parameters: Configuration object.
    """
    def __init__(self,
                 pob_data: npt.NDArray[float],
                 parameters: HypothesisParam) -> None:
        self.__pob_data = pob_data
        self.__hzero_mean = parameters.hzero_mean
        self.__std = parameters.std
        self.__alpha = parameters.significance
        self.__tail = parameters.tail

        self.__distribution = None
        self.__statictic = None
        self.__critical_value = None
        self.__p_value = None

        self._calculate_param()

    @property
    def pob_mean(self):
        return np.mean(self.__pob_data)

    @property
    def n(self):
        return self.__pob_data.shape[0]

    @property
    def quasivariance(self):
        return math.sqrt(np.sum((self.__pob_data - self.pob_mean) ** 2) /
                                         (self.n - 1))

    def _calculate_param(self) -> None:
        """
        Computes all necessary parameters for the hypothesis test.

        Sets the appropriate distribution (normal or t-distribution), calculates the test statistic,
        critical value (if alpha is given), and p-value.
        """
        if self.__std:
            self.__distribution = Normal(0, 1)
            self.__statictic = ((self.pob_mean - self.__hzero_mean) /
                                (self.__std / math.sqrt(self.n)))
        else:
            self.__distribution = TStudent(self.n - 1)
            self.__statictic = ((self.pob_mean - self.__hzero_mean) /
                                (self.quasivariance / math.sqrt(self.n)))
        self.__p_value = self.__distribution.p_value(self.__statictic, self.__tail)

        if self.__alpha:
            self.__critical_value = self.__distribution.critical_value(self.__alpha,
                                                                       self.__tail == "bilateral")

    def hypothesis(self) -> str:
        """
        Returns the result of the hypothesis test.

        If alpha is provided, the decision is based on the critical value.
        Otherwise, it uses the p-value for inference.

        :return: Conclusion of the hypothesis test.
        """
        if self.__alpha:
            if abs(self.__critical_value) < abs(self.__statictic):
                return "Reject (via critical value)"
            return "No reject (via critical value)"

        if self.__p_value <= 0.01:
            return "Reject (via p-value)"
        if 0.01 > self.__p_value > 0.2:
            return "Doubtful region (via p-value)"
        return "No reject (via p-value)"

    def summary(self) -> str:
        """
        Generates a detailed summary of the hypothesis test.

        Includes hypotheses, distribution used, test statistic, p-value
        and the test conclusion.

        :return: Formatted string with test summary.
        """
        return f"""
        Parameter: mean
        Standard Deviation: {self.__std if self.__std else "unknown"}\n
        Null hypothesis: mean = {self.__hzero_mean}
        Alternative hypothesis: {"mean < " + str(self.__hzero_mean) if self.__tail == "left" \
                                 else "mean > " + str(self.__hzero_mean) if self.__tail == "right" \
                                 else "mean != " + str(self.__hzero_mean)}\n
        Distribution: {self.__distribution}
        Statistic: {self.__statictic}
        P-value: {self.__p_value}\n
        Hypothesis conclusion: {self.hypothesis()}
        """

    def show(self) -> None:
        """
        Displays a plot of the distribution used in the test.

        Highlights the test statistic, rejection region(s), and the significance level.
        """
        self.__distribution.plot(d=self.__statictic, alpha=self.__alpha, tail=self.__tail)

if __name__ == '__main__':
    config_data = {
        "hzero_mean": 40,
        "tail": "bilateral",
        "significance": 0.05
    }

    data = np.array([42, 39, 41, 38, 40, 43, 39, 37, 44, 41])
    config = HypothesisParam(**config_data)
    test = Mean(pob_data=data, parameters=config)
    print(test.summary())
    test.show()
