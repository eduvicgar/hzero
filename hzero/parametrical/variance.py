"""
Module for hypothesis testing of a population variance.

This module defines the `Variance` class, which performs hypothesis testing on a single sample variance.

It supports one-tailed and two-tailed tests, automatic calculation of the test statistic,
critical value, and p-value, and can display a plot of the distribution with relevant highlights.
"""
import math
import numpy as np
import numpy.typing as npt
from config import VarianceHypothesisParam
from hzero.distributions import ChiSquare

class Variance:
    """
    Performs a hypothesis test for the population variance.

    The test can be done assuming known population mean
    or unknown mean, depending on the parameters.

    :param pob_data: Sample data as a NumPy array.
    :param parameters: Configuration object.
    """
    def __init__(self,
                 pob_data: npt.NDArray[float],
                 parameters: VarianceHypothesisParam) -> None:
        self.__pob_data = pob_data
        self.__hzero_var = parameters.hzero_var
        self.__mean = parameters.mean
        self.__alpha = parameters.significance
        self.__tail = parameters.tail

        self.__distribution = None
        self.__statistic = None
        self.__critical_value = None
        self.__p_value = None

        self._calculate_param()

    @property
    def pob_mean(self) -> np.floating:
        """
        Returns the mean of the population data.
        """
        return np.mean(self.__pob_data)

    @property
    def n(self) -> int:
        """
        Returns the number of samples in the population data.
        """
        return self.__pob_data.shape[0]

    @property
    def quasivariance(self) -> float:
        """
        Returns the quasivariance of the population data.
        """
        return np.sum((self.__pob_data - self.pob_mean) ** 2) / (self.n - 1)

    def _calculate_param(self):
        if self.__mean:
            self.__distribution = ChiSquare(self.n)
            self.__statistic = np.sum(((self.__pob_data - self.__mean)
                                       / math.sqrt(self.__hzero_var)) ** 2)
        else:
            self.__distribution = ChiSquare(self.n - 1)
            self.__statistic = ((self.n - 1) / self.__hzero_var) * self.quasivariance
        self.__p_value = self.__distribution.p_value(self.__statistic, self.__tail)

        if self.__alpha:
            self.__critical_value = self.__distribution.critical_value(self.__alpha,
                                                                       self.__tail)

    def hypothesis(self) -> str:
        """
        Returns the result of the hypothesis test.

        If alpha is provided, the decision is based on the critical value.
        Otherwise, it uses the p-value for inference.

        :return: Conclusion of the hypothesis test.
        """
        if self.__alpha:
            if self.__critical_value < self.__statistic:
                return "Reject (statistic > critical value)"
            return "No reject (statistic < critical value)"

        if self.__p_value <= 0.01:
            return "Reject (p-value <= 0.01)"
        if 0.01 > self.__p_value > 0.2:
            return "Doubtful region (p-value in (0.01, 0.2))"
        return "No reject (p-value > 0.2)"

    def report(self) -> str:
        """
        Generates a detailed report of the hypothesis test.

        Includes hypotheses, distribution used, test statistic, p-value
        and the test conclusion.

        :return: Formatted string with test report.
        """
        return f"""
        Parameter: variance
        Mean: {self.__mean if self.__mean else "unknown"}\n
        Null hypothesis: variance = {self.__hzero_var}
        Alternative hypothesis: {"variance < " + str(self.__hzero_var) if self.__tail == "left" \
                                 else "variance > " + str(self.__hzero_var) if self.__tail == "right" \
                                 else "variance != " + str(self.__hzero_var)}\n
        Distribution: {self.__distribution}
        Statistic: {self.__statistic}
        P-value: {self.__p_value}\n
        Hypothesis conclusion: {self.hypothesis()}
        """

    def show(self) -> None:
        """
        Displays a plot of the distribution used in the test.

        Highlights the test statistic, rejection region(s), and the significance level.
        """
        self.__distribution.plot(d=self.__statistic, alpha=self.__alpha, tail=self.__tail)

if __name__ == '__main__':
    config_data = {
        "hzero_var": 1.5,
        "significance": 0.05,
        "tail": "right"
    }

    data = np.array([19.8, 20.1, 21.0, 20.5, 19.6, 20.4, 20.2, 20.7, 19.9, 20.3, 21.2, 20.0])
    config = VarianceHypothesisParam(**config_data)
    test = Variance(pob_data=data, parameters=config)
    print(test.report())
    test.show()