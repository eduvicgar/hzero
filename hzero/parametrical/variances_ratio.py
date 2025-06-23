"""
Module for hypothesis testing of a populations' ratio of variances.

This module defines the `VariancesRatio` class, which performs hypothesis testing on two sample means.

It supports one-tailed and two-tailed tests, automatic calculation of the test statistic,
critical value, and p-value, and can display a plot of the distribution with relevant highlights.
"""
import numpy as np
import numpy.typing as npt
from config import VariancesRatioHypothesisParam
from hzero.distributions import FSnedecor


class VariancesRatio:
    """
    Performs a hypothesis test for the populations' ratio of variances.

    The test can be done assuming known population standard deviation
    or unknown standard deviation, depending on the parameters.

    :param x_data: Sample data of x sample as a NumPy array.
    :param y_data: Sample data of y sample as a NumPy array.
    :param parameters: Configuration object.
    """
    def __init__(self,
                 x_data: npt.NDArray[np.float64],
                 y_data: npt.NDArray[np.float64],
                 parameters: VariancesRatioHypothesisParam):
        self.__x_data = x_data
        self.__y_data = y_data
        self.__hzero_vardiff = parameters.hzero_vardiff
        self.__alpha = parameters.significance
        self.__tail = parameters.tail

        self.__distribution = None
        self.__statistic = None
        self.__critical_value = None
        self.__p_value = None

        self._calculate_param()

    @staticmethod
    def pob_mean(pob_data: npt.NDArray[np.float64]) -> np.floating:
        """
        Returns the mean of the population data.
        """
        return np.mean(pob_data)

    @staticmethod
    def n(pob_data: npt.NDArray[np.float64]) -> int:
        """
        Returns the number of samples in the population data.
        """
        return pob_data.shape[0]

    def quasivariance(self, pob_data: npt.NDArray[np.float64]) -> float:
        """
        Returns the quasivariance of the population data.
        """
        return np.sum((pob_data - self.pob_mean(pob_data)) ** 2) / (self.n(pob_data) - 1)

    def _calculate_param(self):
        self.__distribution = FSnedecor(self.n(self.__y_data)-1, self.n(self.__x_data)-1)
        self.__statistic = ((self.quasivariance(self.__y_data) / self.quasivariance(self.__x_data)) *
                            self.__hzero_vardiff)
        self.__critical_value = self.__distribution.critical_value(self.__alpha, self.__tail)
        self.__p_value = self.__distribution.p_value(self.__statistic, self.__tail)

    def hypothesis(self) -> str:
        """
        Returns the result of the hypothesis test.

        If alpha is provided, the decision is based on the critical value.
        Otherwise, it uses the p-value for inference.

        :return: Conclusion of the hypothesis test.
        """
        if self.__alpha:
            if self.__tail == "bilateral":
                return "Reject (statistic in rejection region)" \
                    if self.__statistic < self.__critical_value[0] or self.__statistic > self.__critical_value[1] \
                    else "No reject (statistic out of rejection region)"
            if self.__tail == "left":
                return "Reject (statistic < critical value)" \
                    if self.__critical_value > self.__statistic \
                    else "No reject (statistic > critical value)"
            return "Reject (statistic > critical value)" \
                if self.__critical_value < self.__statistic \
                else "No reject (statistic < critical value)"

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
        Parameter: variance ratio
        Null hypothesis: variance ratio = {self.__hzero_vardiff}
        Alternative hypothesis: {"variance ratio < " + str(self.__hzero_vardiff) if self.__tail == "left" \
                                 else "variance ratio > " + str(self.__hzero_vardiff) if self.__tail == "right" \
                                 else "variance ratio != " + str(self.__hzero_vardiff)}\n
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
    x = np.array([102.1, 98.7, 101.5, 100.2, 99.8, 98.5, 101.0, 100.8])
    y = np.array([99.9, 100.1, 99.7, 100.3, 100.5, 99.6, 100.0, 99.8, 100.2, 100.4])

    config_data = {
        "hzero_vardiff": 1,
        "tail": "bilateral",
        "significance": 0.05,
    }

    config = VariancesRatioHypothesisParam(**config_data)
    test = VariancesRatio(x_data=x, y_data=y, parameters=config)
    print(test.report())
    test.show()
