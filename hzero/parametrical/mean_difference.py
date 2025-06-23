import math
import numpy as np
import numpy.typing as npt
from config import MeanDiffHypothesisParam
from hzero.distributions import Normal, TStudent


class MeanDiff:
    def __init__(self,
                 x_data: npt.NDArray[np.float64],
                 y_data: npt.NDArray[np.float64],
                 parameters: MeanDiffHypothesisParam):

        self.__x_data = x_data
        self.__y_data = y_data
        self.__hzero_meandiff = parameters.hzero_meandiff
        self.__std1 = parameters.std1
        self.__std2 = parameters.std2
        self.__stds_equal = parameters.stds_equal
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

    def _pooled_variance(self,
                        x_data: npt.NDArray[np.float64],
                        y_data: npt.NDArray[np.float64]) -> float:
        num = (((self.n(x_data)-1) * self.quasivariance(x_data)) +
               ((self.n(y_data)-1) * self.quasivariance(y_data)))

        return num / (self.n(x_data) + self.n(y_data) - 2)

    def _delta(self,
              x_data: npt.NDArray[np.float64],
              y_data: npt.NDArray[np.float64]) -> int:
        a = self.quasivariance(x_data) / self.n(x_data)
        b = self.quasivariance(y_data) / self.n(y_data)
        num = ((self.n(y_data) - 1) * a - (self.n(x_data) - 1) * b)**2
        denom = (self.n(y_data) - 1) * a**2 + (self.n(x_data) - 1) * b**2

        return round(num / denom)

    def _calculate_param(self):
        """
        Computes all necessary parameters for the hypothesis test.

        Sets the appropriate distribution (normal or t-distribution), calculates the test statistic,
        critical value (if alpha is given), and p-value.
        """
        if self.__std1 and self.__std2:
            self.__distribution = Normal(0,1)
            self.__statistic = ((self.pob_mean(self.__x_data) - self.pob_mean(self.__y_data) - self.__hzero_meandiff) /
                                math.sqrt(self.__std1**2 / self.n(self.__x_data) + self.__std2**2 / self.n(self.__y_data)))
            self.__critical_value = self.__distribution.critical_value(self.__alpha, self.__tail == "bilateral")
        if not self.__std1 and not self.__std2:
            if self.__stds_equal:
                self.__distribution = TStudent(self.n(self.__x_data) + self.n(self.__y_data) - 2)
                self.__statistic = ((self.pob_mean(self.__x_data) - self.pob_mean(self.__y_data) - self.__hzero_meandiff) /
                                     math.sqrt(self._pooled_variance(self.__x_data, self.__y_data) * (1/self.n(self.__x_data) + 1/self.n(self.__y_data))))
            else:
                self.__distribution = TStudent(self.n(self.__x_data) + self.n(self.__y_data) - 2 - self._delta(self.__x_data, self.__y_data))
                self.__statistic = ((self.pob_mean(self.__x_data) - self.pob_mean(self.__y_data) - self.__hzero_meandiff) /
                                    math.sqrt(self.quasivariance(self.__x_data) / self.n(self.__x_data) + self.quasivariance(self.__y_data) / self.n(self.__y_data)))
            self.__critical_value = self.__distribution.critical_value(self.__alpha, self.__tail == "bilateral")
        self.__p_value = self.__distribution.p_value(self.__statistic, self.__tail)

    def hypothesis(self) -> str:
        """
        Returns the result of the hypothesis test.

        If alpha is provided, the decision is based on the critical value.
        Otherwise, it uses the p-value for inference.

        :return: Conclusion of the hypothesis test.
        """
        if self.__alpha:
            if abs(self.__critical_value) < abs(self.__statistic):
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
        Parameter: difference of means
        Standard Deviation 1: {self.__std1 if self.__std1 else "unknown"}
        Standard Deviation 2: {self.__std2 if self.__std2 else "unknown"}
        Same standard deviations: {self.__stds_equal}\n
        Null hypothesis: mean diff = {self.__hzero_meandiff}
        Alternative hypothesis: {"mean diff < " + str(self.__hzero_meandiff) if self.__tail == "left" \
                                 else "mean diff > " + str(self.__hzero_meandiff) if self.__tail == "right" \
                                 else "mean diff != " + str(self.__hzero_meandiff)}\n
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
    region_a = np.array([7.1, 6.9, 7.3, 7.2, 6.8, 7.0, 6.7, 7.1, 7.2, 6.9])  # Región A
    region_b = np.array([6.5, 6.3, 6.6, 6.7, 6.2, 6.4, 6.8, 6.6])  # Región B

    config_data = {
        "hzero_meandiff": 0,
        "tail": "bilateral",
        "significance": 0.05,
        "stds_equal": False,
    }

    config = MeanDiffHypothesisParam(**config_data)
    test = MeanDiff(x_data=region_a, y_data=region_b, parameters=config)
    print(test.report())
    test.show()