from typing import Sequence, Optional
import numpy as np
import pandas as pd
from hzero.distributions.base import BaseDiscrete
from hzero.distributions import Binomial, ChiSquare

class ChisqPearsonDiscrete:
    def __init__(self,
                 obs_data: Sequence[int],
                 distribution: BaseDiscrete,
                 significance: Optional[float] = None,
                 start: Optional[int] = 0):
        self.distribution = distribution
        self.significance = significance
        self.obs_data = np.array(obs_data)
        self.__start = start
        self.labels = [f"{self.__start + i}" for i in range(len(self.obs_data))]
        self.prob_arr = self._calc_prob_arr()
        self._merge_low_expected_frequencies(umbral=5)

    def _merge_low_expected_frequencies(self, umbral=5):
        obs = self.obs_data.tolist()
        prob = self.prob_arr.tolist()
        labels = self.labels.copy()
        i = 0
        while i < len(prob):
            exp = self.distribution.trials * prob[i]
            if exp < umbral:
                if len(prob) == 1:
                    break
                if i == 0:
                    prob[1] += prob[0]
                    obs[1] += obs[0]
                    labels[1] = f"{labels[0]}–{labels[1]}"
                    del prob[0]
                    del obs[0]
                    del labels[0]
                elif i == len(prob) - 1:
                    prob[-2] += prob[-1]
                    obs[-2] += obs[-1]
                    labels[-2] = f"{labels[-2]}–{labels[-1]}"
                    del prob[-1]
                    del obs[-1]
                    del labels[-1]
                    break
                else:
                    prob[i + 1] += prob[i]
                    obs[i + 1] += obs[i]
                    labels[i + 1] = f"{labels[i]}–{labels[i + 1]}"
                    del prob[i]
                    del obs[i]
                    del labels[i]
            else:
                i += 1

        self.prob_arr = np.array(prob)
        self.obs_data = np.array(obs)
        self.labels = labels
        self.exp_data = self._calc_exp_data()
        self.d_arr = self._calc_d_arr(self.obs_data, self.exp_data)

    @property
    def df(self):
        df = pd.DataFrame({
            'obs_data': self.obs_data,
            'prob_arr': self.prob_arr,
            'exp_data': self.exp_data,
            'd_arr': self.d_arr
        }, index=self.labels)
        return df

    @property
    def k(self):
        return self.df.shape[0]

    @property
    def d(self):
        return np.sum(self.d_arr)

    def _calc_prob_arr(self):
        prob_arr = []
        for i in range(0, self.k):
            prob_arr.append(self.distribution.probability(i))
        return np.array(prob_arr)

    def _calc_exp_data(self):
        exp_data = []
        for prob in self.prob_arr:
            exp_data.append(self.distribution.trials * prob)
        return np.array(exp_data)

    @staticmethod
    def _calc_d_arr(obs_data, exp_data):
        d_arr = []
        for i in range(0, len(obs_data)):
            d_arr.append((obs_data[i] - exp_data[i])**2 / exp_data[i])
        return np.array(d_arr)

    @staticmethod
    def _preprocess_for_chi_square(arr, umbral):
        arr = arr.copy().tolist()  # trabajamos con una copia
        i = 0
        while i < len(arr):
            if arr[i] < umbral:
                if i == 0 and len(arr) > 1:
                    arr[i + 1] += arr[i]
                    del arr[i]
                    # no incrementamos i, ya que el siguiente elemento ocupa esta posición
                elif i == len(arr) - 1 and len(arr) > 1:
                    arr[i - 1] += arr[i]
                    del arr[i]
                    # ya estamos al final, no hace falta avanzar
                    break
                else:
                    arr[i + 1] += arr[i]
                    del arr[i]
                    # no incrementamos i, ya que el nuevo elemento a revisar está en la misma posición
            else:
                i += 1
        return np.array(arr)

    def hypothesis(self):
        chisq = ChiSquare(self.k - self.distribution.estimated_param - 1)
        if self.significance is not None:
            test_value = chisq.critical_value(self.significance, tail="left")
            if self.d > test_value:
                return "Reject (D statistic > Chisq critical value)"
            return "No reject (D statistic < Chisq critical value)"

        test_pvalue = chisq.p_value(self.d, tail="left")
        if test_pvalue <= 0.01:
            return "Reject (p-value <= 0.01)"
        if 0.01 > test_pvalue > 0.2:
            return "Doubtful region (p-value in (0.01, 0.2))"
        return "No reject (p-value > 0.2)"

    def report(self):
        return f"""
                Test: Pearson's chi-squared test
                Null hypothesis: {print(self.distribution)}\n
                Distribution: {self.distribution}
                D statistic: {self.d}
                P-value: {ChiSquare(self.k - self.distribution.estimated_param - 1).p_value(self.d, tail="left")}\n
                Hypothesis conclusion: {self.hypothesis()}
                """

if __name__ == '__main__':
    obs_test = [39, 61, 34, 13, 3]
    distrib_test = Binomial(n=4, data=obs_test, trials=150)
    test = ChisqPearsonDiscrete(obs_data=obs_test,
                                distribution=distrib_test,
                                significance=0.05,
                                start=0)
    print(test.df)
    print(test.report())
