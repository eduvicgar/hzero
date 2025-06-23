from typing import Optional, Literal
from dataclasses import dataclass


@dataclass
class MeanHypothesisParam:
    """
    Defines a class that contains the parameters for the mean hypothesis test.
    """
    hzero_mean: float
    std: Optional[float] = None
    significance: Optional[float] = None
    tail: Literal["left", "right", "bilateral"] = "bilateral"

@dataclass
class VarianceHypothesisParam:
    """
    Defines a class that contains the parameters for the variance hypothesis test.
    """
    hzero_var: float
    mean: Optional[float] = None
    significance: Optional[float] = None
    tail: Literal["left", "right", "bilateral"] = "bilateral"

@dataclass
class MeanDiffHypothesisParam:
    """
    Defines a class that contains the parameters for the mean difference hypothesis test.
    """
    hzero_meandiff: float
    std1: Optional[float] = None
    std2: Optional[float] = None
    stds_equal: Optional[bool] = None
    significance: Optional[float] = None
    tail: Literal["left", "right", "bilateral"] = "bilateral"

@dataclass
class VariancesRatioHypothesisParam:
    """
    Defines a class that contains the parameters for the variances ratio hypothesis test.
    """
    hzero_vardiff: float
    significance: Optional[float] = None
    tail: Literal["left", "right", "bilateral"] = "bilateral"
