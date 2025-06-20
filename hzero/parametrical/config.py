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