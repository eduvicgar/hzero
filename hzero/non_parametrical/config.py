from hzero.distributions.base import BaseDiscrete
from typing import Optional, Literal
from dataclasses import dataclass


@dataclass
class ChisqPearsonDiscreteParam:
    distribution: BaseDiscrete
    significance: Optional[float] = None
