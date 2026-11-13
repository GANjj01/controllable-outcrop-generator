from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ConstraintEvidence:
    status: str
    witness_points: np.ndarray
    residual: float
    method: str
    error_bound: float | None
    reason: str
