from dataclasses import dataclass
import math
from .errors import invalid


@dataclass(frozen=True)
class FieldRule:
    unit: str | None = None
    value_type: str = "number"
    positive: bool = False
    minimum: float | None = None
    maximum: float | None = None

    def validate(self,value,path):
        if self.value_type=="choice":
            if type(value) not in (str,int,float) or isinstance(value,bool): invalid(path,"scalar choice required")
        elif type(value) not in (int,float) or not math.isfinite(value): invalid(path,"finite number required")
        if self.value_type=="integer" and type(value) is not int: invalid(path,"integer required")
        if self.positive and value<=0: invalid(path,"positive value required")
        if self.minimum is not None and value<self.minimum: invalid(path,"below minimum")
        if self.maximum is not None and value>self.maximum: invalid(path,"above maximum")


@dataclass(frozen=True)
class Fixed:
    value: object
    unit: str | None = None


@dataclass(frozen=True)
class Range:
    minimum: float
    maximum: float
    unit: str | None = None


@dataclass(frozen=True)
class Distribution:
    name: str
    arguments: dict


@dataclass(frozen=True)
class ParameterRef:
    id: str
