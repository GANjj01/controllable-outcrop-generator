from dataclasses import dataclass, asdict
from collections.abc import Mapping
from statistics import NormalDist
import hashlib
import json
import math
import numpy as np
from .errors import ModelError,invalid
from .parameters import FieldRule,Fixed,Range,Distribution


@dataclass(frozen=True)
class RandomKey:
    rng_version: str
    root_seed: int
    sample_id: str
    object_id: str
    parameter_path: str
    sampler_version: str
    group_attempt: int
    element_id: str


@dataclass(frozen=True)
class SamplingContext:
    root_seed: int
    sample_id: str
    rng_version: str = "rng-v2"
    sampler_version: str = "sampler-v2"
    object_id: str = "parameter"
    parameter_path: str = "value"
    range_policy: str | None = None
    def __post_init__(self):
        if type(self.root_seed) is not int or self.root_seed<0:invalid('root_seed','nonnegative integer required')
        if not isinstance(self.sample_id,str) or not self.sample_id:invalid('sample_id','nonempty stable sample ID required')
        if self.rng_version!='rng-v2' or self.sampler_version!='sampler-v2':invalid('versions','supported RNG/sampler versions required')


@dataclass(frozen=True)
class ResolutionRecord:
    path: str
    value: object
    key: RandomKey | None
    attempt: int
    sampler: str
    draws: int = 0


@dataclass(frozen=True)
class SampledValue:
    value: object
    record: ResolutionRecord


def rng_for_key(key):
    if type(key.root_seed) is not int or key.root_seed<0: invalid("root_seed","nonnegative integer required")
    if type(key.group_attempt) is not int or key.group_attempt<0: invalid("attempt","nonnegative integer required")
    payload=json.dumps(list(asdict(key).values()),separators=(",",":"),ensure_ascii=False).encode("utf-8")
    entropy=np.frombuffer(hashlib.sha256(payload).digest(),dtype="<u4")
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(entropy)))


def _number(p,key,path,default=None):
    v=p.get(key,default)
    if type(v) not in (int,float) or not math.isfinite(v): invalid(path+"."+key,"finite numeric argument required")
    return v


def sample_parameter(parameter,rule,context,*,attempt,element_id):
    path=context.parameter_path
    if isinstance(parameter,Fixed): parameter={"kind":"fixed","value":parameter.value,"unit":parameter.unit}
    if isinstance(parameter,Range): parameter={"kind":"range","min":parameter.minimum,"max":parameter.maximum,"unit":parameter.unit}
    if isinstance(parameter,Distribution): parameter={"kind":"distribution","name":parameter.name,**parameter.arguments}
    if not isinstance(parameter,Mapping):
        rule.validate(parameter,path)
        return SampledValue(parameter,ResolutionRecord(path,parameter,None,attempt,"fixed"))
    p=dict(parameter); kind=p.get("kind"); draws=0; key=None
    if rule.unit and p.get("unit",rule.unit)!=rule.unit: invalid(path+".unit",f"expected {rule.unit}")
    if kind=="fixed": value=p.get("value"); name="fixed"
    else:
        key=RandomKey(context.rng_version,context.root_seed,context.sample_id,context.object_id,path,context.sampler_version,attempt,str(element_id))
        rng=rng_for_key(key)
        name=p.get("name") if kind=="distribution" else context.range_policy
        if kind=='range' and name=='uniform' and rule.value_type=='integer':name='integer_uniform'
        if kind=="range" and name is None: invalid(path+".range_policy","range_policy required")
        if kind not in {"range","distribution"}: invalid(path,"unresolved or unknown parameter kind")
        draws=1
        if name in {"uniform","midpoint","integer_uniform","circular_uniform","truncated_normal","lognormal"}:
            lo=_number(p,"min",path,0 if name=="lognormal" else None)
            hi=_number(p,"max",path) if p.get("max") is not None else None
            if name not in {"circular_uniform","lognormal"} and (hi is None or lo>hi): invalid(path,"ordered finite bounds required")
        if name=="midpoint": value=(lo+hi)/2; draws=0
        elif name=="uniform": value=float(rng.uniform(lo,hi))
        elif name=="integer_uniform":
            if type(lo) is not int or type(hi) is not int: invalid(path,"integer bounds required")
            value=int(rng.integers(lo,hi+1))
        elif name=="circular_uniform":
            if hi is None: invalid(path,"circular endpoints required")
            width=360 if p.get("full_circle",False) else (hi-lo)%360
            value=float((lo+rng.uniform(0,width))%360)
        elif name=="weighted_choice":
            values=p.get("values",[])
            try:weights=np.asarray(p.get("weights",[]),float)
            except (ValueError,TypeError):invalid(path+'.weights','numeric weights required')
            if weights.ndim!=1 or len(values)!=len(weights) or not len(values) or not np.isfinite(weights).all() or np.any(weights<0) or weights.sum()<=0: invalid(path,"nonnegative finite weights with positive sum required")
            value=values[int(rng.choice(len(values),p=weights/weights.sum()))]
        elif name=="truncated_normal":
            mean=_number(p,"mean",path); std=_number(p,"std",path)
            if std<=0: invalid(path,"positive std required")
            if lo==hi: value=lo
            else:
                normal=NormalDist(mean,std); a,b=normal.cdf(lo),normal.cdf(hi)
                if a>=b or not 0<=a<b<=1: raise ModelError(path,"unresolved","truncated CDF collapsed")
                u=float(rng.uniform(a,b))
                if not 0<u<1: raise ModelError(path,"unresolved","CDF endpoint cannot be inverted")
                value=normal.inv_cdf(u)
        elif name=="lognormal":
            mu=_number(p,"mu",path); sigma=_number(p,"sigma",path)
            if sigma<0 or lo<0 or (hi is not None and hi<lo): invalid(path,"valid positive lognormal bounds required")
            limit=p.get("max_draws",256)
            if type(limit) is not int or limit<=0: invalid(path,"positive integer max_draws required")
            value=None
            for draws in range(1,limit+1):
                with np.errstate(over="ignore"): candidate=float(rng.lognormal(mu,sigma))
                if math.isfinite(candidate) and candidate>=lo and (hi is None or candidate<=hi): value=candidate; break
            if value is None: raise ModelError(path,"constraint_failed","lognormal draw budget exhausted",{"draws":limit})
        else: invalid(path,f"unsupported sampler {name}")
    rule.validate(value,path)
    return SampledValue(value,ResolutionRecord(path,value,key,attempt,str(name),draws))
