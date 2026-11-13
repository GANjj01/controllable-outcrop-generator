from dataclasses import dataclass, field
from collections.abc import Mapping
from .schema import NumericModelV2,freeze,thaw
from .errors import invalid


@dataclass(frozen=True)
class ResolvedModelV2:
    numeric: NumericModelV2
    root_seed: int
    sample_id: str
    records: tuple = ()
    motions: Mapping = field(default_factory=dict)
    evidence: tuple = ()
    resource_hashes: Mapping = field(default_factory=dict)
    versions: Mapping = field(default_factory=dict)
    selectors: Mapping = field(default_factory=dict)
    attempt_history: tuple = ()
    source_spec: object = None


def materialize_parameters(spec,values):
    def walk(value,path):
        if path in values: return values[path]
        if isinstance(value,Mapping):
            if value.get("kind")=="ref":
                if value["id"] not in values: invalid(path,"unresolved parameter reference")
                return values[value["id"]]
            if value.get("kind")=="fixed": return value["value"]
            if value.get("kind") in {"range","distribution"}: invalid(path,"unresolved random parameter")
            return {k:thaw(freeze(v)) if k in {'metadata','source'} else walk(v,f"{path}.{k}" if path else k) for k,v in value.items()}
        if isinstance(value,(list,tuple)): return [walk(v,f"{path}[{i}]") for i,v in enumerate(value)]
        return value
    return NumericModelV2(freeze(walk(thaw(spec.data),"")))
