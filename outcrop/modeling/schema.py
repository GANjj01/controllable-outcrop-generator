from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FrozenMap(Mapping):
    entries: tuple
    def __getitem__(self, key):
        for k, v in self.entries:
            if k == key: return v
        raise KeyError(key)
    def __iter__(self): return (k for k, _ in self.entries)
    def __len__(self): return len(self.entries)


def freeze(value):
    if isinstance(value, Mapping): return FrozenMap(tuple((k, freeze(v)) for k,v in value.items()))
    if isinstance(value, (list,tuple)): return tuple(freeze(v) for v in value)
    return value


def thaw(value):
    if isinstance(value, Mapping): return {k: thaw(v) for k,v in value.items()}
    if isinstance(value, tuple): return [thaw(v) for v in value]
    return value


@dataclass(frozen=True)
class ModelSpecV2:
    data: FrozenMap
    def __getitem__(self, key): return self.data[key]
    def get(self, key, default=None): return self.data.get(key, default)


@dataclass(frozen=True)
class NumericModelV2(ModelSpecV2):
    pass


@dataclass(frozen=True)
class ObjectRef:
    id: str
    stage: str = "initial"
    kind: str | None = None


@dataclass(frozen=True)
class SourceSpan:
    text: str
    start: int
    end: int
    source: str | None = None


@dataclass(frozen=True)
class ResourceRef:
    id: str
    path: str
    sha256: str
    media_type: str


@dataclass(frozen=True)
class VerifiedResource:
    ref: ResourceRef
    bytes: bytes


@dataclass(frozen=True)
class ResourceSet:
    by_id: Mapping
    @classmethod
    def empty(cls): return cls({})
