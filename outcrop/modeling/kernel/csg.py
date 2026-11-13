from dataclasses import dataclass
import numpy as np
from .types import AABB,Classification,IntervalEvidence


@dataclass
class CSGRegion:
    id: str
    kind: str
    operands: tuple
    bbox: AABB | None = None
    def bounds(self):
        if self.bbox is not None: return self.bbox
        boxes=[r.bounds() for r in self.operands]
        if self.kind=="difference": return boxes[0]
        if self.kind=="union":
            return AABB(tuple(np.min([b.minimum for b in boxes],axis=0)),tuple(np.max([b.maximum for b in boxes],axis=0)))
        lo=np.max([b.minimum for b in boxes],axis=0);hi=np.min([b.maximum for b in boxes],axis=0)
        return AABB(tuple(lo),tuple(np.maximum(lo,hi)))
    def values(self,points):
        values=[r.values(points) for r in self.operands]
        if self.kind=="difference": return np.maximum(values[0],-values[1])
        return (np.minimum.reduce if self.kind=="union" else np.maximum.reduce)(values)
    def classify(self,points,tol):
        cs=[r.classify(points,tol) for r in self.operands]
        inside=[c.inside for c in cs];closed=[c.inside|c.boundary for c in cs]
        if self.kind=="difference":
            yes=inside[0]&cs[1].outside;possible=closed[0]&~cs[1].inside
        elif self.kind=="union": yes=np.logical_or.reduce(inside);possible=np.logical_or.reduce(closed)
        else: yes=np.logical_and.reduce(inside);possible=np.logical_and.reduce(closed)
        unknown=np.logical_or.reduce([c.unresolved for c in cs])
        return Classification(yes&~unknown,possible&~yes&~unknown,~possible&~unknown,unknown)
    def interval(self,box):
        v=[r.interval(box) for r in self.operands]
        if self.kind=="difference":lo=max(v[0].lower,-v[1].upper);hi=max(v[0].upper,-v[1].lower)
        else:
            op=min if self.kind=="union" else max;lo=op(x.lower for x in v);hi=op(x.upper for x in v)
        return IntervalEvidence(lo,hi,all(x.certified for x in v),"sign")
    def boundary_leaves(self,box):
        unique={leaf.source_id:leaf for r in self.operands for leaf in r.boundary_leaves(box)}
        return tuple(unique[x] for x in sorted(unique))
