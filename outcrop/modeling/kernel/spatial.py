from dataclasses import dataclass
import numpy as np
from .types import AABB


class AABBTree:
    """Deterministic binary boxes; no optional proximity engine."""
    def __init__(self,minimum,maximum,leaf_size=8):
        self.minimum=np.asarray(minimum,float);self.maximum=np.asarray(maximum,float);self.leaf_size=leaf_size
        self.root=self._build(np.arange(len(self.minimum),dtype=np.int64)) if len(self.minimum) else None
    def _build(self,ids):
        lo=self.minimum[ids].min(0);hi=self.maximum[ids].max(0)
        if len(ids)<=self.leaf_size:return (lo,hi,ids,None,None)
        axis=int(np.argmax(hi-lo));center=(self.minimum[ids,axis]+self.maximum[ids,axis])/2
        ids=ids[np.lexsort((ids,center))];mid=len(ids)//2
        return (lo,hi,None,self._build(ids[:mid]),self._build(ids[mid:]))
    def query(self,box):
        result=[]
        def visit(node):
            if node is None:return
            lo,hi,ids,left,right=node
            if np.any(hi<box.minimum) or np.any(lo>box.maximum):return
            if ids is None:visit(left);visit(right)
            else:
                mask=np.all(self.maximum[ids]>=box.minimum,axis=1)&np.all(self.minimum[ids]<=box.maximum,axis=1);result.extend(ids[mask])
        visit(self.root);return np.array(sorted(result),np.int64)
