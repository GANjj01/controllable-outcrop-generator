from dataclasses import dataclass
from itertools import permutations,product
import numpy as np
from .types import AABB,IntervalEvidence
from ..errors import invalid


@dataclass(frozen=True)
class CellComplex:
    vertices: np.ndarray
    cells: np.ndarray
    source_ids: tuple


def freudenthal_cells(bounds,resolution):
    counts=np.asarray(resolution)
    if counts.shape!=(3,) or counts.dtype.kind not in "iu" or np.any(counts<=0):invalid("resolution","three positive integer counts required")
    counts=counts.astype(int)
    axes=[np.linspace(bounds.minimum[j],bounds.maximum[j],int(counts[j])+1) for j in range(3)]
    return freudenthal_axes(axes)


def freudenthal_axes(axes):
    axes=tuple(np.asarray(a,float) for a in axes)
    if len(axes)!=3 or any(a.ndim!=1 or len(a)<2 or not np.isfinite(a).all() or np.any(np.diff(a)<=0) for a in axes):invalid('grid.axes','three finite strictly increasing axes required')
    counts=np.array([len(a)-1 for a in axes]);shape=tuple(counts+1)
    nodes=list(product(*(range(n) for n in shape)));lookup={p:i for i,p in enumerate(nodes)}
    # Pin endpoint bytes to the declared bounds. lo+n*step can round past hi
    # and turn an exterior ROI face into a falsely unpaired interior facet.
    vertices=np.array([[axes[j][p[j]] for j in range(3)] for p in nodes]);cells=[];ids=[]
    for origin in product(*(range(int(n)) for n in counts)):
        for order in permutations(range(3)):
            q=list(origin);indices=[lookup[tuple(q)]]
            for axis in order:q[axis]+=1;indices.append(lookup[tuple(q)])
            cells.append(indices);ids.append("cell:"+":".join(map(str,(*origin,*order))))
    return CellComplex(vertices,np.array(cells,np.int64),tuple(ids))


def anchored_axes(bounds,origin,spacing,budget):
    """Intersect an origin-anchored lattice with exact declared ROI endpoints."""
    from math import floor,ceil,prod
    from ..errors import unresolved
    ranges=[]
    for lo,hi,o in zip(bounds.minimum,bounds.maximum,origin):
        if not np.isfinite((lo-o)/spacing) or not np.isfinite((hi-o)/spacing):unresolved('partition','grid index exceeds arithmetic range')
        ranges.append((floor((lo-o)/spacing)+1,ceil((hi-o)/spacing)))
    counts=[max(0,b-a)+1 for a,b in ranges]
    if 6*prod(counts)>budget.max_cells or prod(n+1 for n in counts)*64>budget.max_bytes:unresolved('partition','source tetra or memory budget exhausted')
    return tuple(np.array([lo,*[o+j*spacing for j in range(a,b) if lo<o+j*spacing<hi],hi]) for lo,hi,o,(a,b) in zip(bounds.minimum,bounds.maximum,origin,ranges))


class GridField:
    def __init__(self,oid,origin,spacing,values,unit="1"):
        self.id=oid;self.origin=np.asarray(origin,float);self.spacing=np.asarray(spacing,float);self.data=np.asarray(values,float);self.unit=unit
        if self.origin.shape!=(3,) or self.spacing.shape!=(3,) or np.any(self.spacing<=0) or self.data.ndim!=3 or min(self.data.shape)<2 or not np.isfinite(self.data).all():invalid(oid,"finite scalar grid with positive spacing required")
        self.axes=tuple(self.origin[j]+np.arange(self.data.shape[j])*self.spacing[j] for j in range(3))
        if any(np.any(np.diff(a)<=0) for a in self.axes):invalid(oid,'grid node spacing is not representable')
    def bounds(self):return AABB(tuple(a[0] for a in self.axes),tuple(a[-1] for a in self.axes))
    def cell_complex(self):return freudenthal_axes(self.axes)
    def _local(self,points):
        p=np.asarray(points,float);shape=np.array(self.data.shape)
        if p.ndim!=2 or p.shape[1]!=3 or not np.isfinite(p).all() or not self.bounds().contains(p).all():invalid(self.id,"point outside scalar-grid validity domain")
        cell=np.column_stack([np.clip(np.searchsorted(a,p[:,j],side='right')-1,0,len(a)-2) for j,a in enumerate(self.axes)])
        frac=np.column_stack([(p[:,j]-a[cell[:,j]])/(a[cell[:,j]+1]-a[cell[:,j]]) for j,a in enumerate(self.axes)]);order=np.argsort(-frac,axis=1,kind="stable");v=[];weights=[];positions=[]
        for c,f,o in zip(cell,frac,order):
            indices=[c.copy()];cursor=c.copy()
            for axis in o:cursor=cursor.copy();cursor[axis]+=1;indices.append(cursor)
            vals=np.array([self.data[tuple(x)] for x in indices]);rank=f[o];w=np.array([1-rank[0],rank[0]-rank[1],rank[1]-rank[2],rank[2]])
            v.append(vals);weights.append(w);positions.append(np.array([[self.axes[j][index[j]] for j in range(3)] for index in indices]))
        return np.array(v),np.array(weights),np.array(positions)
    def values(self,points):
        values,weights,_=self._local(points);return np.sum(values*weights,axis=1)
    def gradient(self,points):
        values,_,pos=self._local(points);return np.array([np.linalg.solve(v[1:]-v[0],f[1:]-f[0]) for f,v in zip(values,pos)])
    def interval(self,box):
        bounds=self.bounds()
        if np.any(np.asarray(box.minimum)<bounds.minimum) or np.any(np.asarray(box.maximum)>bounds.maximum):return IntervalEvidence(-np.inf,np.inf,False,self.unit)
        lo=np.floor((np.asarray(box.minimum)-self.origin)/self.spacing).astype(int);hi=np.ceil((np.asarray(box.maximum)-self.origin)/self.spacing).astype(int)
        block=self.data[tuple(slice(int(a),int(b)+1) for a,b in zip(lo,hi))]
        return IntervalEvidence(float(block.min()),float(block.max()),True,self.unit)
    def hessian_bound(self,box):
        # Only zero inside one source tetra; callers must insert source facets.
        return None
