from dataclasses import dataclass
from fractions import Fraction
import numpy as np
from .predicates import point,rational,dot,cross,sub,add,scale,plane_value,plane_sign,polygon_normal,sort_coplanar,facet_key,canonical_plane
from .native import exact_cut_native,filter_statistics


@dataclass(frozen=True)
class Plane:
    normal: tuple
    offset: object
    source_id: str
    def __post_init__(self):
        from ..errors import invalid
        if len(self.normal)!=3 or not any(self.normal): invalid(self.source_id,"nonzero plane normal required")
    @property
    def key(self): return canonical_plane(self.normal,self.offset)


def _exact_polygon(poly):
    # Public callers can supply floats: retain their original arithmetic and
    # output types. The topology pipeline already supplies exact points.
    return all(isinstance(x,(int,Fraction)) for p in poly for x in p)


def _native_signs(poly,signs):
    return signs is None or (len(signs)==len(poly) and all(
        isinstance(s,(int,np.integer)) and s in (-1,0,1) for s in signs))


def clip_polygon(poly,plane,negative=True,signs=None):
    native=exact_cut_native()
    if native is not None and _exact_polygon(poly) and _native_signs(poly,signs):
        result=native.clip_polygon(poly,(plane.normal,plane.offset),negative,signs)
        filter_statistics()['cpp_clip_calls']+=1
        return result
    # Filtered signs decide the topology; exact values are needed only on crossing edges.
    # ``signs``: precomputed plane_sign of every vertex (shared by the two sides of one cut).
    flip=1 if negative else -1
    signs=[(plane_sign(plane.normal,plane.offset,p) if signs is None else signs[i])*flip for i,p in enumerate(poly)]
    out=[];crossings=[]
    for i,a in enumerate(poly):
        j=(i+1)%len(poly);b=poly[j];sa=signs[i];sb=signs[j]
        if sa<=0: out.append(a)
        if sa*sb<0:
            va=plane_value(plane.normal,plane.offset,a)*flip;vb=plane_value(plane.normal,plane.offset,b)*flip
            p=tuple((va*y-vb*x)/(va-vb) for x,y in zip(a,b));out.append(p);crossings.append(p)
        elif sa==0: crossings.append(a)
    clean=[]
    for p in out:
        if not clean or clean[-1]!=p: clean.append(p)
    if len(clean)>1 and clean[0]==clean[-1]: clean.pop()
    return tuple(clean),tuple(crossings)


def split_polygon(poly,plane,signs=None):
    """Both ordered half-polygons, sharing native conversion and intersections."""
    native=exact_cut_native()
    if native is not None and _exact_polygon(poly) and _native_signs(poly,signs):
        result=native.split_polygon(poly,(plane.normal,plane.offset),signs)
        filter_statistics()['cpp_bisect_calls']+=1
        return result
    return (clip_polygon(poly,plane,signs=signs)[0],
            clip_polygon(poly,plane,False,signs=signs)[0])


def clip_polygon_by_cell(poly,cell,source_id='partition-face'):
    """The original negative halfspace chain, with one native point roundtrip."""
    native=exact_cut_native()
    if native is not None and _exact_polygon(poly):
        cached=getattr(cell,'_native_clip_planes',None)
        if cached is None or cached[0] is not native:
            rows=[]
            for face in cell.polygons:
                n=polygon_normal(face);plane=Plane(n,dot(n,face[0]),source_id)
                rows.append((plane.normal,plane.offset))
            cached=(native,native.prepare_planes(tuple(rows)),len(rows))
            object.__setattr__(cell,'_native_clip_planes',cached)
        result=native.clip_polygon_chain(cached[1],range(cached[2]),poly)
        filter_statistics()['cpp_chain_calls']+=1
        return result
    for face in cell.polygons:
        n=polygon_normal(face);poly,_=clip_polygon(poly,Plane(n,dot(n,face[0]),source_id))
        if len(poly)<3:break
    return poly


@dataclass(frozen=True)
class ConvexCell:
    polygons: tuple
    source_id: str
    constraints: frozenset = frozenset()
    @property
    def exact_vertices(self): return tuple(sorted({p for poly in self.polygons for p in poly}))
    @property
    def vertices(self): return np.asarray(self.exact_vertices,float).reshape(-1,3)
    @property
    def faces(self):
        index={p:i for i,p in enumerate(self.exact_vertices)};return tuple(np.array([index[p] for p in poly],dtype=np.int64) for poly in self.polygons)
    @property
    def exact_volume(self):
        total=Fraction(0)
        for poly in self.polygons:
            for i in range(1,len(poly)-1): total+=dot(poly[0],cross(poly[i],poly[i+1]))/6
        return abs(total)
    @property
    def volume(self): return float(self.exact_volume)
    @property
    def centroid(self):
        ps=self.exact_vertices;return np.array([float(sum(p[j] for p in ps)/len(ps)) for j in range(3)])
    @classmethod
    def from_polygons(cls,polygons,source_id,constraints=frozenset()):
        polys=[tuple(point(p) for p in poly) for poly in polygons if len(set(map(tuple,poly)))>=3]
        polys=[p for p in polys if any(polygon_normal(p))]
        if not polys: return cls((),source_id,constraints)
        pts={p for poly in polys for p in poly};center=tuple(sum(p[j] for p in pts)/len(pts) for j in range(3))
        outward=[]
        for poly in polys:
            if dot(polygon_normal(poly),sub(poly[0],center))<0: poly=tuple(reversed(poly))
            outward.append(poly)
        return cls(tuple(sorted(outward,key=facet_key)),source_id,constraints)
    @classmethod
    def box(cls,minimum,maximum,source_id):
        lo,hi=point(minimum),point(maximum);faces=[]
        for axis in range(3):
            others=[i for i in range(3) if i!=axis]
            for value in (lo[axis],hi[axis]):
                poly=[]
                for a,b in ((0,0),(1,0),(1,1),(0,1)):
                    p=list(lo);p[axis]=value;p[others[0]]=hi[others[0]] if a else lo[others[0]];p[others[1]]=hi[others[1]] if b else lo[others[1]];poly.append(tuple(p))
                faces.append(tuple(poly))
        return cls.from_polygons(faces,source_id)
    @classmethod
    def tetra(cls,vertices,source_id):
        v=tuple(point(p) for p in vertices)
        return cls.from_polygons(tuple(tuple(v[j] for j in range(4) if j!=i) for i in range(4)),source_id)


@dataclass(frozen=True)
class SplitResult:
    cells: tuple
    cap_face_ids: tuple


def split_polytope(poly,plane,tol):
    native=exact_cut_native()
    # Fused splitting bypasses from_polygons. Raw integer cells still need its
    # Fraction normalization, otherwise untouched integer faces divide as floats
    # in exact_volume. Cells built through the normal constructors are rational.
    if native is not None and all(isinstance(x,Fraction)
                                  for face in poly.polygons for p in face for x in p):
        polygons=native.split_polytope(poly.polygons,(plane.normal,plane.offset))
        filter_statistics()['cpp_split_calls']+=1
        if polygons is None:return SplitResult((poly,),())
        return SplitResult(tuple(ConvexCell(p,poly.source_id,poly.constraints|{plane.source_id})
                                 for p in polygons),(plane.source_id,))
    signs=[plane_sign(plane.normal,plane.offset,p) for p in poly.exact_vertices]
    if not any(s<0 for s in signs) or not any(s>0 for s in signs): return SplitResult((poly,),())
    result=[]
    for negative in (True,False):
        faces=[];cap=[]
        for face in poly.polygons:
            clipped,intersection=clip_polygon(face,plane,negative)
            if len(clipped)>=3: faces.append(clipped)
            cap.extend(intersection)
        cap=sort_coplanar(cap,point(plane.normal) if negative else scale(point(plane.normal),-1))
        if len(cap)>=3: faces.append(cap)
        cell=ConvexCell.from_polygons(faces,poly.source_id,poly.constraints|{plane.source_id})
        if cell.exact_volume>0: result.append(cell)
    return SplitResult(tuple(result),(plane.source_id,))
