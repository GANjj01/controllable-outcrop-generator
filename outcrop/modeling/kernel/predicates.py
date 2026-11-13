"""Exact signs of the stored binary inputs, including sub-tolerance features."""
from fractions import Fraction
from functools import cmp_to_key
from math import gcd,lcm
import numpy as np


def rational(x): return x if isinstance(x,Fraction) else Fraction(x) if isinstance(x,int) else Fraction.from_float(float(x))
def point(p): return tuple(rational(x) for x in p)
def sub(a,b): return tuple(x-y for x,y in zip(a,b))
def add(a,b): return tuple(x+y for x,y in zip(a,b))
def scale(a,s): return tuple(x*s for x in a)
def dot(a,b): return sum(x*y for x,y in zip(a,b))
def cross(a,b): return (a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])
def sign(x): return (x>0)-(x<0)
def orient2d(a,b,c):
    a,b,c=map(point,(a,b,c));return sign((b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]))
def orient3d(a,b,c,d):
    a,b,c,d=map(point,(a,b,c,d));return sign(dot(sub(b,a),cross(sub(c,a),sub(d,a))))
def plane_value(n,offset,p): return dot(point(n),point(p))-rational(offset)
# Filtered exact sign: a float evaluation decides when its value exceeds a rigorous bound on its
# rounding error (input conversion, three products, three sums: < 8 ulp of the absolute sum, plus
# an underflow term); otherwise the exact rational value decides. The result equals
# sign(plane_value(...)) for every input.
_FILTER_REL=1e-14;_FILTER_ABS=1e-300
# Correctly rounded like float(Fraction) (int / int), without the generic numbers.Rational path.
def _fl(x): return x._numerator/x._denominator if x.__class__ is Fraction else float(x)
def plane_sign(n,offset,p):
    try:
        n0,n1,n2=_fl(n[0]),_fl(n[1]),_fl(n[2]);p0,p1,p2=_fl(p[0]),_fl(p[1]),_fl(p[2]);o=_fl(offset)
    except OverflowError:return sign(plane_value(n,offset,p))
    a=n0*p0;b=n1*p1;c=n2*p2;v=a+b+c-o
    bound=_FILTER_REL*(abs(a)+abs(b)+abs(c)+abs(o))+_FILTER_ABS*(abs(n0)+abs(n1)+abs(n2)+abs(p0)+abs(p1)+abs(p2)+1.)
    if v>bound and bound<1e300:return 1
    if v<-bound and bound<1e300:return -1
    return sign(plane_value(n,offset,p))


def plane_rows(planes):
    """Float normals and offsets of planes for possibly_on_planes (NaN where not representable)."""
    N=np.empty((len(planes),3));O=np.empty(len(planes))
    for i,p in enumerate(planes):
        try:N[i]=[_fl(x) for x in p.normal];O[i]=_fl(p.offset)
        except OverflowError:N[i]=np.nan;O[i]=np.nan
    return N,O


def filtered_signs(N,O,poly):
    """Signs (planes x vertices) decided by the float filter of plane_sign; 0 where it cannot
    decide (the exact predicate must). Nonzero entries equal the exact signs."""
    try:P=np.array([[_fl(x) for x in v] for v in poly],float).reshape(-1,3)
    except OverflowError:return np.zeros((len(O),len(poly)),np.int64)
    prod=N[:,None,:]*P[None,:,:]
    v=prod.sum(-1)-O[:,None]
    bound=_FILTER_REL*(np.abs(prod).sum(-1)+np.abs(O)[:,None])+_FILTER_ABS*(np.abs(N).sum(1)[:,None]+np.abs(P).sum(1)[None,:]+1.)
    ok=bound<1e300
    return np.where(ok&(v>bound),1,np.where(ok&(v<-bound),-1,0)).astype(np.int64)


def possibly_on_planes(N,O,poly):
    """Mask of the planes (rows of plane_rows) that every vertex of poly might lie on. A plane
    outside the mask has a vertex whose plane_sign is provably nonzero (same filter bound, any
    summation order); exact predicates must decide the planes inside it."""
    try:P=np.array([[_fl(x) for x in v] for v in poly],float).reshape(-1,3)
    except OverflowError:return np.ones(len(O),bool)
    prod=N[:,None,:]*P[None,:,:]
    v=prod.sum(-1)-O[:,None]
    bound=_FILTER_REL*(np.abs(prod).sum(-1)+np.abs(O)[:,None])+_FILTER_ABS*(np.abs(N).sum(1)[:,None]+np.abs(P).sum(1)[None,:]+1.)
    off=(np.abs(v)>bound)&(bound<1e300)
    return ~off.any(1)


def canonical_plane(normal,offset):
    q=point(normal)+(rational(-offset),);den=lcm(*(x.denominator for x in q))
    ints=tuple(int(x*den) for x in q);common=gcd(*ints);ints=tuple(x//common for x in ints)
    first=next(x for x in ints if x);return tuple(x if first>0 else -x for x in ints)


def polygon_normal(poly):
    for i in range(1,len(poly)-1):
        n=cross(sub(poly[i],poly[0]),sub(poly[i+1],poly[0]))
        if any(n): return n
    return (Fraction(0),)*3


def sort_coplanar(points,normal):
    pts=sorted(set(points));drop=max(range(3),key=lambda i:abs(normal[i]));axes=[i for i in range(3) if i!=drop]
    center=tuple(sum(p[j] for p in pts)/len(pts) for j in axes)
    def vector(p): return (p[axes[0]]-center[0],p[axes[1]]-center[1])
    def half(v): return 0 if v[1]>0 or (v[1]==0 and v[0]>=0) else 1
    def cmp(a,b):
        u,v=vector(a),vector(b);h=half(u)-half(v)
        if h: return h
        turn=u[0]*v[1]-u[1]*v[0]
        return -sign(turn) if turn else sign(dot(u,u)-dot(v,v))
    ordered=tuple(sorted(pts,key=cmp_to_key(cmp)))
    if dot(polygon_normal(ordered),normal)<0: ordered=tuple(reversed(ordered))
    return ordered


def facet_key(poly):
    return tuple(sorted(set(poly)))


def affine_coefficients(vertices,values):
    """Exact affine interpolant of stored binary nodal values on a tetra."""
    v=tuple(point(p) for p in vertices);f=point(values)
    rows=tuple(sub(p,v[0]) for p in v[1:]);rhs=sub(f[1:],(f[0],)*3)
    det=dot(rows[0],cross(rows[1],rows[2]))
    if not det:raise ValueError("degenerate source tetra")
    n=tuple(sum(rhs[j]*cross(rows[(j+1)%3],rows[(j+2)%3])[i] for j in range(3))/det for i in range(3))
    return n,dot(n,v[0])-f[0]
