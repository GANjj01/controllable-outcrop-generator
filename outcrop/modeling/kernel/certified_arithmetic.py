"""Bounded rational enclosures; no floating optimization is a proof."""
from dataclasses import dataclass
from fractions import Fraction
from math import isqrt
import math
import numpy as np
from .predicates import point,dot,cross,sub,rational
from ..errors import unresolved


@dataclass(frozen=True)
class RatInterval:
    lower: Fraction
    upper: Fraction


def float_upper(q):
    f=float(q)
    if not math.isfinite(f):unresolved('arithmetic','certificate overflow')
    return float(np.nextafter(f,np.inf)) if rational(f)<q else f


def float_lower(q):
    f=float(q)
    if not math.isfinite(f):unresolved('arithmetic','certificate overflow')
    return float(np.nextafter(f,-np.inf)) if rational(f)>q else f


def sqrt_interval(q,bits=128):
    q=rational(q)
    if q<0:unresolved('arithmetic','negative square root certificate')
    a,b=isqrt(q.numerator),isqrt(q.denominator)
    if a*a==q.numerator and b*b==q.denominator:
        exact=Fraction(a,b);return RatInterval(exact,exact)
    S=1<<bits;j=isqrt(q.numerator*S*S//q.denominator)
    lo=Fraction(j,S);return RatInterval(lo,lo if lo*lo==q else Fraction(j+1,S))


def inverse3(matrix):
    rows=tuple(point(row) for row in matrix);det=dot(rows[0],cross(rows[1],rows[2]))
    if not det:unresolved('arithmetic','singular exact matrix')
    return tuple(tuple(cross(rows[(j+1)%3],rows[(j+2)%3])[i]/det for j in range(3)) for i in range(3))


def exact_quadratic_nodes(field,vertices):
    Q=tuple(tuple((rational(field.matrix[i][j])+rational(field.matrix[j][i]))/2 for j in range(3)) for i in range(3));c=point(field.center);level=rational(field.level)
    return tuple(dot(z,tuple(dot(row,z) for row in Q))-level for z in (sub(point(v),c) for v in vertices))


def quadratic_bounds(field,vertices):
    Q=tuple(tuple((rational(field.matrix[i][j])+rational(field.matrix[j][i]))/2 for j in range(3)) for i in range(3))
    trace=sum(Q[i][i] for i in range(3));pairs=[Q[i][i]*Q[j][j]-Q[i][j]**2 for i,j in ((0,1),(0,2),(1,2))];det=dot(Q[0],cross(Q[1],Q[2]))
    if min(*(Q[i][i] for i in range(3)),*pairs,det)<0:unresolved('quadratic','stored matrix has no PSD certificate')
    if det>0:
        inverse=inverse3(Q);mu=max(1/max(sum(abs(x) for x in row) for row in inverse),4*det/trace**2)
    elif sum(pairs)>0:mu=sum(pairs)/trace
    elif trace>0:mu=trace
    else:unresolved('quadratic','constant quadratic has no zero witness')
    v=tuple(point(p) for p in vertices);E=Fraction(3,8)*max(dot(d,tuple(dot(row,d) for row in Q)) for a in v for b in v for d in (sub(a,b),))
    level=rational(field.level)
    if level<=E:return E,None,None
    denominator=sqrt_interval(mu).lower*(sqrt_interval(level).lower+sqrt_interval(level-E).lower)
    position=E/denominator;gradient=2*sqrt_interval(mu*(level-E)).lower
    return E,position,gradient


def sin_interval(theta,tolerance=Fraction(1,1<<100),max_terms=128):
    x=rational(theta)
    if abs(x)>16:unresolved('sine','phase exceeds certified Taylor budget; explicit phase reduction required')
    term=x;total=x
    for j in range(max_terms):
        remainder=abs(term*x*x/Fraction((2*j+2)*(2*j+3)))
        if remainder<=tolerance:return RatInterval(max(Fraction(-1),total-remainder),min(Fraction(1),total+remainder))
        term=-term*x*x/Fraction((2*j+2)*(2*j+3));total+=term
    unresolved('sine','Taylor enclosure term budget exhausted')


def periodic_linear_nodes(field,mapping,vertices):
    M=mapping.frame.matrix;inverse=inverse3(M[:3,:3]);u=inverse[0];v=tuple(rational(M[i,2]) for i in range(3));translation=point(M[:3,3]);d=-dot(u,translation);n=point(field.normal)
    A=rational(mapping.A);k=rational(mapping.k);phase=rational(mapping.phase);beta=A*dot(n,v);samples=[];eta=Fraction(0);s=[]
    for p in map(point,vertices):
        a=dot(u,p);s.append(a);base=dot(n,p)-rational(field.offset)
        if not beta:samples.append(base);continue
        sine=sin_interval(k*(a+d)+phase);middle=(sine.lower+sine.upper)/2
        # Dyadic rational proxy samples bound numerator/denominator growth.
        exact=base-beta*middle;sample=Fraction(round(exact*(1<<128)),1<<128)
        eta=max(eta,abs(beta)*(sine.upper-sine.lower)/2+abs(sample-exact));samples.append(sample)
    delta=abs(beta)*k*k*(max(s)-min(s))**2/8+eta
    m2=dot(n,n)-dot(n,u)**2/dot(u,u)
    if not beta:return tuple(samples),Fraction(0),Fraction(0),Fraction(0)
    lower=sqrt_interval(m2).lower
    if lower<=0:unresolved('shear','directional zero witness unavailable')
    return tuple(samples),delta,delta/lower,lower
