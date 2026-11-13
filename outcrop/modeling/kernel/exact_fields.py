import numpy as np
from .analytic import LinearField,QuadraticField
from .predicates import point,dot,sub,rational
from .certified_arithmetic import exact_quadratic_nodes,inverse3,float_upper,float_lower
from .types import IntervalEvidence


class ExactLinearField(LinearField):
    def values(self,points):return np.array([float(dot(point(self.normal),point(p))-rational(self.offset)) for p in points])
    def gradient(self,points):return np.broadcast_to(np.asarray(self.normal,float),(len(points),3)).copy()
    def interval(self,box):
        n=point(self.normal);lo=point(box.minimum);hi=point(box.maximum);offset=rational(self.offset)
        a=sum(v*(l if v>=0 else h) for v,l,h in zip(n,lo,hi))-offset;b=sum(v*(h if v>=0 else l) for v,l,h in zip(n,lo,hi))-offset
        return IntervalEvidence(float_lower(a),float_upper(b),True,self.unit)


class ExactQuadraticField(QuadraticField):
    def values(self,points):return np.array([float(v) for v in exact_quadratic_nodes(self,points)])
    def gradient(self,points):return 2*(np.asarray(points,float)-np.asarray(self.center,float))@np.asarray(self.matrix,float)


def pullback_affine(field,mapping,inverted=False):
    full=getattr(mapping,'exact_matrix',tuple(point(row) for row in mapping.matrix));M=tuple(row[:3] for row in full[:3]);t=tuple(row[3] for row in full[:3]);inverse=inverse3(M)
    if inverted:M,inverse=inverse,M;t=tuple(-dot(row,t) for row in M)
    shift=tuple(-dot(row,t) for row in inverse)
    if isinstance(field,LinearField):
        n=point(field.normal);normal=tuple(sum(inverse[j][i]*n[j] for j in range(3)) for i in range(3));offset=rational(field.offset)-dot(n,shift)
        return ExactLinearField(normal,offset,field.unit)
    if isinstance(field,QuadraticField):
        Q=tuple(point(row) for row in field.matrix);c=point(field.center);center=tuple(dot(row,c)+v for row,v in zip(M,t));matrix=tuple(tuple(sum(inverse[a][i]*Q[a][b]*inverse[b][j] for a in range(3) for b in range(3)) for j in range(3)) for i in range(3))
        return ExactQuadraticField(center,matrix,field.level,field.unit)
    return None
