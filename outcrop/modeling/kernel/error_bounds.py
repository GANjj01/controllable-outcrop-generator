from dataclasses import dataclass
import numpy as np
from .types import AABB
from .analytic import LinearField,QuadraticField,OffsetField


@dataclass(frozen=True)
class ApproximationEvidence:
    certified: bool
    value_error: float
    position_error_m: float | None
    gradient_lower: float | None
    reason: str
    possible_zero: bool


def linearization_evidence(leaf,cell):
    box=AABB(tuple(cell.vertices.min(0)),tuple(cell.vertices.max(0)))
    if leaf.field is None:return ApproximationEvidence(True,0.,0.,None,"exact input PL",True)
    f=leaf.field;interval=f.interval(box);possible=interval.lower<=0<=interval.upper
    if not possible:return ApproximationEvidence(interval.certified,0.,0.,None,"interval excludes zero",False)
    if isinstance(f,LinearField):return ApproximationEvidence(True,0.,0.,float(np.linalg.norm(f.normal)),"exact linear",True)
    bound=f.hessian_bound(box) if hasattr(f,"hessian_bound") else None
    if bound is None:return ApproximationEvidence(False,np.inf,None,None,"source partition/derivative bound needed",True)
    delta=.5*bound*box.diameter**2;position=None;lower=None
    if isinstance(f,QuadraticField):
        eigen=np.linalg.eigvalsh(f.matrix)
        # Exact interpolation excess is a barycentric variance. The 3/8
        # factor bounds sum(lambda_i*lambda_j) for four source vertices.
        differences=cell.vertices[:,None,:]-cell.vertices[None,:,:]
        delta=float(np.nextafter(3/8*np.max(np.einsum('...i,ij,...j->...',differences,f.matrix,differences)),np.inf))
        positive=eigen[eigen>np.finfo(float).eps*np.max(np.abs(eigen))*16]
        if len(positive) and eigen.min()>=0 and f.level>delta:
            # Convex quadratic interpolant sandwiches the true ellipsoid
            # between level and level-delta; radial inclusion proves existence.
            position=(np.sqrt(f.level)-np.sqrt(f.level-delta))/np.sqrt(positive.min())
            lower=2*np.sqrt(positive.min()*(f.level-delta))
    return ApproximationEvidence(interval.certified,delta,float(position) if position is not None else None,lower,"quadratic sandwich" if position is not None else "scalar bound only; zero existence not certified",True)
