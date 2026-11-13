from dataclasses import dataclass
from typing import Protocol
import numpy as np
from .types import AABB,Tolerance,GeometryRequest
from .grid import GridField,freudenthal_cells
from ..errors import invalid,unresolved,ModelError
from .predicates import point,dot,cross,rational
from .certified_arithmetic import inverse3,float_lower,float_upper,sqrt_interval


@dataclass(frozen=True)
class InverseCandidate:
    points: np.ndarray
    source_cell_ids: np.ndarray
    valid: np.ndarray
    residual_m: np.ndarray


@dataclass(frozen=True)
class MapCertificate:
    status: str
    minimum_det: float
    global_injective: bool
    evidence: dict


class CertifiedMap(Protocol):
    def forward(self,points):...
    def inverse_candidates(self,points,tol):...
    def jacobian(self,points):...
    def image_bounds(self,box):...
    def certify(self,request):...


class AffineMap:
    def __init__(self,oid,matrix):
        self.id=oid;self.matrix=np.asarray(matrix,float)
        if self.matrix.shape!=(4,4) or not np.isfinite(self.matrix).all() or not np.array_equal(self.matrix[3],[0,0,0,1]) or np.linalg.det(self.matrix[:3,:3])<=0:invalid(oid,"orientation-preserving nonsingular affine map required")
        self.inverse_matrix=np.linalg.inv(self.matrix)
        self.exact_matrix=tuple(point(row) for row in self.matrix)
    @classmethod
    def from_exact(cls,oid,matrix):
        result=cls(oid,np.asarray(matrix,float));result.exact_matrix=tuple(point(row) for row in matrix);return result
    def forward(self,p):return np.asarray(p,float)@self.matrix[:3,:3].T+self.matrix[:3,3]
    def inverse_candidates(self,p,tol):
        p=np.asarray(p,float);q=p@self.inverse_matrix[:3,:3].T+self.inverse_matrix[:3,3];res=np.linalg.norm(self.forward(q)-p,axis=1)
        return (InverseCandidate(q,np.full(len(p),-1,np.int64),res<=tol.inverse_residual_m,res),)
    def jacobian(self,p):return np.broadcast_to(self.matrix[:3,:3],(len(p),3,3)).copy()
    def image_bounds(self,box):
        return self._bounds(box,False)
    def preimage_bounds(self,box):
        return self._bounds(box,True)
    def _bounds(self,box,inverse):
        M=tuple(row[:3] for row in self.exact_matrix[:3]);t=tuple(row[3] for row in self.exact_matrix[:3])
        if inverse:M=inverse3(M);t=tuple(-dot(row,t) for row in M)
        p=tuple(tuple(dot(row,point(c))+v for row,v in zip(M,t)) for c in box.corners)
        return AABB(tuple(float_lower(min(c[j] for c in p)) for j in range(3)),tuple(float_upper(max(c[j] for c in p)) for j in range(3)))
    def certify(self,request):
        from .map_certification import _norm_upper
        M=tuple(row[:3] for row in self.exact_matrix[:3]);det=dot(M[0],cross(M[1],M[2]));cond=_norm_upper(M)*_norm_upper(inverse3(M))
        if det<=0:invalid(self.id,'orientation-preserving exact affine map required')
        if cond>rational(request.tolerance.max_condition):unresolved(self.id,"affine condition budget exceeded")
        return MapCertificate("pass",float_lower(det),True,{"method":"exact analytic inverse","condition_upper":float_upper(cond)})


class PeriodicShearMap:
    def __init__(self,spec):
        self.id=spec["id"];self.A=float(spec["amplitude_m"]);self.k=float(spec["wavenumber_per_m"]);self.phase=np.deg2rad(spec.get("phase_deg",0));self.frame=AffineMap(self.id+":frame",spec.get("frame",np.eye(4)))
    def _local(self,p):return np.asarray(p,float)@self.frame.inverse_matrix[:3,:3].T+self.frame.inverse_matrix[:3,3]
    def forward(self,p):
        q=self._local(p);q[:,2]+=self.A*np.sin(self.k*q[:,0]+self.phase);return self.frame.forward(q)
    def inverse_candidates(self,p,tol):
        q=self._local(p);q[:,2]-=self.A*np.sin(self.k*q[:,0]+self.phase);q=self.frame.forward(q);res=np.linalg.norm(self.forward(q)-p,axis=1)
        return (InverseCandidate(q,np.full(len(q),-1,np.int64),res<=tol.inverse_residual_m,res),)
    def jacobian(self,p):
        q=self._local(p);J=np.broadcast_to(np.eye(3),(len(q),3,3)).copy();J[:,2,0]=self.A*self.k*np.cos(self.k*q[:,0]+self.phase)
        return self.frame.matrix[:3,:3]@J@self.frame.inverse_matrix[:3,:3]
    def image_bounds(self,box):
        extension=np.abs(self.A*self.frame.matrix[:3,2]);return AABB(tuple(np.asarray(box.minimum)-extension),tuple(np.asarray(box.maximum)+extension))
    def preimage_bounds(self,box):return self.image_bounds(box)
    def certify(self,request):
        self.frame.certify(request)
        M=tuple(row[:3] for row in self.frame.exact_matrix[:3]);I=inverse3(M);u=tuple(row[2] for row in M);v=I[0]
        # J = I + s*u*v^T and J^-1 = I - s*u*v^T globally.
        norm=sqrt_interval(dot(u,u)).upper*sqrt_interval(dot(v,v)).upper
        condition=(1+abs(rational(self.A)*rational(self.k))*norm)**2
        if condition>rational(request.tolerance.max_condition):unresolved(self.id,'periodic shear condition budget exceeded')
        return MapCertificate("pass",1.,True,{"method":"global analytic periodic shear inverse","condition_upper":float_upper(condition)})


class ComposedMap:
    def __init__(self,maps):self.maps=tuple(maps);self.id="composition:"+":".join(m.id for m in maps)
    def forward(self,p):
        for m in self.maps:p=m.forward(p)
        return p
    def inverse_candidates(self,p,tol):
        candidates=(InverseCandidate(np.asarray(p,float),np.full(len(p),-1,np.int64),np.ones(len(p),bool),np.zeros(len(p))),)
        for m in reversed(self.maps):
            candidates=tuple(InverseCandidate(c.points,c.source_cell_ids,c.valid&prior.valid,c.residual_m+prior.residual_m) for prior in candidates for c in m.inverse_candidates(prior.points,tol))
        return candidates
    def jacobian(self,p):
        q=np.asarray(p,float);J=np.broadcast_to(np.eye(3),(len(q),3,3)).copy()
        for m in self.maps:J=m.jacobian(q)@J;q=m.forward(q)
        return J
    def image_bounds(self,box):
        for m in self.maps:box=m.image_bounds(box)
        return box
    def preimage_bounds(self,box):
        for m in reversed(self.maps):box=m.preimage_bounds(box)
        return box
    def certify(self,request):
        certificates=[m.certify(request) for m in self.maps]
        condition=rational(1)
        for c in certificates:
            if 'condition_upper' not in c.evidence:unresolved(self.id,'composite map lacks a condition certificate')
            condition*=rational(c.evidence['condition_upper'])
        if condition>rational(request.tolerance.max_condition):unresolved(self.id,'composite condition budget exceeded')
        return MapCertificate("pass",float(np.prod([c.minimum_det for c in certificates])),True,{"methods":[c.evidence for c in certificates],"condition_upper":float_upper(condition)})


class DisplacementGridMap:
    def __init__(self,spec):
        self.id=spec["id"];data=np.asarray(spec["values"],float)
        if data.ndim!=4 or data.shape[-1]!=3:invalid(self.id,"NxMxKx3 displacement required")
        self.fields=tuple(GridField(self.id+f":{i}",spec["origin_m"],spec["spacing_m"],data[...,i],"m") for i in range(3));self.extension=spec.get("extension","bounded");self.data=data
        self.complex=self.fields[0].cell_complex();self.source=self.complex.vertices[self.complex.cells]
        displacements=data.reshape(-1,3);self.target=(self.complex.vertices+displacements)[self.complex.cells]
        self.Js=np.array([np.linalg.solve(s[1:]-s[0],t[1:]-t[0]).T for s,t in zip(self.source,self.target)])
        self.cell_determinants=np.linalg.det(self.Js);self._certified=False
    def _inside(self,p):return self.fields[0].bounds().contains(p)
    def forward(self,p):
        p=np.asarray(p,float);inside=self._inside(p);out=p.copy()
        if not inside.all() and self.extension!="zero":invalid(self.id,"point outside bounded map")
        if inside.any():out[inside]+=np.column_stack([f.values(p[inside]) for f in self.fields])
        return out
    def jacobian(self,p):
        p=np.asarray(p,float);inside=self._inside(p);J=np.broadcast_to(np.eye(3),(len(p),3,3)).copy()
        if not inside.all() and self.extension!="zero":invalid(self.id,"point outside bounded map")
        if inside.any():J[inside]+=np.stack([f.gradient(p[inside]) for f in self.fields],axis=1)
        return J
    def inverse_candidates(self,p,tol):
        p=np.asarray(p,float);candidates=[]
        for i,(s,t,J) in enumerate(zip(self.source,self.target,self.Js)):
            if np.linalg.det(J)<=0:continue
            q=(p-t[0])@np.linalg.inv(J).T+s[0];bary=(q-s[0])@np.linalg.inv((s[1:]-s[0]).T).T
            valid=(bary>=-1e-12).all(1)&(bary.sum(1)<=1+1e-12);res=np.linalg.norm((q-s[0])@J.T+t[0]-p,axis=1);valid&=res<=tol.inverse_residual_m
            if valid.any():candidates.append(InverseCandidate(q,np.full(len(p),i,np.int64),valid,res))
        if self.extension=="zero":
            outside=~self._inside(p)
            if outside.any():candidates.append(InverseCandidate(p.copy(),np.full(len(p),-1,np.int64),outside,np.zeros(len(p))))
        return tuple(candidates)
    def image_bounds(self,box):
        d=self.data.reshape(-1,3);return AABB(tuple(np.asarray(box.minimum)+np.minimum(d.min(0),0)),tuple(np.asarray(box.maximum)+np.maximum(d.max(0),0)))
    def preimage_bounds(self,box):return self.fields[0].bounds()
    def certify(self,request):
        from .map_certification import certify_grid_map
        certificate=certify_grid_map(self,request);self._certified=certificate.global_injective;return certificate


def compile_map(spec,kernel=None):
    kind=spec["kind"]
    if kind=="affine":return AffineMap(spec["id"],spec["matrix"])
    if kind=="periodic_shear":return PeriodicShearMap(spec)
    if kind=="displacement_grid":return DisplacementGridMap(spec)
    raise ModelError(spec["id"],"unsupported",f"unknown map {kind}")
