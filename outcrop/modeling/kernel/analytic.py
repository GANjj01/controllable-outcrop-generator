"""Analytic occupancy with conservative intervals, including hidden components."""
from dataclasses import dataclass
from itertools import product
import numpy as np
from .types import AABB,Classification,IntervalEvidence,BoundaryLeaf,KernelContext
from ..errors import invalid,ModelError
from ..schema import thaw,ResourceSet


def points3(points):
    p=np.asarray(points,dtype=float)
    if p.ndim!=2 or p.shape[1]!=3 or not np.isfinite(p).all(): invalid("points","finite Nx3 points required")
    return p


def _interval_linear(n,offset,box):
    lo=np.asarray(box.minimum);hi=np.asarray(box.maximum)
    return IntervalEvidence(float(np.dot(n,np.where(n>=0,lo,hi))-offset),float(np.dot(n,np.where(n>=0,hi,lo))-offset),True,"m")


@dataclass(frozen=True)
class LinearField:
    normal: np.ndarray
    offset: float
    unit: str = "m"
    def values(self,points): return points3(points)@self.normal-self.offset
    def gradient(self,points): return np.broadcast_to(self.normal,points3(points).shape).copy()
    def interval(self,box): return _interval_linear(self.normal,self.offset,box)
    def hessian_bound(self,box): return 0.


@dataclass(frozen=True)
class QuadraticField:
    center: np.ndarray
    matrix: np.ndarray
    level: float
    unit: str = "m2"
    def values(self,points):
        q=points3(points)-self.center;return np.einsum("ni,ij,nj->n",q,self.matrix,q)-self.level
    def gradient(self,points): return 2*(points3(points)-self.center)@self.matrix
    def interval(self,box):
        # Interval arithmetic retains all cross terms; tight for diagonal Q.
        lo=np.asarray(box.minimum)-self.center;hi=np.asarray(box.maximum)-self.center
        lower=upper=-self.level
        for i in range(3):
            for j in range(3):
                if i==j:
                    a=0. if lo[i]<=0<=hi[i] else min(lo[i]**2,hi[i]**2)
                    b=max(lo[i]**2,hi[i]**2)
                else:
                    vals=[lo[i]*lo[j],lo[i]*hi[j],hi[i]*lo[j],hi[i]*hi[j]];a,b=min(vals),max(vals)
                coeff=self.matrix[i,j];lower+=min(coeff*a,coeff*b);upper+=max(coeff*a,coeff*b)
        return IntervalEvidence(float(lower),float(upper),True,self.unit)
    def hessian_bound(self,box): return 2*float(np.linalg.norm(self.matrix,2))


@dataclass(frozen=True)
class OffsetField:
    field: object
    level: float
    sign: float = 1.
    @property
    def unit(self): return self.field.unit
    def values(self,points): return self.sign*(self.field.values(points)-self.level)
    def gradient(self,points): return self.sign*self.field.gradient(points)
    def interval(self,box):
        a=self.field.interval(box);lo=self.sign*(a.lower-self.level);hi=self.sign*(a.upper-self.level)
        return IntervalEvidence(min(lo,hi),max(lo,hi),a.certified,a.unit)
    def hessian_bound(self,box): return self.field.hessian_bound(box)


@dataclass
class FieldRegion:
    id: str
    field: object
    bbox: AABB
    def _validity(self):
        from .grid import GridField
        field=self.field
        while isinstance(field,OffsetField):field=field.field
        return field.bounds() if isinstance(field,GridField) else None
    def values(self,points):
        p=points3(points);validity=self._validity()
        if validity is None:return self.field.values(p)
        mask=validity.contains(p);values=np.full(len(p),np.inf)
        if mask.any():values[mask]=self.field.values(p[mask])
        return values
    def _constant_gradient(self):
        """Gradient row of a linear field (through OffsetField sign changes), else None."""
        field,sign=self.field,1.
        while isinstance(field,OffsetField):sign*=field.sign;field=field.field
        return sign*np.asarray(field.normal,float) if isinstance(field,LinearField) else None
    def classify(self,points,tol):
        p=points3(points);validity=self._validity();active=np.ones(len(p),bool) if validity is None else validity.contains(p);v=self.values(p);g=np.zeros(len(p));safe=np.where(active,v,1.)
        row=self._constant_gradient()
        # Linear fields: every row of the gradient is the same vector, so its norm is computed once
        # with the same per-row reduction (bit-identical to the per-point evaluation).
        if active.any():g[active]=np.linalg.norm(row[None,:],axis=1)[0] if row is not None else np.linalg.norm(self.field.gradient(p[active]),axis=1)
        epsilon=tol.length_m*g+tol.scalar_by_unit.get(self.field.unit,0.)
        # Rounding is only a boundary classification bound, never a geometric shift.
        epsilon+=np.finfo(float).eps*8*np.maximum(np.abs(safe),1)
        on=active&(np.abs(v)<=epsilon)
        if validity is not None:on|=active&(v<=0)&np.any((p==validity.minimum)|(p==validity.maximum),axis=1)
        return Classification(active&(v<0)&~on,on,(~active|(v>0))&~on,np.zeros(len(p),bool))
    def bounds(self):
        validity=self._validity()
        if validity is None:return self.bbox
        lo=np.maximum(self.bbox.minimum,validity.minimum);hi=np.minimum(self.bbox.maximum,validity.maximum);return AABB(tuple(lo),tuple(np.maximum(lo,hi)))
    def interval(self,box):
        validity=self._validity()
        if validity is None:return self.field.interval(box)
        lo=np.maximum(box.minimum,validity.minimum);hi=np.minimum(box.maximum,validity.maximum)
        if np.any(lo>hi):return IntervalEvidence(1.,1.,True,self.field.unit)
        result=self.field.interval(AABB(tuple(lo),tuple(hi)))
        return IntervalEvidence(result.lower,np.inf if not validity.contains(box.corners).all() else result.upper,result.certified,result.unit)
    def boundary_leaves(self,box): return (BoundaryLeaf(self.id,self.field,None,self,"analytic"),)


@dataclass
class EmptyRegion:
    id: str = "empty"
    def classify(self,points,tol):
        n=len(points3(points));z=np.zeros(n,bool);return Classification(z,z.copy(),np.ones(n,bool),z.copy())
    def values(self,points): return np.ones(len(points3(points)))
    def bounds(self): return AABB((0.,0.,0.),(0.,0.,0.))
    def interval(self,box): return IntervalEvidence(1.,1.)
    def boundary_leaves(self,box): return ()


def box_region(oid,minimum,maximum,frame=None):
    from .csg import CSGRegion
    lo=np.asarray(minimum,float);hi=np.asarray(maximum,float)
    if lo.shape!=(3,) or hi.shape!=(3,) or np.any(lo>=hi): invalid(oid,"positive box extents required")
    a=np.eye(4) if frame is None else np.asarray(frame,float); inv=np.linalg.inv(a)
    corners=AABB(tuple(lo),tuple(hi)).corners@a[:3,:3].T+a[:3,3]
    bbox=AABB(tuple(corners.min(0)),tuple(corners.max(0))); parts=[]
    for axis in range(3):
        for sign,level in ((-1,-lo[axis]),(1,hi[axis])):
            local=np.eye(3)[axis]*sign; normal=inv[:3,:3].T@local
            offset=level-np.dot(local,inv[:3,3])
            parts.append(FieldRegion(f"{oid}:face:{axis}:{sign}",LinearField(normal,float(offset)),bbox))
    return CSGRegion(oid,"intersection",tuple(parts),bbox)


def compile_geometry(numeric,resources=None,request=None):
    from .csg import CSGRegion
    d=thaw(numeric.data); bbox=AABB(tuple(d["domain"]["min_m"]),tuple(d["domain"]["max_m"]))
    from .types import model_request
    k=KernelContext(resources=resources or ResourceSet.empty(),request=request or model_request(numeric)); specs={s["id"]:s for s in d["geometry"]}; active=set()
    def get(oid):
        if oid in k.regions: return k.regions[oid]
        if oid in k.fields: return k.fields[oid]
        if oid in k.surfaces: return k.surfaces[oid]
        if oid in k.maps: return k.maps[oid]
        if oid in active: invalid("geometry",f"construction cycle at {oid}")
        if oid not in specs: invalid("geometry",f"missing geometry {oid}")
        active.add(oid);s=specs[oid];kind=s["kind"];r=None;field=None;surface=None;mapping=None
        if kind=="empty": r=EmptyRegion(oid)
        elif kind in {"halfspace","scalar_plane","plane"}:
            if "normal" in s: n=np.asarray(s["normal"],float)
            else:
                from ..kinematics import orientation_basis,resolve_orientation
                o=resolve_orientation(s)
                n=np.array(orientation_basis(o["strike_deg"],o["dip_deg"]).normal)
            norm=np.linalg.norm(n)
            if n.shape!=(3,) or not np.isfinite(n).all() or norm==0: invalid(oid,"nonzero finite plane normal required")
            n=n/norm;offset=float(s.get("offset_m",np.dot(n,s.get("point_m",[0,0,0]))))/norm if "normal" in s else float(s.get("offset_m",np.dot(n,s.get("point_m",[0,0,0]))))
            field=LinearField(n,offset,s.get("unit","m"))
            if kind=="halfspace": r=FieldRegion(oid,field,bbox)
            elif kind=="plane":
                # Surface compilation is completed with actual finite cells in Task 5.
                surface=PlaneSurface(oid,field,bbox,s.get("extent_m"),s.get("point_m"))
                if 'frame' in s:
                    from ..deformations import MappedSurface
                    from .maps import AffineMap
                    mapping_frame=AffineMap(oid+':local-frame',s['frame']);mapping_frame.certify(k.request);surface=MappedSurface(surface,mapping_frame)
        elif kind=="box": r=box_region(oid,s["min_m"],s["max_m"],s.get("frame"))
        elif kind in {"sphere","ellipsoid"}:
            center=np.asarray(s.get("center_m",[0,0,0]),float)
            radii=np.repeat(s["radius_m"],3) if kind=="sphere" else np.asarray(s["radii_m"],float)
            if np.any(radii<=0) or not np.isfinite(radii).all(): invalid(oid,"positive finite radii required")
            a=np.asarray(s.get("frame",np.eye(4)),float);center=center@a[:3,:3].T+a[:3,3]
            inv=np.linalg.inv(a[:3,:3]);Q=inv.T@np.diag(1/radii**2)@inv
            extent=np.sqrt(np.diag(np.linalg.inv(Q)));field=QuadraticField(center,Q,1.,"1")
            r=FieldRegion(oid,field,AABB(tuple(center-extent),tuple(center+extent)))
        elif kind=="cylinder":
            p=np.asarray(s["start_m"],float);q=np.asarray(s["end_m"],float);radius=float(s["radius_m"])
            length=np.linalg.norm(q-p)
            if length==0 or radius<=0: invalid(oid,"positive cylinder length and radius required")
            n=(q-p)/length;Q=np.eye(3)-np.outer(n,n);extent=radius*np.sqrt(np.maximum(0,1-n*n))
            bounds=AABB(tuple(np.minimum(p,q)-extent),tuple(np.maximum(p,q)+extent))
            r=CSGRegion(oid,"intersection",(FieldRegion(oid+":side",QuadraticField(p,Q,radius**2),bounds),
                FieldRegion(oid+":start",LinearField(-n,-float(n@p)),bounds),FieldRegion(oid+":end",LinearField(n,float(n@q)),bounds)),bounds)
        elif kind in {"union","intersection","difference"}:
            parts=tuple(get(x) for x in s["operands"])
            if not parts or (kind=="difference" and len(parts)!=2): invalid(oid,"nonempty operands; difference needs two")
            if not all(hasattr(p,"classify") for p in parts): invalid(oid,"CSG operands must be regions")
            r=CSGRegion(oid,kind,parts)
        elif kind in {"band","levelset"}:
            base=get(s["field"])
            if isinstance(base,FieldRegion): base=base.field
            if not hasattr(base,"gradient"): invalid(oid,"scalar field required")
            if s.get("unit",base.unit)!=base.unit: invalid(oid,"field unit mismatch")
            if kind=="levelset": r=FieldRegion(oid,OffsetField(base,float(s.get("level",0))),bbox)
            else:
                lo,hi=float(s["min"]),float(s["max"])
                if lo>=hi: invalid(oid,"positive band thickness required")
                r=CSGRegion(oid,"intersection",(FieldRegion(oid+":lo",OffsetField(base,lo,-1),bbox),FieldRegion(oid+":hi",OffsetField(base,hi),bbox)),bbox)
        elif kind in {"affine","periodic_shear","displacement_grid"}:
            from .maps import compile_map
            from .types import GeometryRequest
            mapping=compile_map(s,k);mapping.certify(k.request)
        elif kind=="mapped":
            from .mapped import MappedRegion
            region=get(s["geometry"]);mapping=get(s["map"])
            r=MappedRegion(oid,region,mapping)
        elif kind=="grid":
            from .grid import GridField
            field=GridField(oid,s["origin_m"],s["spacing_m"],s["values"],s.get("unit","1"))
            if "level" in s:r=FieldRegion(oid,OffsetField(field,float(s["level"])),field.bounds())
        elif kind=="sweep":
            from .sweeps import build_sweep
            from .types import GeometryRequest
            r=build_sweep(s,k.request)
        elif kind=="mesh":
            from .mesh import MeshSurface,MeshRegion
            if "resource" in s:
                import json
                if s["resource"] not in k.resources.by_id:invalid(oid,"verified mesh resource required")
                payload=json.loads(k.resources.by_id[s["resource"]].bytes);vertices,faces=payload["vertices"],payload["faces"]
            else:vertices,faces=s["vertices"],s["faces"]
            from .types import GeometryRequest
            if s.get("closed",False):r=MeshRegion(oid,vertices,faces,k.request)
            else:surface=MeshSurface(oid,vertices,faces)
        else:
            raise ModelError(oid,"unsupported",f"geometry backend not implemented for {kind}")
        if r is not None: k.regions[oid]=r
        if field is not None: k.fields[oid]=field
        if surface is not None: k.surfaces[oid]=surface
        if mapping is not None: k.maps[oid]=mapping
        active.remove(oid);return r if r is not None else surface if surface is not None else mapping if mapping is not None else field
    for oid in sorted(specs): get(oid)
    return k


@dataclass
class PlaneSurface:
    id: str
    field: LinearField
    bbox: AABB
    extent_m: object = None
    point_m: object = None
    axes: object = None
    def bounds(self): return self.bbox
    def normal(self,points): return self.field.gradient(points)
    def cells(self,request):
        from .polytopes import ConvexCell
        from .predicates import point,sort_coplanar,plane_value,rational,dot
        from .surface_arrangement import cells_from_polygons
        n=self.field.normal;center=np.asarray(self.point_m,float) if self.point_m is not None else n*self.field.offset
        if self.extent_m is not None:
            if self.axes is None:
                axis=np.eye(3)[int(np.argmin(np.abs(n)))];u=np.cross(axis,n);u/=np.linalg.norm(u);v=np.cross(n,u)
            else:u,v=self.axes
            N=point(n);component=max(range(3),key=lambda i:abs(N[i]))
            def on_plane(vector,level):
                p=list(point(vector));p[component]=(level-sum(N[i]*p[i] for i in range(3) if i!=component))/N[component];return tuple(p)
            C=on_plane(center,rational(self.field.offset));U=on_plane(u,rational(0));V=on_plane(v,rational(0));a,b=(rational(x)/2 for x in self.extent_m)
            poly=sort_coplanar(tuple(tuple(C[k]+i*a*U[k]+j*b*V[k] for k in range(3)) for i,j in ((-1,-1),(1,-1),(1,1),(-1,1))),N)
        else:
            box=ConvexCell.box(self.bbox.minimum,self.bbox.maximum,"domain");crossings=set()
            for face in box.polygons:
                for i,p in enumerate(face):
                    q=face[(i+1)%len(face)];x=plane_value(n,self.field.offset,p);y=plane_value(n,self.field.offset,q)
                    if x==0:crossings.add(p)
                    if (x<0<y) or (y<0<x):crossings.add(tuple((x*b-y*a)/(x-y) for a,b in zip(p,q)))
            if len(crossings)<3:return cells_from_polygons([],[])
            poly=sort_coplanar(crossings,point(n))
        result=cells_from_polygons([poly],[self.id])
        if self.extent_m is not None:
            from dataclasses import replace
            from .mapped_surfaces import _rounded
            from .certified_arithmetic import float_upper,float_lower
            exact=result.exact_vertices;_,rounding=_rounded(exact);result=replace(result,position_error_m=float_upper(rounding));object.__setattr__(result,'exact_vertices',exact)
            self.bbox=AABB(tuple(float_lower(min(p[i] for p in poly)) for i in range(3)),tuple(float_upper(max(p[i] for p in poly)) for i in range(3)))
            if result.position_error_m>request.max_position_error_m:
                from ..errors import unresolved
                unresolved(self.id,'finite plane arithmetic error budget exhausted')
        return result
    def restrict(self,cells,request):
        from .mesh import MeshSurface
        source=self.cells(request);return MeshSurface(self.id,source.vertices,source.faces).restrict(cells,request)
