from dataclasses import dataclass,replace
import numpy as np
from .kernel.maps import AffineMap,ComposedMap,InverseCandidate
from .kernel.mapped import MappedField,MappedRegion
from .kernel.analytic import FieldRegion,OffsetField,PlaneSurface
from .kernel.csg import CSGRegion
from .kernel.types import GeometryRequest,AABB
from .kernel.sets import certify_empty
from .kinematics import resolve_fault_motion,resolve_orientation,orientation_basis
from .body_fields import MaterialSamples
from .events import OccupancyBranch
from .errors import ModelError,invalid,unresolved


class InverseMap:
    def __init__(self,mapping):self.mapping=mapping;self.id="inverse:"+mapping.id
    def forward(self,p):
        from .kernel.types import Tolerance
        candidates=self.mapping.inverse_candidates(p,Tolerance());out=np.zeros_like(np.asarray(p,float));found=np.zeros(len(p),bool)
        for c in candidates:out[c.valid]=c.points[c.valid];found|=c.valid
        if not found.all():unresolved(self.id,"inverse map validity missing")
        return out
    def inverse_candidates(self,p,tol):
        q=self.mapping.forward(p);return (InverseCandidate(q,np.full(len(q),-1,np.int64),np.ones(len(q),bool),np.zeros(len(q))),)
    def jacobian(self,p):return np.linalg.inv(self.mapping.jacobian(self.forward(p)))
    def image_bounds(self,box):return self.mapping.preimage_bounds(box)
    def preimage_bounds(self,box):return self.mapping.image_bounds(box)
    def certify(self,request):return self.mapping.certify(request)


@dataclass(frozen=True)
class FaultMap:
    forward_map: object
    side_classifier: object
    source_support: object | None
    edge_policy: str | None
    motion: object


def compile_fault_map(spec,kernel):
    surface=kernel.surfaces.get(spec["geometry"])
    if not isinstance(surface,PlaneSurface):raise ModelError(spec["id"],"unsupported","fault needs a side-defining base plane plus certified map")
    n=np.asarray(surface.field.normal);n=n/np.linalg.norm(n)
    if n[2]<0:n=-n
    dip=float(np.degrees(np.arccos(np.clip(n[2],-1,1))));direction=float(np.degrees(np.arctan2(n[0],n[1])))%360;strike=(direction-90)%360 if dip>0 else 0.
    orientation=spec.get("orientation",{"strike_deg":strike,"dip_deg":dip});o=resolve_orientation(orientation)
    declared=np.asarray(orientation_basis(o['strike_deg'],o['dip_deg']).normal)
    if np.linalg.norm(np.cross(n,declared))>np.sin(np.deg2rad(kernel.request.tolerance.angle_deg)):raise ModelError(spec['id'],'constraint_failed','orientation does not match actual fault plane')
    motion=resolve_fault_motion(orientation,spec["motion"],moving_side=spec["moving_side"],tolerance=kernel.request.tolerance)
    from .kernel.predicates import point,dot,rational
    from .kernel.certified_arithmetic import sqrt_interval,float_upper
    exact=list(point(motion.vector_enu_m));normal=point(surface.field.normal);input_motion=spec['motion'];correction=0.
    if input_motion['kind']!='vector' and not input_motion.get('opening_m',0):
        if input_motion['kind']=='throw':exact[2]=-rational(input_motion['throw_down_m'])
        candidates=[i for i in (0,1) if normal[i]] or [2];axis=max(candidates,key=lambda j:abs(normal[j]));before=exact[axis];exact[axis]=-sum(normal[i]*exact[i] for i in range(3) if i!=axis)/normal[axis];correction=float_upper(abs(exact[axis]-before))
        if correction>kernel.request.tolerance.length_m:unresolved(spec['id'],'named motion exact tangent realization exceeds arithmetic residual budget')
    residuals={**motion.residuals,'tangent_realization_correction_m':correction,'exact_vector_enu_m':[str(x) for x in exact]};motion=replace(motion,vector_enu_m=tuple(map(float,exact)),residuals=residuals)
    translation=[list(row) for row in point_rows(np.eye(4))]
    for i in range(3):translation[i][3]=exact[i]
    forward=AffineMap.from_exact(spec["id"]+":slip",translation);side=surface.field
    if "map" in spec:
        if spec.get("motion_frame")!="base":invalid(spec["id"],"curved motion requires explicit base frame")
        G=kernel.maps[spec["map"]];forward=ComposedMap((InverseMap(G),forward,G));side=MappedField(side,G)
    support=kernel.regions.get(spec.get("support"))
    if support is not None and spec.get("edge_policy")!="ideal_cut":invalid(spec["id"],"finite support requires implemented explicit edge policy (ideal_cut)")
    if surface.extent_m is not None and support is None:invalid(spec["id"],"finite patch needs explicit action region and edge policy")
    forward.certify(kernel.request)
    return FaultMap(forward,side,support,spec.get("edge_policy"),motion)


def point_rows(matrix):
    from .kernel.predicates import point
    return tuple(point(row) for row in matrix)


class FaultRegion(CSGRegion):
    def __init__(self,oid,reference,moving,policy=None):super().__init__(oid,"union",(reference,moving));self.policy=policy
    def classify(self,p,tol):
        reference,moving=(r.classify(p,tol) for r in self.operands)
        if np.any(reference.inside&moving.inside) and self.policy is None:raise ModelError(self.id,"constraint_failed","fault branch overlap")
        return super().classify(p,tol)


class BranchMaterialField:
    def __init__(self,reference_region,reference_field,moving_region,moving_field,tolerance,policy=None):self.reference=reference_region;self.reference_field=reference_field;self.moving=moving_region;self.moving_field=moving_field;self.tolerance=tolerance;self.policy=policy
    def sample(self,p):
        ref=self.reference.classify(p,self.tolerance);mov=self.moving.classify(p,self.tolerance);a=ref.inside|ref.boundary;b=mov.inside|mov.boundary
        if np.any(a&b) and self.policy not in {"keep_reference","keep_moving"}:raise ModelError("material","constraint_failed","fault material inverse branches ambiguous")
        if self.policy=="keep_moving":a&=~b
        else:b&=~a
        ids=np.full(len(p),None,object);composition=[{} for _ in p];properties=[{} for _ in p];units=np.full(len(p),None,object);candidates=[() for _ in p];on=np.zeros(len(p),bool)
        for mask,field in ((a,self.reference_field),(b,self.moving_field)):
            result=field.sample(np.asarray(p)[mask]);ids[mask]=result.lithology_ids
            if result.unit_ids is not None:units[mask]=result.unit_ids
            for j,i in enumerate(np.where(mask)[0]):
                if result.unit_candidates:candidates[i]=result.unit_candidates[j]
                if result.on_interface is not None:on[i]=result.on_interface[j]
            for i,c,prop in zip(np.where(mask)[0],result.composition,result.properties):composition[i]=c;properties[i]=prop
        if not (a|b).all():unresolved("material","no predecessor material branch")
        return MaterialSamples(ids,tuple(composition),tuple(properties),units,tuple(candidates),on)
    def mapped(self,mapping):
        from .body_fields import MappedMaterialField
        return MappedMaterialField(self,mapping)


def apply_fault(snapshot,event,kernel):
    specification=dict(event["fault"]) if isinstance(event["fault"],dict) else {"geometry":event["fault"]}
    specification.update({k:event[k] for k in ("motion","moving_side","support","edge_policy") if k in event});specification["id"]=event["id"]
    fault=compile_fault_map(specification,kernel);bodies=dict(snapshot.bodies);branches=list(snapshot.branches);targets=set(event["targets"]);policy=event.get("overlap_policy")
    if policy not in {None,"keep_reference","keep_moving"}:invalid(event["id"],"explicit implemented overlap policy required")
    for oid in targets:
        source=bodies[oid];sign=-1 if specification["moving_side"]=="positive" else 1
        side=FieldRegion(event["id"]+":side",OffsetField(fault.side_classifier,0.,sign),source.region.bounds())
        if fault.source_support is not None:side=CSGRegion(event["id"]+":action","intersection",(side,fault.source_support))
        moving_source=CSGRegion(oid+":moving-source@"+event["id"],"intersection",(source.region,side));reference=CSGRegion(oid+":reference@"+event["id"],"difference",(source.region,side))
        moved=MappedRegion(oid+":moved@"+event["id"],moving_source,fault.forward_map)
        if policy=="keep_reference":moved=CSGRegion(moved.id+":resolved","difference",(moved,reference))
        elif policy=="keep_moving":reference=CSGRegion(reference.id+":resolved","difference",(reference,moved))
        region=FaultRegion(oid+"@"+event["id"],reference,moved,policy)
        material=BranchMaterialField(reference,source.material,moved,source.material.mapped(fault.forward_map),snapshot.tolerance,policy)
        bodies[oid]=replace(source,region=region,material=material,version=event["id"],lineage=source.lineage+(event["id"],))
        branches.extend((OccupancyBranch(reference,snapshot.id,side,(),oid,(event["id"],)),OccupancyBranch(moving_source,snapshot.id,side,(fault.forward_map,),oid,(event["id"],))))
    request=replace(kernel.request,tolerance=snapshot.tolerance)
    # Opt-in construction proof. A zero-opening named motion is a translation tangential to the
    # fault plane, so every moved branch stays in the moving open half-space while every reference
    # branch stays in the other one; translating pairwise-disjoint predecessors by one map keeps
    # them disjoint. With all bodies targeted, no finite support and no overlap policy, branch
    # overlap and body collision are therefore impossible and the exact emptiness proofs are skipped.
    proof_mode=event.get("collision_proof","exact")
    if proof_mode not in {"exact","rigid_partition"}:invalid(event["id"],"collision_proof must be exact or rigid_partition")
    if proof_mode=="rigid_partition":
        motion=specification.get("motion",{})
        if policy is not None or fault.source_support is not None or "support" in specification:
            invalid(event["id"],"rigid_partition proof requires an unbounded fault without overlap policy")
        if targets!=set(snapshot.bodies):invalid(event["id"],"rigid_partition proof requires every body as a target")
        if motion.get("kind")=="vector" or float(motion.get("opening_m",0) or 0)!=0:
            invalid(event["id"],"rigid_partition proof requires a named zero-opening tangential motion")
    if policy is None and proof_mode=="exact":
        for oid in sorted(targets):
            reference,moving=bodies[oid].region.operands;proof=certify_empty(CSGRegion(oid+':branch-overlap','intersection',(reference,moving)),request)
            if proof.status!='pass':raise ModelError(event['id'],'constraint_failed','fault predecessor branches overlap',{'body':oid,'witness':proof.witness_points.tolist()})
    seen=set()
    for oid in (targets if proof_mode=="exact" else ()):
        for other,body in bodies.items():
            if other==oid:continue
            pair=tuple(sorted((oid,other)))
            if pair in seen:continue
            seen.add(pair)
            overlap=CSGRegion(f"{oid}/{other}:collision","intersection",(bodies[oid].region,body.region))
            proof=certify_empty(overlap,request)
            if proof.status!="pass":raise ModelError(event["id"],"constraint_failed","fault overlap/collision",{"body":oid,"other":other,"witness":proof.witness_points.tolist()})
    interfaces=dict(snapshot.interfaces);fabrics=dict(snapshot.fabrics)
    selected=event.get('interfaces',[])
    if selected=='hosted':selected=[oid for oid,s in interfaces.items() if set(s.hosts)&targets]
    target_region=CSGRegion(event['id']+':target-source','union',tuple(snapshot.bodies[oid].region for oid in sorted(targets)))
    sign=-1 if specification['moving_side']=='positive' else 1;action=FieldRegion(event['id']+':field-side',OffsetField(fault.side_classifier,0.,sign),target_region.bounds())
    action=CSGRegion(event['id']+':field-action','intersection',(action,target_region,*((fault.source_support,) if fault.source_support is not None else ())))
    for oid in selected:
        state=interfaces[oid];moving=CSGRegion(oid+':moving-field','intersection',(state.support,action));reference=CSGRegion(oid+':reference-field','difference',(state.support,action));support=CSGRegion(oid+':fault-support','union',(reference,MappedRegion(oid+':moved-support',moving,fault.forward_map)))
        geometry=BranchedSurface(state.geometry,reference,moving,fault.forward_map)
        terminations=tuple(replace(t,edges=_branch_edges(t.edges,action,fault.forward_map,request)) for t in state.terminations)
        interfaces[oid]=replace(state,geometry=geometry,support=support,version=event['id'],lineage=state.lineage+(event['id'],),terminations=terminations)
    selected=event.get('fabrics',[])
    if selected=='hosted':selected=[oid for oid,s in fabrics.items() if set(s.hosts)&targets]
    for oid in selected:
        state=fabrics[oid];fabrics[oid]=replace(state,field=BranchFabric(state.field,action,fault.forward_map,policy),version=event['id'])
    from .query import InterfaceState
    from .deformations import MappedSurface
    oid=event['id']+':interface'
    if oid in interfaces:invalid(oid,'generated fault interface ID collides with declared interface')
    surface=kernel.surfaces[specification['geometry']]
    if 'map' in specification:surface=MappedSurface(surface,kernel.maps[specification['map']])
    support=CSGRegion(oid+':hosts','union',tuple(bodies[b].region for b in sorted(targets)))
    if fault.source_support is not None:support=CSGRegion(oid+':support','intersection',(support,fault.source_support))
    interfaces[oid]=InterfaceState(oid,surface,support,tuple(sorted(targets)),event['id'],event['id'],(event['id'],))
    return replace(snapshot,id=event.get("output_stage",event["id"]),bodies=bodies,branches=tuple(branches),interfaces=interfaces,fabrics=fabrics)


def _branch_edges(edges,action,mapping,request):
    from fractions import Fraction
    from .kernel.boundary import region_planes
    from .kernel.predicates import point,plane_value,plane_sign
    from .kernel.partition import interior
    result=[]
    for edge in edges:
        a,b=map(point,edge);planes={p.source_id:p for p in region_planes(action,AABB(tuple(np.min(edge,axis=0)),tuple(np.max(edge,axis=0))))};cuts={Fraction(0),Fraction(1)}
        for p in planes.values():
            # 0 < -u/(v-u) < 1 exactly when u and v have strictly opposite signs.
            if plane_sign(p.normal,p.offset,a)*plane_sign(p.normal,p.offset,b)<0:
                u=plane_value(p.normal,p.offset,a);v=plane_value(p.normal,p.offset,b);cuts.add(-u/(v-u))
        def at(t):return tuple(x+t*(y-x) for x,y in zip(a,b))
        ordered=sorted(cuts)
        for lo,hi in zip(ordered,ordered[1:]):
            segment=np.asarray((at(lo),at(hi)),float)
            if interior(action,at((lo+hi)/2),planes):
                if not isinstance(mapping,AffineMap):unresolved('termination','curved fault transport of termination edges needs a curve certificate')
                segment=mapping.forward(segment)
            result.append(segment)
    return np.asarray(result,float).reshape(-1,2,3)


class BranchedSurface:
    def __init__(self,surface,reference,moving,mapping):self.surface=surface;self.reference=reference;self.moving=moving;self.mapping=mapping;self.id=surface.id+'@'+mapping.id
    def bounds(self):return CSGRegion(self.id+':bounds','union',(self.reference,MappedRegion(self.id+':moved',self.moving,self.mapping))).bounds()
    def cells(self,request):
        from .kernel.boundary import clip_surface_to_region
        from .kernel.surface_arrangement import SurfaceCells
        a=clip_surface_to_region(self.surface,self.reference,request);b=clip_surface_to_region(self.surface,self.moving,request)
        from .kernel.mapped_surfaces import forward_surface
        moved=forward_surface(b,self.mapping,request)
        result=SurfaceCells(np.r_[a.vertices,moved.vertices],np.r_[a.faces,moved.faces+len(a.vertices)],a.source_ids+moved.source_ids,max(a.position_error_m,moved.position_error_m))
        if hasattr(moved,'exact_vertices'):
            from .kernel.predicates import point
            object.__setattr__(result,'exact_vertices',getattr(a,'exact_vertices',tuple(point(v) for v in a.vertices))+moved.exact_vertices)
        return result
    def normal(self,p):
        from .kernel.mesh import MeshSurface
        c=self.cells(GeometryRequest());return MeshSurface(self.id,c.vertices,c.faces).normal(p)


class BranchFabric:
    def __init__(self,field,action,mapping,policy=None):self.original=field;self.action=action;self.mapping=mapping;self.policy=policy;self.id=field.id;self.support=CSGRegion(field.id+':branch-support','union',(CSGRegion(field.id+':reference','difference',(field.support,action)),MappedRegion(field.id+':moving',CSGRegion(field.id+':moving-source','intersection',(field.support,action)),mapping)))
    def sample(self,p):
        from .fabrics import FabricSamples
        p=np.asarray(p,float);tol=GeometryRequest().tolerance;valid=np.zeros(len(p),bool);normals=np.zeros_like(p);values=np.full(len(p),np.nan)
        a=self.original.sample(p);side=self.action.classify(p,tol);mask=a.valid&side.outside;valid[mask]=True;normals[mask]=a.normals[mask];values[mask]=a.scalar_coordinate[mask]
        for candidate in self.mapping.inverse_candidates(p,tol):
            q=candidate.points;b=self.original.sample(q);source=self.action.classify(q,tol);moving=candidate.valid&b.valid&(source.inside|source.boundary)
            if np.any(valid&moving) and self.policy is None:raise ModelError(self.id,'constraint_failed','fabric inverse branch ambiguity')
            if self.policy=='keep_reference':moving&=~valid
            J=self.mapping.jacobian(q[moving]);n=np.linalg.solve(J.transpose(0,2,1),b.normals[moving,...,None])[...,0];n/=np.linalg.norm(n,axis=1)[:,None]
            valid[moving]=True;normals[moving]=n;values[moving]=b.scalar_coordinate[moving]
        return FabricSamples(valid,normals,values)
    def mapped(self,mapping):return MappedFabricSampler(self,mapping)


class MappedFabricSampler:
    def __init__(self,field,mapping):self.original=field;self.mapping=mapping;self.id=field.id;self.support=MappedRegion(field.id+':mapped-support',field.support,mapping)
    def sample(self,p):
        from .fabrics import FabricSamples
        from .kernel.types import Tolerance
        p=np.asarray(p,float);q=np.zeros_like(p);found=np.zeros(len(p),bool)
        for candidate in self.mapping.inverse_candidates(p,Tolerance()):q[candidate.valid]=candidate.points[candidate.valid];found|=candidate.valid
        s=self.original.sample(q);valid=found&s.valid;J=self.mapping.jacobian(q[valid]);n=np.zeros_like(p);n[valid]=np.linalg.solve(J.transpose(0,2,1),s.normals[valid,...,None])[...,0];n[valid]/=np.linalg.norm(n[valid],axis=1)[:,None]
        return FabricSamples(valid,n,s.scalar_coordinate)
    def mapped(self,mapping):return MappedFabricSampler(self,mapping)


def inverse_branches(branches,points):
    from .kernel.types import Tolerance
    results=[]
    for index,branch in enumerate(branches):
        mapping=ComposedMap(branch.map_chain)
        for candidate in mapping.inverse_candidates(points,Tolerance()):
            source=branch.source_region.classify(candidate.points,Tolerance());valid=candidate.valid&(source.inside|source.boundary)
            results.append((index,candidate.points,valid,candidate.residual_m))
    return tuple(results)
