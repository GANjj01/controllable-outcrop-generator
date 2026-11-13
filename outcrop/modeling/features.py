from dataclasses import replace
import numpy as np
from .schema import thaw
from .fabrics import FabricField,FabricState
from .kernel.csg import CSGRegion
from .kernel.analytic import LinearField,PlaneSurface,OffsetField,FieldRegion
from .kernel.types import AABB
from .body_fields import PiecewiseMaterialField,MaterialField,compile_material
from .query import InterfaceState
from .joints import expand_joint_set
from .errors import invalid,ModelError


def compile_fabric(spec,hosts,kernel):
    regions=tuple(hosts[h].region for h in spec["hosts"])
    if not regions:invalid(spec["id"],"fabric host bodies required")
    support=CSGRegion(spec["id"]+":hosts","union",regions)
    if "support" in spec:support=CSGRegion(spec["id"]+":support","intersection",(support,kernel.regions[spec["support"]]))
    field_spec=spec["fabric"]
    if field_spec["kind"]=="scalar":field=kernel.fields[field_spec["field"]]
    elif field_spec["kind"]=="direction":
        n=np.asarray(field_spec["direction"],float)
        if not np.linalg.norm(n):invalid(spec['id'],'nonzero fabric direction required')
        n=n/np.linalg.norm(n);field=LinearField(n,0.,field_spec.get("unit","m"))
    else:invalid(spec["id"],"scalar/direction fabric required")
    if field_spec.get('unit',field.unit)!=field.unit:invalid(spec['id'],'fabric field unit mismatch')
    return FabricField(spec["id"],field,support)


def compile_interfaces(numeric,kernel):
    from .query import initial_snapshot
    return initial_snapshot(numeric,kernel).interfaces


def compile_features(numeric,snapshot,kernel,feature_ids=None):
    d=thaw(numeric.data);interfaces=dict(snapshot.interfaces);fabrics=dict(snapshot.fabrics);bodies=dict(snapshot.bodies)
    for spec in sorted(d.get("features",[]),key=lambda x:x["id"]):
        if feature_ids is not None and spec['id'] not in feature_ids:continue
        if feature_ids is None and spec.get('at_stage','initial')!='initial':continue
        kind=spec["kind"]
        if kind in {"bedding","foliation"}:
            fabric=compile_fabric(spec,bodies,kernel);fabrics[spec["id"]]=FabricState(spec["id"],fabric,tuple(spec["hosts"]))
            if spec.get("purpose","fabric_only")=="unit_boundaries":
                if not isinstance(fabric.field,LinearField):raise ModelError(spec["id"],"unsupported","unit boundary surface backend requires actual scalar surfaces")
                units=spec.get("units",[])
                if not units:invalid(spec["id"],"explicit units required")
                levels=sorted({float(u[k]) for u in units for k in ("min","max")})
                for i,level in enumerate(levels):
                    oid=f"{spec['id']}:boundary:{i:06d}";normal=fabric.field.normal;field=LinearField(normal,fabric.field.offset+level,fabric.field.unit)
                    interfaces[oid]=InterfaceState(oid,PlaneSurface(oid,field,fabric.support.bounds()),fabric.support,tuple(spec["hosts"]),spec["id"])
                for host_id in spec["hosts"]:
                    host=bodies[host_id];pieces=[]
                    for unit in units:
                        region=CSGRegion(unit["id"],"intersection",(fabric.support,FieldRegion(unit["id"]+":lo",OffsetField(fabric.field,float(unit["min"]),-1),host.region.bounds()),FieldRegion(unit["id"]+":hi",OffsetField(fabric.field,float(unit["max"])),host.region.bounds())))
                        material=compile_material({'kind':'homogeneous','lithology':unit['lithology']},{l['id']:l for l in d['lithologies']},kernel,snapshot.tolerance);material.unit_id=unit['id'];pieces.append((region,material))
                    bodies[host_id]=replace(host,material=PiecewiseMaterialField(pieces,host.material,snapshot.tolerance))
        elif kind=="joint_set":
            for joint in expand_joint_set(spec,kernel):
                state=joint.interface
                if state.hosts:state=replace(state,support=CSGRegion(state.id+":host-support","intersection",(state.support,CSGRegion(state.id+":hosts","union",tuple(bodies[h].region for h in state.hosts)))))
                interfaces[state.id]=state
        elif kind in {"joint","fault"}:
            if "interface" in spec:
                if spec["interface"] not in interfaces:invalid(spec["id"],"unknown feature interface")
                continue
            geometry=kernel.surfaces.get(spec.get("geometry"))
            if geometry is None:invalid(spec["id"],"actual feature surface required")
            hosts=tuple(spec.get("hosts",[]));support=kernel.regions[spec["support"]] if "support" in spec else CSGRegion(spec["id"]+":support","union",tuple(bodies[h].region for h in hosts))
            interfaces[spec["id"]]=InterfaceState(spec["id"],geometry,support,hosts,spec["id"])
        elif kind in {"fold","intrusion"}:pass # explicit event applies the geometry
        else:invalid(spec["id"],"unknown feature kind")
    interfaces={oid:replace(state,version=snapshot.id) if oid not in snapshot.interfaces else state for oid,state in interfaces.items()}
    from .fractures import build_fracture_region
    for spec in d['features']:
        if 'aperture' not in spec or (feature_ids is not None and spec['id'] not in feature_ids) or (feature_ids is None and spec.get('at_stage','initial')!='initial'):continue
        rule=spec['aperture']
        if rule.get('fill') is not None:raise ModelError(spec['id'],'unsupported','aperture fill requires a separate explicit intrusion body')
        if rule.get('side','both')!='both':raise ModelError(spec['id'],'unsupported','implemented aperture is symmetric on both sides')
        matches=[state for oid,state in interfaces.items() if oid==spec['id'] or oid.startswith(spec['id']+':joint:')]
        if not matches:invalid(spec['id'],'aperture requires an actual joint/fault interface')
        for state in matches:
            region=build_fracture_region(state,rule,kernel)
            for host_id in state.hosts:
                body=bodies[host_id];bodies[host_id]=replace(body,region=CSGRegion(host_id+':aperture:'+state.id,'difference',(body.region,region)),version=spec['id']+':aperture',lineage=body.lineage+(spec['id']+':aperture',))
    return replace(snapshot,bodies=bodies,interfaces=interfaces,fabrics=fabrics)
