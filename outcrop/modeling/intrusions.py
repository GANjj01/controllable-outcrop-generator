from dataclasses import replace
from .kernel.csg import CSGRegion
from .query import InterfaceState
from .errors import invalid,ModelError


class RegionBoundarySurface:
    def __init__(self,region):self.region=region;self.id=region.id+":boundary"
    def bounds(self):return self.region.bounds()
    def cells(self,request):
        from .meshing import extract_region_boundary
        return extract_region_boundary(self.region,request)
    def normal(self,points):
        import numpy as np
        leaves=self.region.boundary_leaves(self.region.bounds());fields=[l.field for l in leaves if l.field is not None]
        if not fields:
            return leaves[0].surface.normal(points)
        gradients=[]
        for p in points:
            field=min(fields,key=lambda f:abs(f.values(np.asarray([p]))[0]));g=field.gradient(np.asarray([p]))[0];gradients.append(g/np.linalg.norm(g))
        return np.asarray(gradients)


def apply_intrusion(snapshot,event,kernel):
    oid=event["body"]
    if oid in snapshot.bodies:invalid(event["id"],"intrusive body already exists")
    new=kernel.body_templates[oid];targets=set(event.get("targets",[]));bodies=dict(snapshot.bodies);interfaces=dict(snapshot.interfaces)
    if not targets.issubset(bodies):invalid(event["id"],"intrusion target absent at predecessor stage")
    for target in targets:
        body=bodies[target];region=CSGRegion(f"{body.id}@{event['id']}","difference",(body.region,new.region))
        bodies[target]=replace(body,region=region,version=event["id"],lineage=body.lineage+(event["id"],))
    bodies[oid]=replace(new,version=event["id"],lineage=(event["id"],))
    interface_id=event["id"]+":contact";interfaces[interface_id]=InterfaceState(interface_id,RegionBoundarySurface(new.region),new.region,tuple(sorted(targets|{oid})),oid,event["id"],(event["id"],))
    return replace(snapshot,id=event.get("output_stage",event["id"]),bodies=bodies,interfaces=interfaces)
