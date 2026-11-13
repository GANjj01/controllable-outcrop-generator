"""Map composition is shared with stage events; material coordinates use inverse."""
from .kernel.maps import ComposedMap
from .kernel.mapped import MappedRegion,MappedField


class MappedSurface:
    def __init__(self,surface,mapping):self.surface=surface;self.mapping=mapping;self.id=surface.id+"@"+mapping.id
    def bounds(self):return self.mapping.image_bounds(self.surface.bounds())
    def cells(self,request):
        from dataclasses import replace
        from .kernel.mapped_surfaces import forward_surface
        from .kernel.analytic import PlaneSurface
        from .kernel.maps import PeriodicShearMap
        import numpy as np
        if isinstance(self.surface,PlaneSurface) and self.surface.extent_m is None and isinstance(self.mapping,PeriodicShearMap) and np.dot(self.surface.field.normal,self.mapping.frame.matrix[:3,2])==0:
            return replace(self.surface,bbox=self.bounds()).cells(request)
        source=self.surface.cells(replace(request,max_position_error_m=request.max_position_error_m/2));return forward_surface(source,self.mapping,request)
    def normal(self,points):
        import numpy as np
        from .kernel.types import Tolerance
        candidates=self.mapping.inverse_candidates(points,Tolerance());q=np.zeros_like(points,dtype=float);found=np.zeros(len(points),bool)
        for c in candidates:q[c.valid]=c.points[c.valid];found|=c.valid
        if not found.all():
            from .errors import unresolved
            unresolved(self.id,"mapped surface inverse missing")
        n=self.surface.normal(q);J=self.mapping.jacobian(q);normal=np.linalg.solve(J.transpose(0,2,1),n[...,None])[...,0];return normal/np.linalg.norm(normal,axis=1)[:,None]


def apply_deformation(snapshot,event,kernel):
    from dataclasses import replace
    from .errors import invalid
    mapping=kernel.maps[event["map"]];bodies=dict(snapshot.bodies);interfaces=dict(snapshot.interfaces);fabrics=dict(snapshot.fabrics);targets=set(event["targets"])
    for oid in targets:
        if oid not in bodies:invalid(event["id"],"deformation target absent at predecessor stage")
        body=bodies[oid];bodies[oid]=replace(body,region=MappedRegion(body.id+"@"+event["id"],body.region,mapping),material=body.material.mapped(mapping),version=event["id"],lineage=body.lineage+(event["id"],))
    selected_interfaces=event.get("interfaces",[])
    if selected_interfaces=="hosted":selected_interfaces=[oid for oid,s in interfaces.items() if set(s.hosts)&targets]
    for oid in selected_interfaces:
        from .kernel.maps import AffineMap
        from .errors import unresolved
        state=interfaces[oid]
        if state.terminations and not isinstance(mapping,AffineMap):unresolved(oid,'curved transport of termination edges needs a curve/topology certificate')
        terminations=tuple(replace(t,edges=mapping.forward(t.edges.reshape(-1,3)).reshape(-1,2,3)) for t in state.terminations)
        interfaces[oid]=replace(state,geometry=MappedSurface(state.geometry,mapping),support=MappedRegion(state.id+":support@"+event["id"],state.support,mapping),version=event["id"],lineage=state.lineage+(event["id"],),terminations=terminations)
    selected_fabrics=event.get("fabrics",[])
    if selected_fabrics=="hosted":selected_fabrics=[oid for oid,s in fabrics.items() if set(s.hosts)&targets]
    for oid in selected_fabrics:
        state=fabrics[oid];fabrics[oid]=replace(state,field=state.field.mapped(mapping),version=event["id"])
    return replace(snapshot,id=event.get("output_stage",event["id"]),bodies=bodies,interfaces=interfaces,fabrics=fabrics)


def apply_field_deformation(snapshot,event,kernel):
    from dataclasses import replace
    mapping=kernel.maps[event['map']];bodies=dict(snapshot.bodies);fabrics=dict(snapshot.fabrics)
    for oid in event.get('targets',[]):
        body=bodies[oid];bodies[oid]=replace(body,material=body.material.mapped(mapping),version=event['id'],lineage=body.lineage+(event['id'],))
    selected=event.get('fabrics',[])
    if selected=='hosted':selected=[oid for oid,s in fabrics.items() if set(s.hosts)&set(event.get('targets',[]))]
    for oid in selected:
        state=fabrics[oid];fabrics[oid]=replace(state,field=state.field.mapped(mapping),version=event['id'])
    return replace(snapshot,id=event.get('output_stage',event['id']),bodies=bodies,fabrics=fabrics)
