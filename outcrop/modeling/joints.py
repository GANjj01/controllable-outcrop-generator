from dataclasses import dataclass
import numpy as np
from .kernel.types import AABB
from .kernel.analytic import PlaneSurface,LinearField,box_region
from .query import InterfaceState
from .kinematics import orientation_basis,resolve_orientation
from .errors import invalid,unresolved


@dataclass(frozen=True)
class JointState:
    id: str
    interface: InterfaceState
    source_set_id: str
    offset_m: float
    object_version: str = "initial"


def expand_joint_set(spec,kernel):
    placement=spec["placement"];kind=placement["kind"];limit=placement.get("max_objects",10000)
    if type(limit) is not int or limit<=0:invalid(spec["id"],"positive joint object budget required")
    if kind=="explicit":offsets=np.asarray(placement["offsets_m"],float)
    elif kind=="periodic":
        count=placement["count"];spacing=float(placement["spacing_m"])
        if type(count) is not int or count<0 or spacing<=0:invalid(spec["id"],"nonnegative count and positive spacing required")
        if count>limit:unresolved(spec["id"],"joint object budget exhausted")
        offsets=placement.get("start_m",0)+np.arange(count)*spacing
    elif kind=="increments":
        inc=np.asarray(placement["increments_m"],float)
        if np.any(inc<=0):invalid(spec["id"],"strictly positive increments required")
        offsets=float(placement.get("start_m",0))+np.r_[0,np.cumsum(inc)]
    else:invalid(spec["id"],"explicit/periodic/increments placement required")
    if offsets.ndim!=1 or not np.isfinite(offsets).all() or len(set(offsets))!=len(offsets):invalid(spec["id"],"unique finite offsets required")
    if len(offsets)>limit:unresolved(spec["id"],"joint object budget exhausted")
    extent=np.asarray(spec["extent_m"],float)
    if extent.shape!=(2,) or not np.isfinite(extent).all() or np.any(extent<=0):invalid(spec["id"],"two positive extent lengths required")
    orientations=spec["orientation"];joint_list=[]
    if isinstance(orientations,list) and len(orientations)!=len(offsets):invalid(spec["id"],"one orientation per joint required")
    for i,offset in enumerate(offsets):
        orientation=orientations[i] if isinstance(orientations,list) else orientations;o=resolve_orientation(orientation);basis=orientation_basis(o["strike_deg"],o["dip_deg"])
        n=np.array(basis.normal);center=n*offset;u,v=map(np.asarray,(basis.strike,basis.down_dip));corners=np.array([center+a*extent[0]/2*u+b*extent[1]/2*v for a in (-1,1) for b in (-1,1)])
        bounds=AABB(tuple(corners.min(0)),tuple(corners.max(0)));oid=f"{spec['id']}:joint:{i:06d}"
        surface=PlaneSurface(oid,LinearField(n,float(n@center)),bounds,extent,center,(u,v))
        support=kernel.regions[spec["support"]] if "support" in spec else box_region(oid+":support",corners.min(0)-1e-6,corners.max(0)+1e-6)
        interface=InterfaceState(oid,surface,support,tuple(spec.get("hosts",[])),oid)
        joint_list.append(JointState(oid,interface,spec["id"],float(offset)))
    return tuple(joint_list)
