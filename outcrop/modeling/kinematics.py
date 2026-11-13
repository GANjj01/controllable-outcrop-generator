"""ENU, north-clockwise right-hand-rule orientations and explicit signed relative motion."""
from dataclasses import dataclass
import numpy as np
from .kernel.types import Tolerance
from .errors import invalid,ModelError


@dataclass(frozen=True)
class PlaneBasis:
    strike: tuple
    dip_horizontal: tuple
    down_dip: tuple
    normal: tuple


@dataclass(frozen=True)
class CanonicalMotion:
    vector_enu_m: tuple
    strike_slip_m: float
    dip_slip_m: float
    slip_m: float
    rake_deg: float | None
    throw_down_m: float
    heave_dip_m: float
    opening_m: float
    moving_side: str
    residuals: dict


def _number(v,path):
    if isinstance(v,(bool,np.bool_)) or not isinstance(v,(int,float,np.integer,np.floating)) or not np.isfinite(v):invalid(path,"finite numeric value required")
    return float(v)


def resolve_orientation(orientation,tolerance=None):
    tol=tolerance or Tolerance();dip=_number(orientation.get("dip_deg"),"dip_deg")
    if not 0<=dip<=90:invalid("dip_deg","dip in [0,90] required")
    if "strike_deg" in orientation:strike=_number(orientation["strike_deg"],"strike_deg")%360
    elif "dip_direction_deg" in orientation:strike=(_number(orientation["dip_direction_deg"],"dip_direction_deg")-90)%360
    else:invalid("orientation","strike or dip direction required")
    if "dip_direction_deg" in orientation:
        direction=_number(orientation["dip_direction_deg"],"dip_direction_deg")%360
        residual=abs((direction-strike-90+180)%360-180)
        if residual>tol.angle_deg:raise ModelError("orientation","constraint_failed","strike/dip direction violate RHR")
    return {"strike_deg":strike,"dip_deg":dip}


def orientation_basis(strike_deg,dip_deg):
    o=resolve_orientation({"strike_deg":strike_deg,"dip_deg":dip_deg})
    def trig(deg):
        if deg%90==0:return ((0.,1.,0.,-1.)[int(deg//90)%4],(1.,0.,-1.,0.)[int(deg//90)%4])
        return np.sin(np.deg2rad(deg)),np.cos(np.deg2rad(deg))
    sa,ca=trig(o['strike_deg']);sd,cd=trig(o['dip_deg'])
    s=np.array([sa,ca,0.]);h=np.array([ca,-sa,0.]);t=cd*h-sd*np.array([0.,0.,1.]);n=np.cross(t,s)
    return PlaneBasis(tuple(s),tuple(h),tuple(t),tuple(n))


def resolve_fault_motion(orientation,motion,*,moving_side,tolerance=None):
    tol=tolerance or Tolerance();o=resolve_orientation(orientation,tol);basis=orientation_basis(o["strike_deg"],o["dip_deg"]);s,h,t,n=map(np.asarray,(basis.strike,basis.dip_horizontal,basis.down_dip,basis.normal))
    if moving_side not in {"positive","negative"}:invalid("moving_side","positive or negative side required")
    kind=motion.get("kind");opening=_number(motion.get("opening_m",0),"motion.opening_m");d=np.deg2rad(o["dip_deg"])
    if kind=="components":a=_number(motion.get("strike_slip_m"),"motion.strike_slip_m");b=_number(motion.get("dip_slip_m"),"motion.dip_slip_m")
    elif kind=="slip_rake":
        slip=_number(motion.get("slip_m"),"motion.slip_m");rake=np.deg2rad(_number(motion.get("rake_deg"),"motion.rake_deg"))
        if slip<0:invalid("motion.slip_m","nonnegative slip required")
        a=slip*np.cos(rake);b=slip*np.sin(rake)
    elif kind=="vector":
        vector=np.asarray(motion.get("slip_vector_enu_m"),float)
        if vector.shape!=(3,) or not np.isfinite(vector).all():invalid("motion.slip_vector_enu_m","finite three-vector required")
        normal_residual=float(vector@n-opening)
        if abs(normal_residual)>tol.length_m:raise ModelError("motion.slip_vector_enu_m","constraint_failed","unclaimed normal component; explicit opening required")
        a=float(vector@s);b=float(vector@t)
    elif kind in {"throw","heave"}:
        if "strike_slip_m" in motion:a=_number(motion["strike_slip_m"],"motion.strike_slip_m")
        elif motion.get("pure_dip_slip") is True:a=0.
        else:invalid("motion","throw/heave requires strike component or pure_dip_slip:true")
        factor=np.sin(d) if kind=="throw" else np.cos(d)
        if abs(factor)<=max(1/tol.max_condition,np.deg2rad(tol.angle_deg)):raise ModelError("motion","constraint_failed","singular or ill-conditioned inverse motion")
        b=_number(motion.get("throw_down_m" if kind=="throw" else "heave_dip_m"),"motion."+kind)/factor
    else:invalid("motion.kind","explicit motion mode required")
    shear=a*s+b*t
    if kind=='vector':vector=np.asarray(motion['slip_vector_enu_m'],float);shear=vector-opening*n
    else:vector=shear+opening*n
    slip=float(np.hypot(a,b));rake=None if slip==0 else float(np.degrees(np.arctan2(b,a)))
    canonical={"slip_m":slip,"strike_slip_m":float(a),"dip_slip_m":float(b),"throw_down_m":float(-shear[2]),"heave_dip_m":float(shear@h),"opening_m":opening,"slip_vector_enu_m":tuple(vector)}
    residuals={}
    claims=dict(motion.get("redundant",{}))
    primary={"components":{"strike_slip_m","dip_slip_m"},"slip_rake":{"slip_m","rake_deg"},"vector":{"slip_vector_enu_m"},"throw":{"throw_down_m","strike_slip_m"},"heave":{"heave_dip_m","strike_slip_m"}}[kind]
    claims.update({k:v for k,v in motion.items() if k in canonical and k not in primary and k!="opening_m"})
    for name,value in claims.items():
        if name=="rake_deg":
            if rake is None:continue
            residual=abs((_number(value,"motion.rake_deg")-rake+180)%360-180);limit=tol.angle_deg
        elif name in canonical:residual=float(np.max(np.abs(np.asarray(value)-canonical[name])));limit=tol.length_m
        else:invalid("motion.redundant."+name,"unknown redundant motion quantity")
        residuals[name]=residual
        if residual>limit:raise ModelError("motion."+name,"constraint_failed","inconsistent redundant motion")
    return CanonicalMotion(tuple(float(x) for x in vector),float(a),float(b),slip,rake,canonical["throw_down_m"],canonical["heave_dip_m"],opening,moving_side,residuals)
