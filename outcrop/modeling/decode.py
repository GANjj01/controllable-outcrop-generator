"""Strict declarative contracts of the geological model; no expressions or executable callbacks."""
import json
import math
from collections.abc import Mapping
from pathlib import Path
import numpy as np
from .errors import ModelError, invalid
from .schema import ModelSpecV2, freeze, thaw

TABLES = ("lithologies","appearances","bindings","geometry","bodies","interfaces","features","events","relations","constraints","parameters","exposures","resources")
ROOT = set(TABLES) | {"schema_version","model_id","coordinates","domain","sampling","meshing","metadata"}
COMMON = {"id","source","metadata"}
GEOMETRY = {
    "empty": set(), "halfspace": {"normal","offset_m"},
    "plane": {"normal","offset_m","strike_deg","dip_deg","dip_direction_deg","point_m","extent_m","frame"},
    "box": {"min_m","max_m","frame"}, "sphere": {"center_m","radius_m"},
    "ellipsoid": {"center_m","radii_m","frame"}, "cylinder": {"start_m","end_m","radius_m"},
    "union": {"operands"}, "intersection": {"operands"}, "difference": {"operands"},
    "band": {"field","min","max","unit"}, "levelset": {"field","level","unit"},
    "mesh": {"vertices","faces","resource","closed"},
    "grid": {"origin_m","spacing_m","values","shape","unit","level"},
    "sweep": {"path_m","radius_m","section","frame"},
    "mapped": {"geometry","map"}, "affine": {"matrix"},
    "periodic_shear": {"amplitude_m","wavenumber_per_m","phase_deg","frame"},
    "displacement_grid": {"origin_m","spacing_m","values","extension"},
    "scalar_plane": {"normal","offset_m","unit"},
}
FIELDS = {
    "lithologies": {"code","rock_family","composition","properties","name"},
    "appearances": {"color","roughness","metallic","assets","name"},
    "bindings": {"selector","appearance","priority"},
    "bodies": {"geometry","material","fabrics","unit_id","overlap_policy"},
    "interfaces": {"geometry","support","hosts","feature_id","orientation"},
    "features": {"kind","hosts","geometry","support","interface","fabric","placement","orientation","extent_m","purpose","units","aperture","groove","at_stage","event"},
    "events": {"kind","after","input_stage","output_stage","targets","frame","map","geometry","body","material","fault","motion","moving_side","support","edge_policy","interfaces","fabrics","features","overlap_policy","collision_proof","cutter","remove_side"},
    "relations": {"kind","source","target","targets","at_stage","final_check","required","required_intersection","target_geometry","retain_side","seed_m","trim_region","boundary_inclusion","event","cutter","remove_side","selector"},
    "constraints": {"kind","parameters","coefficients","operator","value","relation","generative"},
    "parameters": {"value","rule","derived","dependencies","group"},
    "exposures": {"kind","geometry","bodies","include_domain_boundary","grooves","frame","support","at_stage"},
    "resources": {"path","sha256","media_type"},
}
NESTED = {
    'fault': {'geometry','orientation','map','motion_frame','motion','moving_side','support','edge_policy'},
    "material": {"kind","lithology","pieces","default","composition","properties"},
    "orientation": {"strike_deg","dip_deg","dip_direction_deg"},
    "motion": {"kind","slip_m","rake_deg","strike_slip_m","dip_slip_m","slip_vector_enu_m","throw_down_m","heave_dip_m","pure_dip_slip","opening_m","reference_point_m","redundant"},
    "placement": {"kind","count","spacing_m","offsets_m","increments_m","start_m","max_objects"},
    "fabric": {"kind","geometry","field","levels","direction","unit"},
    "selector": {"ids","bodies","lithologies","units","features","fraction","count"},
    "aperture": {"width_m","fill","side"}, "groove": {"width_m","depth_m","profile"},
    "source": {"text","start","end","source"},
}
VECTOR_FIELDS = {"center_m","radii_m","min_m","max_m","normal","point_m","start_m","end_m","origin_m","spacing_m","direction","slip_vector_enu_m","reference_point_m","seed_m"}
SCALAR_FIELDS = {"radius_m","offset_m","strike_deg","dip_deg","dip_direction_deg","amplitude_m","wavenumber_per_m","phase_deg","level","slip_m","rake_deg","strike_slip_m","dip_slip_m","throw_down_m","heave_dip_m","opening_m","width_m","depth_m","count","max_objects","fraction","code"}


def _mapping(value, field):
    if not isinstance(value,Mapping): invalid(field,"object required")


def _keys(value, allowed, field):
    _mapping(value,field)
    for k in value:
        if k not in allowed: invalid(f"{field}.{k}" if field else k,"unknown field")


def _finite(value, field):
    if isinstance(value,Mapping):
        for k,v in value.items(): _finite(v, f"{field}.{k}" if field else k)
    elif isinstance(value,(list,tuple)):
        for i,v in enumerate(value): _finite(v,f"{field}[{i}]")
    elif isinstance(value,float) and not math.isfinite(value): invalid(field,"finite value required")


def _parameter(value,field):
    if type(value) in (int,float): return
    _mapping(value,field)
    kind=value.get("kind")
    allowed={"fixed":{"kind","value","unit"},"range":{"kind","min","max","unit"},
             "distribution":{"kind","name","min","max","mean","std","mu","sigma","values","weights","unit","full_circle","max_draws"},
             "ref":{"kind","id"}}
    if kind not in allowed: invalid(field,"scalar or tagged parameter required")
    _keys(value,allowed[kind],field)
    if kind=="fixed" and type(value.get("value")) not in (int,float,str): invalid(field,"valid fixed value required")
    if kind=="range":
        for k in ("min","max"):
            if type(value.get(k)) not in (int,float): invalid(field,"numeric range endpoints required")


def _object(value,field):
    for k,v in value.items():
        p=f"{field}.{k}"
        if k in {'metadata','source'}:continue
        if k in VECTOR_FIELDS:
            if k=="start_m" and not isinstance(v,(list,tuple)) or k=='spacing_m' and '.placement.' in p: _parameter(v,p); continue
            if not isinstance(v,(list,tuple)) or len(v)!=3: invalid(p,"three components required")
            for i,a in enumerate(v): _parameter(a,f"{p}[{i}]")
        elif k in SCALAR_FIELDS: _parameter(v,p)
        elif k in NESTED and isinstance(v,Mapping):
            _keys(v,NESTED[k],p); _object(v,p)
            if k=='fabric':
                from .validation import required
                shapes={'scalar':({'kind','field','unit'},('field',)),'direction':({'kind','direction','unit'},('direction',))}
                if v.get('kind') not in shapes:invalid(p,'scalar/direction fabric required')
                allowed,mandatory=shapes[v['kind']];_keys(v,allowed,p);required(v,mandatory,p)
            if k=='placement':
                from .validation import required
                shapes={'explicit':({'kind','offsets_m','max_objects'},('offsets_m',)), 'periodic':({'kind','count','spacing_m','start_m','max_objects'},('count','spacing_m')), 'increments':({'kind','increments_m','start_m','max_objects'},('increments_m',))}
                if v.get('kind') not in shapes:invalid(p,'known joint placement kind required')
                allowed,mandatory=shapes[v['kind']];_keys(v,allowed,p);required(v,mandatory,p)
        elif k=='orientation' and isinstance(v,(list,tuple)):
            for i,o in enumerate(v):_keys(o,NESTED['orientation'],f'{p}[{i}]');_object(o,f'{p}[{i}]')
        elif k in {'offsets_m','increments_m'}:
            if isinstance(v,Mapping):_parameter(v,p)
            elif isinstance(v,(list,tuple)):
                for i,a in enumerate(v):_parameter(a,f'{p}[{i}]')
            else:invalid(p,'parameter or numeric array required')
        elif k=='units':
            if not isinstance(v,(list,tuple)):invalid(p,'unit array required')
            for i,u in enumerate(v):
                from .validation import required
                _keys(u,{'id','min','max','lithology'},f'{p}[{i}]');required(u,('id','min','max','lithology'),f'{p}[{i}]');_parameter(u['min'],f'{p}[{i}].min');_parameter(u['max'],f'{p}[{i}].max')
        elif k=="frame" or k=="matrix":
            if isinstance(v,Mapping):
                _keys(v,{"matrix"},p); v=v.get("matrix");value[k]=v
            try: a=np.asarray(v,float)
            except (TypeError,ValueError): invalid(p,"numeric affine 4x4 required")
            if a.shape!=(4,4) or not np.array_equal(a[3],[0,0,0,1]) or np.linalg.det(a[:3,:3])==0: invalid(p,"invertible affine 4x4 required")
            if any(type(x) not in (int,float) for row in v for x in row):invalid(p,'true numeric frame coordinates required')
        elif k=="extent_m":
            if not isinstance(v,(list,tuple)) or len(v)!=2: invalid(p,"two positive lengths required")
            for i,a in enumerate(v): _parameter(a,f"{p}[{i}]")
        elif k in {'closed','include_domain_boundary','final_check','required','required_intersection','generative','pure_dip_slip','full_circle'}:
            if type(v) is not bool:invalid(p,'boolean required')
        elif k=='grooves':
            if not isinstance(v,(list,tuple)):invalid(p,'groove rule array required')
            for i,rule in enumerate(v):
                from .validation import required
                _keys(rule,{'feature','width_m','depth_m','profile','combine'},f'{p}[{i}]');required(rule,('feature','width_m','depth_m'),f'{p}[{i}]');_object(rule,f'{p}[{i}]')


def validate_references(spec):
    d=thaw(spec.data); ids={}
    for table in TABLES:
        for i,obj in enumerate(d.get(table,[])):
            oid=obj.get("id")
            if not isinstance(oid,str) or not oid: invalid(f"{table}[{i}].id","nonempty stable string ID required")
            if oid in ids: invalid(f"{table}[{i}].id",f"duplicate ID {oid}")
            ids[oid]=table
    def ref(v,kind,p):
        if isinstance(v,Mapping):
            if v.get("kind")=="ref":
                _keys(v,{"kind","id"},p)
                if ids.get(v.get("id"))!="parameters":invalid(p,"unknown parameter reference")
                return
            _keys(v,{"id","stage","kind"},p); oid=v.get("id")
            if v.get('kind') is not None and v['kind']!=ids.get(oid):invalid(p,'typed reference kind must match the referenced table')
            if v.get('stage','initial')!='initial':raise ModelError(p,'unsupported','staged ObjectRef requires a versioned reference consumer; use at_stage/input_stage')
        else: oid=v
        if not isinstance(oid,str) or ids.get(oid) not in kind: invalid(p,f"unknown or wrong-type {','.join(kind)} reference {oid}")
    for table in TABLES:
        for i,obj in enumerate(d.get(table,[])):
            p=f"{table}[{i}]"
            for k in ("geometry","support","map","field"):
                if k in obj: ref(obj[k],("geometry",),f"{p}.{k}")
            for k in ("operands",):
                for j,v in enumerate(obj.get(k,[])): ref(v,("geometry",),f"{p}.{k}[{j}]")
            for k in ("hosts","bodies"):
                for j,v in enumerate(obj.get(k,[])): ref(v,("bodies",),f"{p}.{k}[{j}]")
            if table=='bodies':
                for j,v in enumerate(obj.get('fabrics',[])):ref(v,('features',),f'{p}.fabrics[{j}]')
            if "material" in obj:
                from .validation import validate_material
                m=obj['material'];validate_material(m,p+'.material')
                if 'lithology' in m:ref(m['lithology'],('lithologies',),p+'.material.lithology')
                if 'default' in m:ref(m['default'],('lithologies',),p+'.material.default')
                for j,piece in enumerate(m.get('pieces',[])):
                    ref(piece['geometry'],('geometry',),p+f'.material.pieces[{j}].geometry');ref(piece['lithology'],('lithologies',),p+f'.material.pieces[{j}].lithology')
            if table=="bindings": ref(obj.get("appearance"),("appearances",),f"{p}.appearance")
            if table=="interfaces" and "feature_id" in obj: ref(obj["feature_id"],("features",),f"{p}.feature_id")
            if table=="features" and "interface" in obj: ref(obj["interface"],("interfaces",),f"{p}.interface")
            if table=='features' and 'event' in obj:ref(obj['event'],('events',),p+'.event')
            if table=="events":
                for j,v in enumerate(obj.get("after",[])): ref(v,("events",),f"{p}.after[{j}]")
                for k in ("body",):
                    if k in obj: ref(obj[k],("bodies",),f"{p}.{k}")
            if table=="relations":
                for k in ("source","target","cutter"):
                    if k in obj and isinstance(obj[k],(str,Mapping)): ref(obj[k],("bodies","geometry","interfaces","features"),f"{p}.{k}")
    def walk(v,p):
        if isinstance(v,Mapping):
            if v.get("kind")=="ref": ref(v.get("id"),("parameters",),p)
            for k,x in v.items():
                if k not in {'metadata','source'}:walk(x,f"{p}.{k}")
        elif isinstance(v,list):
            for i,x in enumerate(v): walk(x,f"{p}[{i}]")
    walk(d,"")


def decode_model(data):
    _keys(data,ROOT,""); _finite(data,"")
    if type(data.get("schema_version")) is not int or data["schema_version"]!=2: invalid("schema_version","integer version 2 required")
    if not isinstance(data.get("model_id"),str) or not data["model_id"]: invalid("model_id","nonempty string required")
    c=data.get("coordinates",{}); _keys(c,{"frame","length_unit","angle_unit"},"coordinates")
    if c!={"frame":"ENU","length_unit":"m","angle_unit":"deg"}: invalid("coordinates","explicit ENU/m/deg required")
    domain=data.get("domain",{}); _keys(domain,{"min_m","max_m"},"domain")
    for key,value in domain.items():
        if not isinstance(value,(list,tuple)) or any(type(x) not in (int,float) for x in value):invalid('domain.'+key,'numeric coordinates required')
    from .kernel.types import AABB
    try: bounds=AABB(tuple(domain["min_m"]),tuple(domain["max_m"]))
    except (KeyError,TypeError): invalid("domain","min_m and max_m required")
    if np.any(np.asarray(bounds.maximum)<=bounds.minimum): invalid("domain","positive domain extents required")
    d=thaw(freeze(data))
    for table in TABLES:
        entries=d.setdefault(table,[])
        if not isinstance(entries,list): invalid(table,"array of objects required")
        for i,obj in enumerate(entries):
            p=f"{table}[{i}]"; _mapping(obj,p)
            if table=="geometry":
                kind=obj.get("kind")
                if kind not in GEOMETRY: invalid(p+".kind","unknown geometry kind")
                _keys(obj,COMMON|{"kind"}|GEOMETRY[kind],p)
            else: _keys(obj,COMMON|FIELDS[table],p)
            from .validation import validate_record
            validate_record(table,obj,p)
            _object(obj,p)
    for key,allowed in (("sampling",{"root_seed","sample_ids","range_policy","max_attempts","groups"}),
                        ("meshing",{"tolerance","budget","max_cell_size_m","min_feature_m","max_position_error_m","max_refinement","grid_origin_enu_m"})):
        if key in d: _keys(d[key],allowed,key)
    result=ModelSpecV2(freeze(d)); validate_references(result)
    # Initial object references are canonical immutable IDs. Parameter refs
    # and source spans retain their distinct tagged contracts.
    def normalize(v,key=None):
        if isinstance(v,dict):
            if 'id' in v and set(v)<={'id','kind','stage'} and (v.get('kind') in TABLES or 'stage' in v or v.get('kind') is None and key in {'geometry','support','map','field','source','target','cutter','hosts','targets','bodies','fabrics','after','body','interface','feature_id','lithology','default','appearance','operands'}):return v['id']
            return {k:x if k=='metadata' or k=='source' and isinstance(x,dict) and 'text' in x else normalize(x,k) for k,x in v.items()}
        if isinstance(v,list):return [normalize(x,key) for x in v]
        return v
    return ModelSpecV2(freeze(normalize(d)))


def encode_model(spec): return thaw(spec.data)


def read_model_json(path):
    from .resources import reject_links
    path=reject_links(path)
    def pairs(entries):
        result={}
        for k,v in entries:
            if k in result: invalid(k,"duplicate JSON key")
            result[k]=v
        return result
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"),object_pairs_hook=pairs,
                          parse_constant=lambda s: invalid("json",f"invalid numeric constant {s}"))
    except (OSError,UnicodeError,json.JSONDecodeError) as e: raise ModelError("json","invalid_contract",str(e)) from e
