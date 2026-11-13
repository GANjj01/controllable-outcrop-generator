"""Declared record shapes, distinct from geometric feasibility checks."""
import re
from collections.abc import Mapping
from .errors import invalid


def required(obj,keys,path):
    for key in keys:
        if key not in obj:invalid(path+'.'+key,'required field missing')


def validate_record(table,obj,path):
    required(obj,('id',),path);kind=obj.get('kind')
    if table=='geometry':
        fields={'halfspace':('normal','offset_m'),'scalar_plane':('normal','offset_m'),'box':('min_m','max_m'),'sphere':('radius_m',),'ellipsoid':('radii_m',),'cylinder':('start_m','end_m','radius_m'),'union':('operands',),'intersection':('operands',),'difference':('operands',),'band':('field','min','max'),'levelset':('field',),'grid':('origin_m','spacing_m','values'),'mapped':('geometry','map'),'affine':('matrix',),'periodic_shear':('amplitude_m','wavenumber_per_m'),'displacement_grid':('origin_m','spacing_m','values'),'sweep':('path_m',)}
        required(obj,fields.get(kind,()),path)
        if kind=='plane' and not ('normal' in obj or 'dip_deg' in obj and ('strike_deg' in obj or 'dip_direction_deg' in obj)):invalid(path,'plane normal or dip and strike/dip-direction required')
        if kind=='mesh':
            if 'resource' not in obj:required(obj,('vertices','faces'),path)
            if 'closed' in obj and type(obj['closed']) is not bool:invalid(path+'.closed','boolean required')
        if kind=='sweep' and 'radius_m' not in obj and 'section' not in obj:invalid(path,'sweep radius or polygon section required')
    elif table=='bodies':required(obj,('geometry','material'),path)
    elif table=='interfaces':required(obj,('geometry',),path)
    elif table=='lithologies':
        required(obj,('code','rock_family'),path)
        if type(obj['code']) is not int or not 1<=obj['code']<=2**32-1:invalid(path+'.code','integer category code in [1,2**32-1] required')
        if __import__('outcrop.dsgen.legacy', fromlist=['name']).name(obj['rock_family']) not in {'group_a','group_b','group_c','mixed','other'}:invalid(path+'.rock_family','known rock family required')
    elif table in {'features','events','relations','constraints','exposures'}:
        tags={'features':{'bedding','foliation','joint_set','joint','fault','fold','intrusion'},'events':{'fold','deformation','intrusion','fault','field_deformation','cuts','truncates','create_features'},'relations':{'cuts','offsets','terminates_at','contains','contacts','truncates'},'constraints':{'linear','relation'},'exposures':{'section','body_boundary','retained_region_boundary'}}
        if kind not in tags[table]:invalid(path+'.kind','known declared kind required')
        if table=='relations':required(obj,{'cuts':('source','targets'),'offsets':('event',),'terminates_at':('source','target'),'contains':('source','target'),'contacts':('source','target'),'truncates':('cutter','targets')}[kind],path)
        if table=='constraints':required(obj,('parameters','coefficients','operator','value') if kind=='linear' else ('relation',),path)
        if table=='exposures' and kind!='body_boundary':required(obj,('geometry',),path)
        if table=='features' and kind in {'bedding','foliation'}:required(obj,('hosts','fabric'),path)
        if table=='features' and kind=='joint_set':required(obj,('placement','orientation','extent_m'),path)
        if table=='features' and kind in {'fold','intrusion'}:required(obj,('event',),path)
        if table=='events':
            required(obj,{'fold':('map','targets'),'deformation':('map','targets'),'intrusion':('body','targets'),'field_deformation':('map',),'fault':('fault','targets'),'cuts':('geometry','targets'),'truncates':('cutter','targets'),'create_features':('features',)}[kind],path)
            if kind=='fault':
                nested=obj['fault'] if isinstance(obj['fault'],Mapping) else {}
                for key in ('motion','moving_side'):
                    if key not in obj and key not in nested:invalid(path+'.'+key,'explicit fault motion and moving side required')
    elif table=='resources':
        required(obj,('path','sha256','media_type'),path)
        if not isinstance(obj['path'],str) or not obj['path']:invalid(path+'.path','nonempty relative resource path required')
        if not isinstance(obj['sha256'],str) or not re.fullmatch('[0-9a-f]{64}',obj['sha256']):invalid(path+'.sha256','lowercase SHA256 required')


def validate_material(material,path):
    if not isinstance(material,Mapping):invalid(path,'material record required')
    kind=material.get('kind','homogeneous')
    if kind=='homogeneous':
        if set(material)-{'kind','lithology','composition','properties'}:invalid(path,'homogeneous material accepts lithology and local properties/composition only')
        required(material,('lithology',),path)
    elif kind=='piecewise':
        if set(material)-{'kind','pieces','default'}:invalid(path,'piecewise material requires explicit pieces and default')
        required(material,('pieces','default'),path)
        if not isinstance(material['pieces'],(list,tuple)):invalid(path+'.pieces','array required')
        for i,piece in enumerate(material['pieces']):
            p=path+f'.pieces[{i}]'
            if not isinstance(piece,Mapping) or set(piece)-{'geometry','lithology','unit_id'}:invalid(p,'geometry/lithology material piece required')
            required(piece,('geometry','lithology'),p)
            if 'unit_id' in piece and (not isinstance(piece['unit_id'],str) or not piece['unit_id'].strip()):invalid(p+'.unit_id','nonempty stable string unit ID required')
    else:invalid(path+'.kind','known material kind required')
