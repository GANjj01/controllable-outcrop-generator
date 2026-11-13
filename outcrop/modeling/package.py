"""Independent geology packages: explicit payload closure and immutable targets."""
from dataclasses import dataclass,asdict
from pathlib import Path
import hashlib
import os
import uuid
import numpy as np
from .schema import ResourceSet,ResourceRef,VerifiedResource
from .decode import decode_model,encode_model,read_model_json
from .resolution import resolved_to_dict,read_resolved,json_value
from .compiler import compile_model
from .exposure import extract
from .materials import bind_appearances
from .labels import build_labels,FaceLabelsV2
from .digests import model_digests,canonical_bytes,digest,ModelDigests,contact_payload
from .mesh_contracts import SurfaceMeshV2,InterfaceMeshV2,TraceMeshV2
from .kernel.surface_arrangement import SurfaceCells
from .kernel.types import GeometryRequest,Tolerance,Budget
from .resources import reject_links,confined_path
from .errors import ModelError,invalid


@dataclass(frozen=True)
class ModelProducts:
    resolved: object
    exposure: dict
    request: GeometryRequest
    extraction: object
    appearance: object
    labels: FaceLabelsV2
    digests: ModelDigests
    resources: ResourceSet


@dataclass(frozen=True)
class PackageReceipt:
    directory: Path
    status: str
    package_digest: str
    digests: ModelDigests


@dataclass(frozen=True)
class VerifiedPackage:
    directory: Path
    manifest: dict
    resources: ResourceSet
    resolved: object
    authority: tuple = ()


@dataclass(frozen=True)
class LoadedPackage:
    surface: SurfaceMeshV2
    labels: FaceLabelsV2
    spec: object
    resolved: object
    exposure: dict
    request: GeometryRequest
    manifest: dict
    resources: ResourceSet


def build_products(resolved,exposure,request,capabilities=None):
    exposure=resolved_exposure(resolved,exposure)
    resources=capabilities or ResourceSet.empty();compiled=compile_model(resolved.numeric,resources,request=request);extraction=extract(compiled,exposure,request);appearance=bind_appearances(resolved.numeric,extraction);labels=build_labels(resolved.numeric,extraction,request=request)
    return ModelProducts(resolved,dict(exposure),request,extraction,appearance,labels,model_digests(resolved.numeric,exposure,request,extraction,appearance),resources)


def resolved_exposure(resolved,exposure):
    from .schema import thaw,freeze
    value=thaw(freeze(exposure));oid=value.get('id')
    declared=next((thaw(v) for v in resolved.numeric['exposures'] if v['id']==oid),None)
    if declared is not None:
        source=next((thaw(v) for v in resolved.source_spec['exposures'] if v['id']==oid),None)
        if value!=declared and value!=source:invalid('exposure','declared exposure ID requires its source or resolved declaration')
        value=declared
    raw=encode_model(resolved.numeric);raw['exposures']=[value];decode_model(raw)
    return value


def _json(path,value):path.write_bytes(canonical_bytes(value)+b'\n')


def _scene(numeric,exposure,extraction,appearance):
    return {'schema_version':2,'kind':'geometry_scene','model_id':numeric['model_id'],'coordinates':numeric['coordinates'],'exposure':exposure,'body_ids':sorted(b['id'] for b in numeric['bodies']),'surface':{'exposure_patch_id':exposure['id'],'face_appearance_ids':appearance.face_appearance_ids,'interface_ids':tuple(i.id for i in extraction.interfaces)},'contacts':contact_payload(extraction)}


def _glb(products,path):
    import trimesh
    mesh=products.extraction.surface;vertices=mesh.vertices[:,[0,2,1]].copy();vertices[:,2]*=-1
    quantized=vertices.astype(np.float32).astype(np.float64);error=np.linalg.norm(vertices-quantized,axis=1).max()
    remaining=products.request.max_position_error_m-products.extraction.error_report.position_error_m
    triangles=quantized[mesh.faces]
    if error>remaining or np.any(np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]),axis=1)==0):raise ModelError('scene.glb','constraint_failed','float32 quantization exceeds geometry error budget or collapses triangles',{'position_error_m':float(error),'remaining_budget_m':remaining})
    scene=trimesh.Scene();mapping=[]
    ids=products.appearance.face_appearance_ids
    for oid in sorted(set(ids),key=lambda x:'' if x is None else x):
        face_ids=np.array([i for i,a in enumerate(ids) if a==oid],np.int64);v=quantized[mesh.faces[face_ids]].reshape(-1,3);f=np.arange(len(v)).reshape(-1,3)
        record=products.appearance.appearances.get(oid,{})
        color=record.get('color',[.65,.65,.65,1]);color=list(color)+([1] if len(color)==3 else [])
        material=trimesh.visual.material.PBRMaterial(name=oid or 'diagnostic',baseColorFactor=color,roughnessFactor=record.get('roughness',.8),metallicFactor=record.get('metallic',0))
        uv=products.appearance.uv_m[face_ids].reshape(-1,2);visual=trimesh.visual.TextureVisuals(uv=uv,material=material)
        name='primitive:'+str(len(mapping));scene.add_geometry(trimesh.Trimesh(v,f,visual=visual,process=False),geom_name=name,node_name=name);mapping.append({'primitive':name,'appearance':oid,'authoritative_face_indices':face_ids.tolist()})
    path.write_bytes(scene.export(file_type='glb'));return mapping,float(error)


def write_package(spec,resolved,products,target):
    if resolved.source_spec is None or encode_model(spec)!=encode_model(resolved.source_spec):invalid('source_spec','package source must match resolved original declaration')
    if canonical_bytes(resolved_to_dict(products.resolved))!=canonical_bytes(resolved_to_dict(resolved)):invalid('products.resolved','products must use the serialized resolved model')
    expected_labels=build_labels(resolved.numeric,products.extraction,request=products.request);expected_appearance=bind_appearances(resolved.numeric,products.extraction);expected_digests=model_digests(resolved.numeric,products.exposure,products.request,products.extraction,expected_appearance)
    if canonical_bytes(expected_labels)!=canonical_bytes(products.labels) or canonical_bytes(expected_appearance)!=canonical_bytes(products.appearance) or expected_digests!=products.digests:invalid('products','labels, appearance or identities do not match resolved products')
    target=reject_links(target)
    if target.exists():invalid('target','new package target required; existing target is immutable')
    target.parent.mkdir(parents=True,exist_ok=True);staging=reject_links(target.parent/('_p_'+uuid.uuid4().hex[:12]));staging.mkdir()
    try:
        _json(staging/'model_spec.json',encode_model(spec));_json(staging/'resolved_model.json',resolved_to_dict(resolved));_json(staging/'relations.json',resolved.evidence)
        mesh=products.extraction.surface
        np.savez(staging/'surface.npz',vertices=mesh.vertices.astype('<f8'),faces=mesh.faces.astype('<i8'))
        label_arrays={k:v for k,v in asdict(products.labels).items() if isinstance(v,np.ndarray)};np.savez(staging/'labels.npz',**label_arrays)
        label_tables={k:v for k,v in asdict(products.labels).items() if not isinstance(v,np.ndarray)};label_tables['surface_body_ids']=mesh.body_ids;label_tables['surface_lithology_ids']=mesh.lithology_ids;label_tables['surface_incident_regions']=mesh.incident_regions;label_tables['surface_unit_ids']=mesh.unit_ids;_json(staging/'labels.json',label_tables)
        interfaces=products.extraction.interfaces;interface_arrays={};interface_tables=[]
        for i,state in enumerate(interfaces):
            interface_arrays[f'vertices_{i}']=state.cells.vertices.astype('<f8');interface_arrays[f'faces_{i}']=state.cells.faces.astype('<i8');interface_tables.append({'id':state.id,'feature_id':state.feature_id,'hosts':state.hosts,'lineage':state.lineage,'source_ids':state.cells.source_ids,'position_error_m':state.cells.position_error_m})
        np.savez(staging/'interfaces.npz',**interface_arrays);_json(staging/'interfaces.json',interface_tables)
        traces=products.extraction.traces;np.savez(staging/'traces.npz',segments=traces.segments.astype('<f8'),points=traces.points.astype('<f8'),coplanar_vertices=traces.coplanar_cells.vertices.astype('<f8'),coplanar_faces=traces.coplanar_cells.faces.astype('<i8'));_json(staging/'traces.json',{'interface_ids':traces.interface_ids,'relation_ids':traces.relation_ids,'coplanar_source_ids':traces.coplanar_cells.source_ids})
        label_tables['surface_position_error_m']=mesh.position_error_m;_json(staging/'labels.json',label_tables)
        _json(staging/'appearance.json',products.appearance);resource_files=[]
        _json(staging/'scene.json',_scene(resolved.numeric,products.exposure,products.extraction,products.appearance))
        for i,(oid,resource) in enumerate(sorted(products.resources.by_id.items())):
            directory=staging/'resources';directory.mkdir(exist_ok=True);relative=f'resources/{i:06d}.bin';(staging/relative).write_bytes(resource.bytes);resource_files.append({'id':oid,'path':relative,'sha256':resource.ref.sha256,'media_type':resource.ref.media_type})
        primitive_map=[];quantization=0.
        if len(mesh.faces):primitive_map,quantization=_glb(products,staging/'scene.glb')
        _json(staging/'glb_primitives.json',{'coordinate_transform':'ENU(E,N,U) -> Y-up(E,U,-N)','primitives':primitive_map,'quantization_error_m':quantization})
        _json(staging/'provenance.json',{'root_seed':resolved.root_seed,'sample_id':resolved.sample_id,'versions':resolved.versions,'exposure':products.exposure,'request':asdict(products.request),'error_report':products.extraction.error_report,'resource_files':resource_files})
        names=['model_spec.json','resolved_model.json','scene.json','surface.npz','labels.npz','labels.json','interfaces.npz','interfaces.json','traces.npz','traces.json','relations.json','appearance.json','glb_primitives.json','provenance.json']+[r['path'] for r in resource_files]+(['scene.glb'] if len(mesh.faces) else [])
        payloads=[{'path':name,'bytes':(staging/name).stat().st_size,'sha256':hashlib.sha256((staging/name).read_bytes()).hexdigest()} for name in sorted(names)]
        status='complete' if len(mesh.faces) else 'empty';package_digest=digest(payloads);bounds=mesh.bounds
        manifest={'schema_version':2,'kind':'outcrop_model_package','status':status,'payloads':payloads,'package_digest':package_digest,'digests':asdict(products.digests),'bounds_enu_m':asdict(bounds) if bounds else None,'face_count':len(mesh.faces),'vertex_count':len(mesh.vertices)}
        _json(staging/'manifest.json',manifest)
        # A populated target is never replaced. Staging is retained on failure
        # for inspection; only a fully written package receives its final name.
        if target.exists():invalid('target','target appeared during package generation')
        os.rename(staging,target)
        return PackageReceipt(target,status,package_digest,products.digests)
    except ModelError:raise
    except (OSError,ValueError,TypeError) as e:raise ModelError('package','invalid_contract',str(e)) from e


def verify_package(target):
    return _verified_package_data(target)[0]


def _verified_package_data(target):
    """Validate once and retain parsed metadata for this operation's loader."""
    target=reject_links(target)
    try:manifest=read_model_json(target/'manifest.json')
    except ModelError as e:raise ModelError('manifest','resource_mismatch',str(e)) from e
    if not isinstance(manifest,dict) or manifest.get('schema_version')!=2 or manifest.get('kind')!='outcrop_model_package' or manifest.get('status') not in {'complete','empty'}:invalid('manifest','versioned v2 package required')
    from .validation import required
    required(manifest,('payloads','package_digest','digests','face_count','vertex_count','bounds_enu_m'),'manifest')
    entries=manifest.get('payloads',[]);paths=set()
    if not isinstance(entries,list):invalid('payloads','explicit payload records required')
    for record in entries:
        if not isinstance(record,dict) or set(record)!={'path','bytes','sha256'}:invalid('payloads','strict payload record required')
        name=record['path']
        if name in paths or type(record['bytes']) is not int or record['bytes']<0:invalid('payloads','unique paths and nonnegative integer sizes required')
        paths.add(name);path=confined_path(target,name)
        try:
            if path.stat().st_size!=record['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest()!=record['sha256']:raise ModelError(name,'resource_mismatch','payload hash/size mismatch')
        except OSError as e:raise ModelError(name,'resource_mismatch',str(e)) from e
    required={'model_spec.json','resolved_model.json','scene.json','surface.npz','labels.npz','labels.json','interfaces.npz','interfaces.json','traces.npz','traces.json','relations.json','appearance.json','glb_primitives.json','provenance.json'}
    if not required<=paths or ('scene.glb' in paths)!=(manifest['status']=='complete'):raise ModelError('payloads','resource_mismatch','required payload closure missing')
    if digest(entries)!=manifest.get('package_digest'):raise ModelError('manifest','resource_mismatch','package payload identity mismatch')
    provenance=read_model_json(target/'provenance.json');resources={}
    from .validation import required as require_fields
    if not isinstance(provenance,dict):invalid('provenance','object required')
    require_fields(provenance,('root_seed','sample_id','versions','exposure','request','error_report','resource_files'),'provenance')
    if not isinstance(provenance['resource_files'],list):invalid('provenance.resource_files','array required')
    _request(provenance['request'])
    for ref in provenance['resource_files']:
        if not isinstance(ref,dict) or set(ref)!={'id','path','sha256','media_type'}:invalid('resources','strict resource file record required')
        if not isinstance(ref['id'],str) or ref['id'] in resources:invalid('resources','unique resource IDs required')
        if ref['path'] not in paths:raise ModelError(ref['id'],'resource_mismatch','resource outside payload closure')
        data=confined_path(target,ref['path']).read_bytes();resources[ref['id']]=VerifiedResource(ResourceRef(**ref),data)
    capabilities=ResourceSet(resources);resolved=read_resolved(target/'resolved_model.json',capabilities=capabilities)
    if canonical_bytes(read_model_json(target/'relations.json'))!=canonical_bytes(resolved.evidence):invalid('relations','stored evidence differs from recomputed resolved relations')
    if set(resources)!=set(resolved.resource_hashes):invalid('resources','exact resolved resource closure required')
    if provenance['root_seed']!=resolved.root_seed or provenance['sample_id']!=resolved.sample_id or provenance['versions']!=dict(resolved.versions):invalid('provenance','resolved identity mismatch')
    spec=decode_model(read_model_json(target/'model_spec.json'))
    if encode_model(spec)!=encode_model(resolved.source_spec):invalid('model_spec','package source differs from resolved original source')
    verified=VerifiedPackage(target,manifest,capabilities,resolved,_authority(target,manifest,resolved,provenance))
    return verified,provenance,spec


def _request(record):
    if not isinstance(record,dict):invalid('request','object required')
    try:
        record=dict(record);record['tolerance']=Tolerance(**record['tolerance']);record['budget']=Budget(**record['budget']);return GeometryRequest(**record)
    except (TypeError,ValueError,KeyError):invalid('request','strict geometry request required')


def load_package(target):
    verified,provenance,spec=_verified_package_data(target)
    surface,labels=verified.authority
    return LoadedPackage(surface,labels,spec,verified.resolved,provenance['exposure'],_request(provenance['request']),verified.manifest,verified.resources)


def replay_package(source,target,capabilities=None):
    loaded=load_package(source);resources=capabilities or loaded.resources;products=build_products(loaded.resolved,loaded.exposure,loaded.request,resources)
    if asdict(products.digests)!=loaded.manifest['digests']:raise ModelError('digests','constraint_failed','recomputed package identities differ from recorded identities')
    return write_package(loaded.spec,loaded.resolved,products,target)


def _npz(path,keys,budget=536870912):
    import zipfile
    try:
        with zipfile.ZipFile(path) as archive:
            if sum(i.file_size for i in archive.infolist())>budget:invalid(str(path),'NPZ decompression byte budget exhausted')
        with np.load(path,allow_pickle=False) as arrays:
            if set(arrays.files)!=set(keys):invalid(str(path),'explicit NPZ array closure required')
            return {k:arrays[k] for k in keys}
    except (OSError,ValueError,zipfile.BadZipFile,KeyError) as e:raise ModelError(str(path),'invalid_contract','invalid NPZ payload') from e


def _authority(root,manifest,resolved,provenance):
    from types import SimpleNamespace
    from .validation import required
    arrays=_npz(root/'surface.npz',('vertices','faces'));vertices=arrays['vertices'];faces=arrays['faces'];_valid_mesh(vertices,faces,'surface')
    tables=read_model_json(root/'labels.json');keys=('face_ids','body_ids','lithology_ids','source_ids','interface_ids','feature_ids','interface_feature_ids','unit_ids')
    if not isinstance(tables,dict):invalid('labels','object required')
    required(tables,(*keys,'surface_body_ids','surface_lithology_ids','surface_incident_regions','surface_unit_ids','surface_position_error_m'),'labels')
    for key in keys:
        if not isinstance(tables[key],list) or any(not isinstance(v,str) and not(key=='interface_feature_ids' and v is None) for v in tables[key]):invalid('labels.'+key,'string ID array required')
    if any(len(set(tables[k]))!=len(tables[k]) for k in ('face_ids','body_ids','lithology_ids','interface_ids','unit_ids','feature_ids')):invalid('labels','duplicate category/entity IDs')
    n=len(faces)
    if any(not isinstance(tables[k],list) or len(tables[k])!=n for k in ('face_ids','source_ids','surface_body_ids','surface_lithology_ids','surface_incident_regions','surface_unit_ids')):invalid('labels','one label per face required')
    bodies=set(b['id'] for b in resolved.numeric['bodies']);lithologies=set(l['id'] for l in resolved.numeric['lithologies'])
    if any(b is not None and b not in bodies for b in tables['surface_body_ids']) or any(l is not None and l not in lithologies for l in tables['surface_lithology_ids']):invalid('labels','unknown body/lithology ID')
    if any(not isinstance(row,list) or any(b is not None and b not in bodies for b in row) for row in tables['surface_incident_regions']):invalid('labels.incident_regions','known incident body IDs required')
    if any(u is not None and (not isinstance(u,str) or u not in tables['unit_ids']) for u in tables['surface_unit_ids']):invalid('labels.unit_ids','known unit IDs required')
    if type(tables['surface_position_error_m']) not in (int,float) or not np.isfinite(tables['surface_position_error_m']) or tables['surface_position_error_m']<0:invalid('surface_position_error_m','nonnegative finite error required')
    surface=SurfaceMeshV2(vertices,faces,tuple(tables['surface_body_ids']),tuple(tables['surface_lithology_ids']),tuple(tables['source_ids']),tuple(tuple(x) for x in tables['surface_incident_regions']),tuple(tables['surface_unit_ids']),tables['surface_position_error_m'])
    interface_records=read_model_json(root/'interfaces.json')
    if not isinstance(interface_records,list) or any(not isinstance(i,dict) for i in interface_records):invalid('interfaces','explicit interface records required')
    interface_keys=tuple(k for i in range(len(interface_records)) for k in (f'vertices_{i}',f'faces_{i}'));interface_arrays=_npz(root/'interfaces.npz',interface_keys);interfaces=[];ids=set()
    for j,i in enumerate(interface_records):
        required(i,('id','feature_id','hosts','lineage','source_ids','position_error_m'),'interfaces')
        if not isinstance(i['id'],str) or i['id'] in ids or not isinstance(i['hosts'],list) or not isinstance(i['lineage'],list) or not isinstance(i['source_ids'],list):invalid('interfaces','strict unique interface identity/label arrays required')
        if any(h not in bodies for h in i['hosts']):invalid('interfaces.hosts','known body IDs required')
        ids.add(i['id']);v=interface_arrays[f'vertices_{j}'];f=interface_arrays[f'faces_{j}'];_valid_mesh(v,f,'interfaces')
        if len(i['source_ids'])!=len(f) or any(not isinstance(s,str) for s in i['source_ids']):invalid('interfaces.source_ids','one string source per face required')
        if type(i['position_error_m']) not in (int,float) or not np.isfinite(i['position_error_m']) or i['position_error_m']<0:invalid('interfaces.position_error_m','nonnegative finite error required')
        interfaces.append(InterfaceMeshV2(i['id'],SurfaceCells(v,f,tuple(i['source_ids']),i['position_error_m']),i['feature_id'],tuple(i['hosts']),tuple(i['lineage'])))
    traces_record=read_model_json(root/'traces.json')
    if not isinstance(traces_record,dict):invalid('traces','object required')
    required(traces_record,('interface_ids','relation_ids','coplanar_source_ids'),'traces')
    trace_arrays=_npz(root/'traces.npz',('segments','points','coplanar_vertices','coplanar_faces'))
    for k,tail in (('segments',(2,3)),('points',(3,))):
        a=trace_arrays[k]
        if a.dtype!=np.dtype('<f8') or a.shape[1:]!=tail or a.ndim!=len(tail)+1 or not np.isfinite(a).all():invalid('traces.'+k,'finite float64 trace array required')
    _valid_mesh(trace_arrays['coplanar_vertices'],trace_arrays['coplanar_faces'],'traces.coplanar')
    for k in ('interface_ids','relation_ids','coplanar_source_ids'):
        if not isinstance(traces_record[k],list) or any(not isinstance(s,str) for s in traces_record[k]):invalid('traces.'+k,'string ID array required')
    if not set(traces_record['interface_ids'])<=ids or len(traces_record['coplanar_source_ids'])!=len(trace_arrays['coplanar_faces']):invalid('traces','trace interface/coplanar source association mismatch')
    traces=TraceMeshV2(trace_arrays['segments'],trace_arrays['points'],SurfaceCells(trace_arrays['coplanar_vertices'],trace_arrays['coplanar_faces'],tuple(traces_record['coplanar_source_ids'])),tuple(traces_record['interface_ids']),tuple(traces_record['relation_ids']))
    scene=read_model_json(root/'scene.json')
    if not isinstance(scene,dict) or not isinstance(scene.get('contacts'),list):invalid('scene','explicit geometry scene and contact records required')
    contacts=[]
    for c in scene['contacts']:
        if not isinstance(c,dict) or set(c)!={'polygon','incident','incident_lithologies','source_ids','domain_boundary'}:invalid('scene.contacts','strict contact facet required')
        try:v=np.asarray(c['polygon'],float)
        except (TypeError,ValueError):invalid('scene.contacts','finite polygon required')
        if v.ndim!=2 or v.shape[1:]!=(3,) or len(v)<3 or not np.isfinite(v).all() or not isinstance(c['incident'],list) or len(c['incident'])!=2 or any(b not in bodies for b in c['incident']) or not isinstance(c['incident_lithologies'],list) or len(c['incident_lithologies'])!=2 or any(l not in lithologies for l in c['incident_lithologies']) or not isinstance(c['source_ids'],list) or any(not isinstance(s,str) for s in c['source_ids']) or type(c['domain_boundary']) is not bool:invalid('scene.contacts','valid body/lithology contact polygon required')
        contacts.append(SimpleNamespace(**{**c,'polygon':v}))
    extraction=SimpleNamespace(surface=surface,interfaces=tuple(interfaces),traces=traces,contacts=tuple(contacts))
    expected=build_labels(resolved.numeric,extraction,request=_request(provenance['request']))
    label_arrays=_npz(root/'labels.npz',('face_body_indices','lithology_codes','incident_offsets','incident_body_indices','face_unit_indices','face_feature_offsets','face_feature_indices'));record={k:tuple(tables[k]) for k in keys};record.update(label_arrays);labels=FaceLabelsV2(**record)
    for key,value in asdict(expected).items():
        actual=getattr(labels,key)
        if isinstance(value,np.ndarray):
            if actual.shape!=value.shape or actual.dtype!=value.dtype or not np.array_equal(actual,value):invalid('labels.'+key,'authoritative labels disagree with resolved category IDs/face incidence')
        elif actual!=value:invalid('labels.'+key,'authoritative ID table mismatch')
    if len(faces)!=manifest['face_count'] or len(vertices)!=manifest['vertex_count'] or json_value(surface.bounds)!=manifest['bounds_enu_m'] or (len(faces)==0)!=(manifest['status']=='empty'):raise ModelError('manifest','resource_mismatch','surface status/count/bounds mismatch')
    appearance=bind_appearances(resolved.numeric,extraction)
    if canonical_bytes(scene)!=canonical_bytes(_scene(resolved.numeric,provenance['exposure'],extraction,appearance)):invalid('scene','scene identity/exposure/appearance association mismatch')
    if canonical_bytes(appearance)!=canonical_bytes(read_model_json(root/'appearance.json')):invalid('appearance','appearance association mismatch')
    if asdict(model_digests(resolved.numeric,provenance['exposure'],_request(provenance['request']),extraction,appearance))!=manifest['digests']:invalid('digests','authority geometry and appearance identity mismatch')
    return surface,labels


def _valid_mesh(vertices,faces,field):
    if vertices.ndim!=2 or vertices.shape[1:]!=(3,) or vertices.dtype!=np.dtype('<f8') or faces.ndim!=2 or faces.shape[1:]!=(3,) or faces.dtype!=np.dtype('<i8') or np.any(faces<0) or np.any(faces>=len(vertices)) or not np.isfinite(vertices).all():invalid(field,'valid float64/int64 triangle mesh required')
    tri=vertices[faces]
    if np.any(np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)==0):invalid(field,'degenerate authoritative triangles forbidden')
