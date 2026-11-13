from dataclasses import dataclass
import numpy as np
from .kernel.analytic import PlaneSurface,FieldRegion,OffsetField,LinearField
from .kernel.csg import CSGRegion
from .kernel.mesh import MeshRegion,MeshSurface
from .kernel.types import GeometryRequest
from .kernel.surface_arrangement import arrange_surface_cells
from .kernel.intersections import intersect_surfaces
from .mesh_contracts import SurfaceMeshV2
from .errors import invalid,unresolved


def _positive(value,field):
    if type(value) not in (float,int) or not np.isfinite(value) or value<=0:invalid(field,'positive finite length required')
    return float(value)


def build_fracture_region(feature,rule,kernel):
    width=_positive(rule['width_m'],'aperture.width_m');surface=feature.geometry;support=feature.support
    if isinstance(surface,PlaneSurface) and surface.extent_m is None:
        parts=(FieldRegion(feature.id+':aperture-lo',OffsetField(surface.field,-width/2,-1),support.bounds()),FieldRegion(feature.id+':aperture-hi',OffsetField(surface.field,width/2),support.bounds()))
        return CSGRegion(feature.id+':aperture','intersection',(*parts,support))
    source=surface.cells(kernel.request);regions=[]
    faces=np.array([[0,2,1],[3,4,5],[0,1,4],[0,4,3],[1,2,5],[1,5,4],[2,0,3],[2,3,5]])
    for i,t in enumerate(source.vertices[source.faces]):
        n=np.cross(t[1]-t[0],t[2]-t[0]);n/=np.linalg.norm(n)
        regions.append(MeshRegion(feature.id+':aperture:'+str(i),np.r_[t-n*width/2,t+n*width/2],faces,kernel.request))
    return CSGRegion(feature.id+':aperture','intersection',(CSGRegion(feature.id+':prisms','union',tuple(regions)),support))


@dataclass(frozen=True)
class SurfaceGrooveProduct:
    surface: SurfaceMeshV2
    evidence: dict
    _trace_certificate: object | None = None


def build_surface_groove(exposure,features,rules,kernel,request):
    if not len(exposure.faces):return SurfaceGrooveProduct(exposure,{'method':'empty exposure'})
    from .kernel.predicates import point,sub,cross,dot
    exact=exposure.cells().exact_triangles;anchor=exact[0][0];N=cross(sub(exact[0][1],anchor),sub(exact[0][2],anchor))
    if any(dot(N,sub(p,anchor)) for tri in exact for p in tri) or any(dot(N,cross(sub(t[1],t[0]),sub(t[2],t[0])))<=0 for t in exact):unresolved('groove','intrinsic distance certificate requires an exact planar oriented patch')
    if exposure.position_error_m and exposure.exact_vertices is None:unresolved('groove','curved/prior surface error needs an intersection transversality certificate')
    triangles=np.asarray(exposure.vertices,float)[exposure.faces];normals=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]);normals/=np.linalg.norm(normals,axis=1)[:,None];normal=normals[0]
    if np.max(np.abs(normals-normal))>request.tolerance.angle_deg*np.pi/180 or np.max(np.abs((exposure.vertices-exposure.vertices[0])@normal))>request.tolerance.length_m:
        unresolved('groove','intrinsic distance certificate currently requires a single planar exposure patch')
    if len(rules)>1 and any(r.get('combine')!='max_depth' for r in rules):invalid('grooves','crossing grooves need explicit max_depth combination')
    from .kernel.surface_arrangement import SurfaceCells
    source=SurfaceCells(exposure.vertices,exposure.faces,tuple(str(i) for i in range(len(exposure.faces))));
    if exposure.exact_vertices is not None:object.__setattr__(source,'exact_vertices',exposure.exact_vertices)
    surface=MeshSurface('groove-exposure',source.vertices,source.faces,getattr(source,'exact_vertices',None));segments=[];profiles=[]
    for rule in rules:
        oid=rule['feature']
        if oid not in features:invalid(oid,'groove feature interface required')
        state=features[oid];width=_positive(rule['width_m'],'groove.width_m');depth=_positive(rule['depth_m'],'groove.depth_m')
        if rule.get('profile','V')!='V':invalid('groove.profile','implemented V profile required')
        from .kernel.boundary import clip_surface_to_region
        target=clip_surface_to_region(state.geometry,state.support,request)
        intersection=intersect_surfaces(surface,MeshSurface(oid,target.vertices,target.faces,getattr(target,'exact_vertices',None),target.position_error_m),request) if len(target.faces) else None
        if intersection is None or not len(intersection.segments):continue
        for exact_segment in intersection.exact_segments:
            segment=np.asarray(exact_segment,float)
            direction=segment[1]-segment[0];length=np.linalg.norm(direction)
            if length==0:continue
            tangent=np.cross(normal,direction/length);segments.extend((exact_segment,segment-tangent*width/2,segment+tangent*width/2));profiles.append((exact_segment,width,depth))
    if not profiles:return SurfaceGrooveProduct(exposure,{'method':'no finite intersection'})
    profiles=_merge_profiles(profiles)
    arranged=arrange_surface_cells(source,segments,request).cells
    from .kernel.mapped_surfaces import subdivide
    for refinement in range(request.max_refinement+1):
        vertices,offset,error,exact_vertices=_groove_certificate(arranged,profiles,normal)
        if error<=request.max_position_error_m:break
        if refinement==request.max_refinement:unresolved('groove','finite-cap interpolation error exhausted refinement')
        arranged=subdivide(arranged,request)
    indices=[int(sid) for sid in arranged.source_ids]
    result=SurfaceMeshV2(vertices,arranged.faces,tuple(exposure.body_ids[i] for i in indices),tuple(exposure.lithology_ids[i] for i in indices),tuple(exposure.source_ids[i] for i in indices),tuple(exposure.incident_regions[i] for i in indices) if exposure.incident_regions else (),tuple(exposure.unit_ids[i] for i in indices) if exposure.unit_ids else (),error,exact_vertices=exact_vertices)
    certificate=None
    if len(profiles)==1:
        from .groove_traces import _NormalGraphTraceCertificate
        from .kernel.predicates import rational
        segment,width,depth=profiles[0]
        certificate=_NormalGraphTraceCertificate(source,point(normal),tuple(map(point,segment)),rational(depth))
    return SurfaceGrooveProduct(result,{'method':'certified planar intrinsic V graph; max depth','occupancy_modified':False,'width_metric':'planar intrinsic distance','maximum_depth_m':float(offset.max()),'position_error_m':result.position_error_m},certificate)


def _merge_profiles(profiles):
    from .kernel.predicates import point,sub,cross,dot
    result=[]
    for segment,width,depth in profiles:
        a,b=map(point,segment);changed=True
        while changed:
            changed=False
            for i,(other,w,d) in enumerate(result):
                c,e=map(point,other);v=sub(b,a)
                if (w,d)!=(width,depth) or any(cross(v,sub(c,a))) or any(cross(v,sub(e,a))):continue
                axis=max(range(3),key=lambda j:abs(v[j]));lo,hi=sorted((a[axis],b[axis]));x,y=sorted((c[axis],e[axis]))
                if max(lo,x)>min(hi,y):continue
                points=sorted((a,b,c,e),key=lambda p:p[axis]);a,b=points[0],points[-1];result.pop(i);changed=True;break
        result.append(((a,b),width,depth))
    return result


def _groove_certificate(cells,profiles,normal):
    from fractions import Fraction
    from .kernel.predicates import point,sub,dot,cross,rational
    from .kernel.certified_arithmetic import sqrt_interval,float_upper
    p=getattr(cells,'exact_vertices',tuple(point(v) for v in cells.vertices));normal=point(normal);records=[];heights=[]
    for segment,width,depth in profiles:
        a,b=map(point,segment);edge=sub(b,a);length=dot(edge,edge);half=rational(width)/2;depth=rational(depth);profile=[];data=[]
        for v in p:
            alpha=dot(sub(v,a),edge)/length;clamped=max(Fraction(0),min(Fraction(1),alpha));residual=sub(sub(v,a),tuple(clamped*x for x in edge));d2=dot(residual,residual);distance=sqrt_interval(d2);lo=max(Fraction(0),depth*(1-distance.upper/half));hi=max(Fraction(0),depth*(1-distance.lower/half));profile.append((lo,hi));data.append((alpha,d2))
        records.append((a,edge,half,depth,data));heights.append(profile)
    intervals=[(max(h[i][0] for h in heights),max(h[i][1] for h in heights)) for i in range(len(p))];offsets=[(lo+hi)/2 for lo,hi in intervals];vertices=np.asarray([tuple(x-height*n for x,n in zip(v,normal)) for v,height in zip(p,offsets)],float);error=Fraction(0)
    for v,rounded,height,(lo,hi) in zip(p,vertices,offsets,intervals):
        exact=tuple(x-height*n for x,n in zip(v,normal));error=max(error,sqrt_interval(sum((rational(x)-y)**2 for x,y in zip(rounded,exact))).upper+(hi-lo)/2*sqrt_interval(dot(normal,normal)).upper)
    interpolation=Fraction(0);L=max(2*rational(depth)/rational(width) for segment,width,depth in profiles)
    for face in cells.faces:
        tri=[p[i] for i in face];diameter=sqrt_interval(max(dot(sub(a,b),sub(a,b)) for a in tri for b in tri)).upper;profile_errors=[]
        for a,edge,half,depth,data in records:
            values=[data[i] for i in face];n=cross(sub(tri[1],tri[0]),sub(tri[2],tri[0]));sides=[dot(cross(edge,sub(v,a)),n) for v in tri]
            inactive=all(sqrt_interval(d2).lower-diameter>=half for alpha,d2 in values)
            if inactive:profile_errors.append(Fraction(0));continue
            signed=[];overrun=Fraction(0);length=sqrt_interval(dot(edge,edge)).upper
            for v,(alpha,d2),side in zip(tri,values,sides):
                residual=sub(sub(v,a),tuple(alpha*x for x in edge));distance=sqrt_interval(dot(residual,residual))
                signed.append((distance.lower,distance.upper) if side>=0 else (-distance.upper,-distance.lower))
                overrun=max(overrun,max(Fraction(0),-alpha,alpha-1)*length)
            lo=min(s[0] for s in signed);hi=max(s[1] for s in signed)
            # V profile is the sum of three hinge functions. Each Jensen gap
            # is bounded by the smaller excursion on either side of its knot.
            hinge=sum(slope*max(Fraction(0),min(hi-knot,knot-lo)) for knot,slope in ((-half,depth/half),(Fraction(0),2*depth/half),(half,depth/half)))
            profile_errors.append(min(L*diameter,hinge+2*depth/half*overrun))
        switch=min(max(intervals[i][1]-heights[j][i][0] for i in face) for j in range(len(heights)))
        interpolation=max(interpolation,max(profile_errors)+switch)
    exact_vertices=tuple(tuple(x-height*n for x,n in zip(v,normal)) for v,height in zip(p,offsets)) if interpolation==0 and all(lo==hi for lo,hi in intervals) else None
    return vertices,np.asarray(offsets,float),float_upper(error+interpolation*sqrt_interval(dot(normal,normal)).upper),exact_vertices
