"""Certified forward surface proxies, separate from exact PL relations."""
from fractions import Fraction
from copy import copy
import numpy as np
from .maps import AffineMap,PeriodicShearMap
from .surface_arrangement import SurfaceCells
from .predicates import point,dot,rational
from .certified_arithmetic import inverse3,sin_interval,sqrt_interval,float_upper
from ..errors import unresolved


def subdivide(cells,request):
    if len(cells.faces)*4>request.budget.max_faces or (len(cells.vertices)+len(cells.faces)*3)*128>request.budget.max_bytes:unresolved('surface','surface refinement face/memory budget exhausted')
    vertices=list(getattr(cells,'exact_vertices',tuple(point(v) for v in cells.vertices)));lookup={};faces=[];sources=[]
    for (a,b,c),sid in zip(cells.faces,cells.source_ids):
        mids=[]
        for i,j in ((a,b),(b,c),(c,a)):
            key=tuple(sorted((int(i),int(j))))
            if key not in lookup:lookup[key]=len(vertices);vertices.append(tuple((x+y)/2 for x,y in zip(vertices[i],vertices[j])))
            mids.append(lookup[key])
        ab,bc,ca=mids;faces.extend(((a,ab,ca),(ab,b,bc),(ca,bc,c),(ab,bc,ca)));sources.extend((sid,)*4)
    result=SurfaceCells(np.asarray(vertices,float),np.asarray(faces,np.int64),tuple(sources),cells.position_error_m);object.__setattr__(result,'exact_vertices',tuple(vertices));return result


def _rounded(vertices):
    values=np.asarray(vertices,float).reshape(-1,3);error=Fraction(0)
    for exact,rounded in zip(vertices,values):
        error=max(error,sqrt_interval(sum((rational(x)-y)**2 for x,y in zip(rounded,exact))).upper)
    return values,error


def forward_surface(source,mapping,request):
    from .proxies import _map_chain
    result=source
    for m,inverse in _map_chain(mapping):
        if isinstance(m,AffineMap):
            M=getattr(m,'exact_matrix',tuple(point(row) for row in m.matrix));linear=tuple(row[:3] for row in M[:3]);translation=tuple(row[3] for row in M[:3])
            if inverse:
                linear=inverse3(linear);translation=tuple(-dot(row,translation) for row in linear)
            result=_forward_affine(result,linear,translation,m.id,request);continue
        if inverse:
            if isinstance(m,PeriodicShearMap):m=copy(m);m.A=-m.A
            else:unresolved('surface','inverse PL map surface needs explicit source tetra segmentation')
        result=_forward_one(result,m,request)
    return result


def _forward_one(source,mapping,request):
    if isinstance(mapping,AffineMap):
        M=getattr(mapping,'exact_matrix',tuple(point(row) for row in mapping.matrix));return _forward_affine(source,tuple(row[:3] for row in M[:3]),tuple(row[3] for row in M[:3]),mapping.id,request)
    if not isinstance(mapping,PeriodicShearMap):unresolved(mapping.id,'surface map requires certified PL segmentation or an implemented analytic forward bound')
    M=mapping.frame.matrix;inv=inverse3(M[:3,:3]);u=inv[0];v=point(M[:3,2]);translation=point(M[:3,3]);A=rational(mapping.A);k=rational(mapping.k);phase=rational(mapping.phase);d=-dot(u,translation)
    # L1 operator bounds enclose the Euclidean Lipschitz constant.
    normM=max(sum(abs(rational(x)) for x in row) for row in M[:3,:3]);normI=max(sum(abs(x) for x in row) for row in inv);L=3*normM*normI*(1+abs(A*k));prior=L*rational(source.position_error_m)
    scale=abs(A)*sqrt_interval(dot(v,v)).upper*k*k;cells=source
    for refinement in range(request.max_refinement+1):
        vertices=tuple(getattr(cells,'exact_vertices',tuple(point(p) for p in cells.vertices)));s=[dot(u,p) for p in vertices];mapped=[];oracle=Fraction(0)
        for p,a in zip(vertices,s):
            sine=sin_interval(k*(a+d)+phase);middle=(sine.lower+sine.upper)/2;mapped.append(tuple(x+A*y*middle for x,y in zip(p,v)));oracle=max(oracle,abs(A)*sqrt_interval(dot(v,v)).upper*(sine.upper-sine.lower)/2)
        values,rounding=_rounded(mapped);width=max((max(s[i] for i in face)-min(s[i] for i in face))**2 for face in cells.faces) if len(cells.faces) else Fraction(0)
        bound=float_upper(prior+scale*width/8+oracle+rounding)
        if bound<=request.max_position_error_m:return SurfaceCells(values,cells.faces,cells.source_ids,bound)
        if refinement==request.max_refinement:unresolved(mapping.id,'curved surface position certificate exhausted refinement')
        cells=subdivide(cells,request)


def _forward_affine(source,linear,translation,oid,request):
    vertices=tuple(tuple(dot(row,p)+t for row,t in zip(linear,translation)) for p in getattr(source,'exact_vertices',tuple(point(v) for v in source.vertices)))
    values,error=_rounded(vertices)
    norm1=max(sum(abs(linear[i][j]) for i in range(3)) for j in range(3));norminf=max(sum(abs(x) for x in row) for row in linear)
    bound=float_upper(error+sqrt_interval(norm1*norminf).upper*rational(source.position_error_m))
    if bound>request.max_position_error_m:unresolved(oid,'surface arithmetic error budget exhausted')
    result=SurfaceCells(values,source.faces,source.source_ids,bound)
    if not source.position_error_m or hasattr(source,'exact_vertices'):object.__setattr__(result,'exact_vertices',vertices)
    return result
