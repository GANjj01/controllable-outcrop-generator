"""Development-only centerline and source-3D coverage metrics.

Predictions are independent RGB results. Geometry is read only at evaluation.
No joining of visibility runs, GT-driven prediction correction, or hidden pruning.
"""
import copy
from collections import Counter
import numpy as np
from PIL import Image,ImageDraw
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

def select_target_vectors(vectors,target_classes=('joint',)):
    """Select task-defined GT classes only; predictions and original labels stay intact."""
    classes=set(target_classes)
    instances=[row for row in vectors['instances'] if row['class_name'] in classes]
    runs=[row for row in vectors['runs'] if row['class_name'] in classes]
    ids={row['interface_id'] for row in instances}
    if any(row['interface_id'] not in ids for row in runs):raise ValueError('Target run lacks a declared source identity')
    record=dict(target_classes=sorted(classes),target_source_instances=[row['interface_id'] for row in instances],target_visible_runs=len(runs),non_target_source_instances=[row['interface_id'] for row in vectors['instances'] if row['class_name'] not in classes],non_target_runs_by_class=dict(Counter(row['class_name'] for row in vectors['runs'] if row['class_name'] not in classes)),non_target_policy='Excluded from GT only, not excluded from the image/evaluation area; predictions on non-target faults/contacts count as false positives unless also within a joint matching tolerance.')
    return runs,instances,record

def transform_runs(runs,scale,offset=(0.,0.)):
    """Isotropic resize then crop, in pixel-centre coordinates; world data untouched."""
    result=copy.deepcopy(runs)
    for row in result:
        row['points_px']=((np.asarray(row['points_px'])+.5)*scale-.5-np.asarray(offset)).tolist()
    return result

def raster_vectors(runs,shape):
    """One-pixel union raster; crossing identity remains in source runs, not this mask."""
    image=Image.new('1',(shape[1],shape[0]));draw=ImageDraw.Draw(image)
    for row in runs:
        points=[tuple(p) for p in np.rint(row['points_px']).astype(int)]
        if len(points)>1:draw.line(points,fill=1,width=1)
        elif points:draw.point(points[0],fill=1)
    return np.asarray(image,dtype=bool).copy()

def skeletonize(mask):
    """Zhang-Suen two-subiteration thinning, preserving connectivity and endpoints."""
    a=np.asarray(mask,bool).copy()
    while True:
        deleted=0
        for phase in (0,1):
            p=np.pad(a,1);n=[p[:-2,1:-1],p[:-2,2:],p[1:-1,2:],p[2:,2:],p[2:,1:-1],p[2:,:-2],p[1:-1,:-2],p[:-2,:-2]]
            count=np.add.reduce(n,dtype=np.uint8)
            transitions=np.add.reduce([~n[i]&n[(i+1)%8] for i in range(8)],dtype=np.uint8)
            if phase==0:triples=(n[0]&n[2]&n[4])|(n[2]&n[4]&n[6])
            else:triples=(n[0]&n[2]&n[6])|(n[0]&n[4]&n[6])
            remove=a&(count>=2)&(count<=6)&(transitions==1)&~triples
            deleted+=int(remove.sum());a[remove]=False
        if not deleted:return a

def graph_node_weights(mask):
    """Half incident Euclidean edge length per vertex; no redundant corner diagonals."""
    p=np.asarray(mask,bool);h,w=p.shape;weights=np.zeros(p.shape,float)
    for dy,dx in ((0,1),(1,0),(1,1),(1,-1)):
        y0,y1=max(0,-dy),min(h,h-dy);x0,x1=max(0,-dx),min(w,w-dx)
        a=p[y0:y1,x0:x1];b=p[y0+dy:y1+dy,x0+dx:x1+dx];edge=a&b
        if dy and dx:
            edge &= ~(p[y0:y1,x0+dx:x1+dx]|p[y0+dy:y1+dy,x0:x1])
        amount=edge*(np.hypot(dy,dx)/2)
        weights[y0:y1,x0:x1]+=amount;weights[y0+dy:y1+dy,x0+dx:x1+dx]+=amount
    return weights

def graph_length(mask):return float(graph_node_weights(mask).sum())

def clip_segment(a,b,width,height):
    """Parametric clipping against pixel-cell extent, not only centre coordinates."""
    lo,hi=0.,1.;d=b-a
    for k,lower,upper in ((0,-.5,width-.5),(1,-.5,height-.5)):
        if abs(d[k])<1e-15:
            if a[k]<lower or a[k]>upper:return None
        else:
            ends=sorted(((lower-a[k])/d[k],(upper-a[k])/d[k]))
            lo=max(lo,ends[0]);hi=min(hi,ends[1])
            if hi<=lo:return None
    return lo,hi

def evaluate(prediction_skeleton,runs,region,tolerance_px,reference_basis,instances=None):
    region=np.asarray(region,bool);original=np.asarray(prediction_skeleton,bool)
    if original.shape!=region.shape or tolerance_px<0:raise ValueError('Shape/tolerance mismatch')
    if not region.any():raise ValueError('No visible rock pixels: report failed/unevaluable condition')
    h,w=region.shape;pred=original&region;gt=raster_vectors(runs,region.shape)&region
    npred,ngt=int(pred.sum()),int(gt.sum())
    dtgt=ndi.distance_transform_edt(~gt) if ngt else np.full(region.shape,np.inf)
    dtpred=ndi.distance_transform_edt(~pred) if npred else np.full(region.shape,np.inf)
    pmatch=pred&(dtgt<=tolerance_px);gmatch=gt&(dtpred<=tolerance_px)
    precision=float(pmatch.sum()/npred) if npred and ngt else (0. if ngt else None)
    recall=float(gmatch.sum()/ngt) if ngt else None
    f1=(2*precision*recall/(precision+recall) if precision+recall else 0.) if ngt else None
    nodes=graph_node_weights(pred);unmatched=float(nodes[pred&~pmatch].sum())
    y,x=np.nonzero(pred);tree=cKDTree(np.c_[x,y]) if npred else None
    basis=np.asarray(reference_basis,float)
    if basis.shape!=(2,3):raise ValueError('A fixed source-world to reference-plane 2x3 basis is required')
    hist=np.zeros(6);covered=np.zeros(6);undefined=0.;projected=0.
    bytrace={row['interface_id']:dict(instance_id=row['instance_id'],total_m=0.,matched_m=0.,projected_length_px=0.,visible_run_count=0) for row in (instances or [])}
    for row in runs:
        key=row['interface_id'];entry=bytrace.setdefault(key,dict(instance_id=row['instance_id'],total_m=0.,matched_m=0.,projected_length_px=0.,visible_run_count=0))
        entry['visible_run_count']+=1
        uv=np.asarray(row['points_px'],float);world=np.asarray(row['world_enu_m'],float)
        if uv.shape!=(len(world),2) or world.shape[1:]!=(3,):raise ValueError('Paired source UV/world run samples required')
        for a,b,wa,wb in zip(uv[:-1],uv[1:],world[:-1],world[1:]):
            clipped=clip_segment(a,b,w,h)
            if clipped is None:continue
            lo,hi=clipped;d=b-a;dw=wb-wa
            count=max(1,int(np.ceil(np.linalg.norm(d)*(hi-lo)/.25)))
            t=lo+(np.arange(count)+.5)/count*(hi-lo);points=a+t[:,None]*d
            iy=np.floor(points[:,1]+.5).astype(int).clip(0,h-1);ix=np.floor(points[:,0]+.5).astype(int).clip(0,w-1)
            use=region[iy,ix];points=points[use]
            weight=float(np.linalg.norm(dw)*(hi-lo)/count);pxweight=float(np.linalg.norm(d)*(hi-lo)/count)
            total=len(points)*weight;px=len(points)*pxweight
            match=tree.query(points)[0]<=tolerance_px if tree is not None else np.zeros(len(points),bool)
            hit=int(match.sum())*weight
            entry['total_m']+=total;entry['matched_m']+=hit;entry['projected_length_px']+=px;projected+=px
            direction=basis@dw
            if np.linalg.norm(direction)<1e-12:undefined+=total
            else:
                angle=float(np.degrees(np.arctan2(direction[1],direction[0]))%180)
                k=min(5,int(angle//30));hist[k]+=total;covered[k]+=hit
    for entry in bytrace.values():entry['recall']=entry['matched_m']/entry['total_m'] if entry['total_m'] else None
    total=sum(e['total_m'] for e in bytrace.values());matched=sum(e['matched_m'] for e in bytrace.values())
    hn=hist/hist.sum() if hist.sum() else None;cn=covered/covered.sum() if covered.sum() else None
    fullweight=graph_node_weights(original);outside=original&~region
    return {
      'centerline':{'precision':precision,'recall':recall,'f1':f1,'predicted_pixels':npred,'gt_union_pixels':ngt,'matched_prediction_pixels':int(pmatch.sum()),'matched_gt_pixels':int(gmatch.sum()),'symmetric_tolerance_px':float(tolerance_px)},
      'length_3d':{'total_m':total,'matched_m':matched,'recall':matched/total if total else None,'scope':'geometrically visible source runs, midpoint arc length quadrature <=0.25 current px'},
      'by_trace':bytrace,
      'false_positive':{'unmatched_skeleton_length_px':unmatched,'per_million_visible_rock_pixels':unmatched/int(region.sum())*1e6,'visible_rock_pixels':int(region.sum()),'unmatched_isolated_pixels':int((pred&~pmatch&(nodes==0)).sum()),'negative_case':len(instances)==0 if instances is not None else total==0,'visible_target_absent':total==0,'source_target_instance_count':len(instances) if instances is not None else None},
      'prediction':{'skeleton_graph_length_px':float(nodes.sum()),'isolated_pixels':int((pred&(nodes==0)).sum()),'outside_evaluation_region_pixels':int(outside.sum()),'outside_region_incident_half_edge_length_px':float(fullweight[outside].sum())},
      'gt_projected_visible_length_px':projected,'projected_total_length_ratio':float(nodes.sum()/projected) if projected else None,
      'direction':{'definition':'source 3D trace projected to fixed main-camera plane; axial 180 degrees; not geological strike/dip','bin_edges_degrees':list(range(0,181,30)),'true_length_m_by_bin':hist.tolist(),'covered_length_m_by_bin':covered.tolist(),'recall_by_bin':[float(b/a) if a else None for a,b in zip(hist,covered)],'true_normalized_histogram':hn.tolist() if hn is not None else None,'covered_normalized_histogram':cn.tolist() if cn is not None else None,'covered_vs_true_total_variation':float(np.abs(hn-cn).sum()/2) if hn is not None and cn is not None else None,'undefined_projection_length_m':undefined},
      'predicted_3d_total_length_and_direction':'NOT_COMPUTED: requires surface backprojection; no unmatched predictions silently discarded',
    }
