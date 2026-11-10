"""Read-only geometric evaluation of recorded sentry trajectories."""
import math


def _geometry(passage):
    if not isinstance(passage,dict):return None
    try:
        center=tuple(float(v) for v in passage['center'])
        axis=tuple(float(v) for v in passage['axis'])
        half=float(passage['half']);width=float(passage['width'])
        if len(center)!=2 or len(axis)!=2 or not all(math.isfinite(v) for v in (*center,*axis,half,width)):
            return None
        norm=math.hypot(*axis)
        if norm<1e-9 or half<=0 or width<=0:return None
        axis=tuple(v/norm for v in axis)
        # Passage planning can reverse its axis. Give one physical roof one
        # stable coordinate frame, while permitting both travel directions.
        if axis[0]<-1e-9 or abs(axis[0])<=1e-9 and axis[1]<0:axis=tuple(-v for v in axis)
        return dict(id=str(passage.get('id','unnamed_roof')),center=center,axis=axis,half=half,width=width)
    except (KeyError,TypeError,ValueError,OverflowError):return None


def _sample(row):
    try:
        position=tuple(float(v) for v in row['position'][:2])
        time=float(row['time']);fold=float(row.get('fold',row.get('fold_progress')))
        if len(position)!=2 or not all(math.isfinite(v) for v in (*position,time,fold)):return None
        return dict(time=time,position=position,fold=fold)
    except (KeyError,TypeError,ValueError,OverflowError):return None


def _slab_interval(first,last,half):
    delta=last-first
    if abs(delta)<1e-12:return (0.,1.) if abs(first)<=half else None
    low,high=sorted(((-half-first)/delta,(half-first)/delta))
    low=max(0.,low);high=min(1.,high)
    return (low,high) if low<=high else None


def detect_tunnel_passages(trace, *, fold_threshold=.8, clearance_m=.05,
                           max_gap_s=.25, lateral_clearance_m=0., max_speed_mps=20.):
    """Require an observed, folded traversal through a recorded roof footprint.

    The tracked center must go from beyond one roof end to beyond the other,
    remain laterally inside the opening, and have at least one observed sample
    under the roof. Fold must meet the threshold throughout every interpolated
    roof intersection. Missing/late samples and implausible jumps break attempts.

    ``clearance_m`` refers to center displacement beyond each roof end; this is
    not a swept-body collision certificate. Recorded passage geometry contains
    no roof height, so this function makes no separate vertical-clearance claim.
    It never reads actions, model IDs, final field X, or requested navigation phase.
    """
    if not 0<=fold_threshold<=1 or clearance_m<0 or lateral_clearance_m<0 or max_gap_s<=0 or max_speed_mps<=0:
        raise ValueError('Invalid tunnel evaluation thresholds')
    trace=list(trace);geometries={}
    for row in trace:
        geometry=_geometry(row.get('navigation',{}).get('passage'))
        if geometry:
            key=(geometry['id'],*(round(v,8) for v in (*geometry['center'],*geometry['axis'],geometry['half'],geometry['width'])))
            geometries[key]=geometry
    samples=[_sample(row) for row in trace];crossings=[]
    for geometry in geometries.values():
        cx,cy=geometry['center'];ax,ay=geometry['axis'];half=geometry['half']
        outer=half+clearance_m;lateral_limit=geometry['width']/2-lateral_clearance_m
        if lateral_limit<=0:continue
        candidate=None;previous=None
        for sample in samples:
            if sample is None:candidate=None;previous=None;continue
            sample=dict(sample)
            dx,dy=sample['position'][0]-cx,sample['position'][1]-cy
            sample.update(along=dx*ax+dy*ay,across=-dx*ay+dy*ax)
            continuous=previous is not None and 0<sample['time']-previous['time']<=max_gap_s+1e-8
            if continuous:
                continuous=math.dist(sample['position'],previous['position'])<=max_speed_mps*(sample['time']-previous['time'])+.01
            if not continuous:candidate=None
            if abs(sample['across'])>lateral_limit+1e-8:
                candidate=None;previous=sample;continue
            if candidate is None:
                if abs(sample['along'])>=outer:
                    candidate=dict(direction=1 if sample['along']<0 else -1,entry=sample,
                        observed_roof_samples=0,min_fold=1.,max_abs_across=0.,roof_enter=None,roof_exit=None)
            elif candidate['direction']*sample['along']<=-outer:
                # Still outside, or returned to the entry after an incomplete
                # attempt. A later successful traversal begins here.
                candidate.update(entry=sample,observed_roof_samples=0,min_fold=1.,max_abs_across=0.,roof_enter=None,roof_exit=None)
            elif previous is not None:
                interval=_slab_interval(previous['along'],sample['along'],half)
                if interval is not None:
                    folds=[previous['fold']+(sample['fold']-previous['fold'])*u for u in interval]
                    cross=[previous['across']+(sample['across']-previous['across'])*u for u in interval]
                    if min(folds)<fold_threshold-1e-8 or max(abs(v) for v in cross)>lateral_limit+1e-8:
                        candidate=None;previous=sample;continue
                    times=[previous['time']+(sample['time']-previous['time'])*u for u in interval]
                    if candidate['roof_enter'] is None:candidate['roof_enter']=min(times)
                    candidate['roof_exit']=max(times)
                    candidate['min_fold']=min(candidate['min_fold'],*folds)
                    candidate['max_abs_across']=max(candidate['max_abs_across'],*(abs(v) for v in cross))
                if abs(sample['along'])<=half+1e-8:
                    candidate['observed_roof_samples']+=1
                if candidate['direction']*sample['along']>=outer and candidate['observed_roof_samples']:
                    crossings.append(dict(passage_id=geometry['id'],geometry=geometry,
                        direction=candidate['direction'],entry=candidate['entry'],exit=sample,
                        roof_enter_time=candidate['roof_enter'],roof_exit_time=candidate['roof_exit'],
                        observed_roof_samples=candidate['observed_roof_samples'],
                        min_fold_during_roof_intersection=candidate['min_fold'],
                        max_abs_across_during_roof_intersection=candidate['max_abs_across']))
                    candidate=None
            previous=sample
    crossings.sort(key=lambda row:row['exit']['time'])
    return dict(passed_tunnel=bool(crossings),algorithm='recorded_passage_geometry_v1',crossings=crossings,
        tested_passages=len(geometries),trace_samples=len(samples),invalid_samples=sum(s is None for s in samples),
        thresholds=dict(fold=fold_threshold,longitudinal_clearance_m=clearance_m,
            lateral_clearance_m=lateral_clearance_m,max_gap_s=max_gap_s,max_speed_mps=max_speed_mps),
        limitations=['Tracked-center geometry, not a swept-body collision certificate',
            'No independent vertical-clearance test without recorded roof height',
            'Requires an observed roof-interior sample; sparse or incomplete traces may remain unconfirmed'])
