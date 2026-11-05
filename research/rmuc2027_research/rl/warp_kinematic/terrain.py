"""Public-map height queries on CUDA, without triangulated render geometry."""
import math
import numpy as np
import warp as wp


@wp.struct
class TerrainData:
    bounds: wp.array(dtype=wp.vec4)
    ranges: wp.array(dtype=wp.vec2i)
    vertices: wp.array(dtype=wp.vec2)
    profiles: wp.array(dtype=wp.vec4)
    ramps: wp.array(dtype=wp.vec4)
    kinds: wp.array(dtype=wp.int32)
    edge_ramps: wp.array(dtype=wp.float32)
    width: float
    height: float
    floor_slope: float


@wp.func
def surface_height(scene: TerrainData, p: wp.vec2):
    result = float(0.0)
    for k in range(scene.bounds.shape[0]):
        b=scene.bounds[k]
        if p[0]>=b[0] and p[0]<=b[2] and p[1]>=b[1] and p[1]<=b[3]:
            span=scene.ranges[k]
            inside=int(0)
            on_edge=int(0)
            edge_distance=float(1.e10)
            for j in range(span[1]):
                a=scene.vertices[span[0]+j]
                c=scene.vertices[span[0]+(j+1)%span[1]]
                cross=(p[0]-a[0])*(c[1]-a[1])-(p[1]-a[1])*(c[0]-a[0])
                edge_distance=wp.min(edge_distance,wp.abs(cross)/wp.max(1.e-8,wp.length(c-a)))
                if wp.abs(cross)<1.e-7 and p[0]>=wp.min(a[0],c[0]) and p[0]<=wp.max(a[0],c[0]) and p[1]>=wp.min(a[1],c[1]) and p[1]<=wp.max(a[1],c[1]):
                    on_edge=1
                if (a[1]>p[1]) != (c[1]>p[1]):
                    if p[0]<(c[0]-a[0])*(p[1]-a[1])/(c[1]-a[1])+a[0]:
                        inside=1-inside
            if inside==1 or on_edge==1:
                profile=scene.profiles[k]
                value=profile[1]
                if scene.kinds[k]==1:
                    ramp=scene.ramps[k]
                    a=wp.vec2(ramp[0],ramp[1]);d=wp.vec2(ramp[2],ramp[3])-a
                    t=wp.clamp(wp.dot(p-a,d)/wp.max(1.e-9,wp.dot(d,d)),0.0,1.0)
                    value=profile[0]+(profile[1]-profile[0])*t
                elif scene.kinds[k]==2:
                    phase=(p[0]-b[0])/profile[2]
                    value=profile[1]+profile[3]*(1.0-wp.cos(6.28318530718*phase))*0.5
                elif scene.edge_ramps[k]>0.0:
                    value=profile[0]+(profile[1]-profile[0])*wp.min(1.0,edge_distance/scene.edge_ramps[k])
                result=wp.max(result,value)
    return result+wp.max(0.0,wp.min(p[1],scene.height-p[1]))*scene.floor_slope


@wp.kernel
def sample_heights(scene: TerrainData, points: wp.array(dtype=wp.vec2), result: wp.array(dtype=float)):
    i=wp.tid()
    result[i]=surface_height(scene,points[i])


@wp.kernel
def limit_terrain(scene: TerrainData, previous: wp.array(dtype=wp.vec2),
                  positions: wp.array(dtype=wp.vec2), velocities: wp.array(dtype=wp.vec2),
                  radii: wp.array(dtype=wp.vec3), steps: wp.array(dtype=float),
                  heights: wp.array(dtype=float)):
    i=wp.tid()
    start=previous[i];end=positions[i];delta=end-start
    accepted=start
    old_height=surface_height(scene,start)
    blocked=int(0)
    # Twelve swept samples bound a 50 ms movement, including narrow ledges.
    for j in range(1,13):
        p=start+delta*(float(j)/12.0)
        h=surface_height(scene,p)
        if p[0]<radii[i][2] or p[0]>scene.width-radii[i][2] or p[1]<radii[i][2] or p[1]>scene.height-radii[i][2] or wp.abs(h-old_height)>steps[i]:
            blocked=1
        if blocked==0:
            accepted=p
            old_height=h
    positions[i]=accepted
    heights[i]=surface_height(scene,accepted)
    if blocked!=0:
        velocities[i]=wp.vec2(0.0,0.0)


class Terrain:
    def __init__(self, arena, device='cuda:0'):
        wp.init();self.device=wp.get_device(device)
        if not self.device.is_cuda: raise ValueError('Terrain requires CUDA')
        bounds=[];ranges=[];vertices=[];profiles=[];ramps=[];kinds=[];edges=[]
        for t in arena['terrain']:
            x,y,w,h=(t[k] for k in ('x','y','w','h'))
            polygon=t.get('polygon') or [(x,y),(x+w,y),(x+w,y+h),(x,y+h)]
            ranges.append((len(vertices),len(polygon)));vertices.extend(polygon)
            bounds.append((x,y,x+w,y+h))
            kind=1 if t['kind']=='ramp' and t.get('ramp_start') else 2 if t['kind']=='bumps' else 0
            if kind==2 and t.get('bump_pitch',0)<=0: raise ValueError('Invalid bump pitch')
            kinds.append(kind);edges.append(t.get('edge_ramp_m',0))
            profiles.append((t.get('base_elevation',0),t['elevation'],t.get('bump_pitch',0),t.get('bump_height',0)))
            ramps.append((*t.get('ramp_start',[0,0]),*t.get('ramp_end',[0,0])))
        self.data=TerrainData()
        for name,values,dtype in [('bounds',bounds,wp.vec4),('ranges',ranges,wp.vec2i),
                                 ('vertices',vertices,wp.vec2),('profiles',profiles,wp.vec4),
                                 ('ramps',ramps,wp.vec4),('kinds',kinds,wp.int32),('edge_ramps',edges,wp.float32)]:
            setattr(self.data,name,wp.array(values,dtype=dtype,device=self.device))
        self.data.width=float(arena['width']);self.data.height=float(arena['height'])
        self.data.floor_slope=math.tan(math.radians(arena.get('floor_slope_deg',0)))

    def heights(self, points):
        values=np.asarray(points,dtype=np.float32)
        if values.ndim!=2 or values.shape[1]!=2 or not np.isfinite(values).all():
            raise ValueError('Expected finite points[N,2]')
        p=wp.array(values,dtype=wp.vec2,device=self.device)
        result=wp.empty(len(values),dtype=float,device=self.device)
        wp.launch(sample_heights,len(values),inputs=[self.data,p,result],device=self.device)
        return result.numpy()


@wp.kernel
def update_body_height(ground: wp.array(dtype=float), offsets: wp.array(dtype=wp.vec2),
                       bounds: wp.array(dtype=wp.vec2)):
    i=wp.tid()
    bounds[i]=wp.vec2(ground[i]+offsets[i][0],ground[i]+offsets[i][1])
