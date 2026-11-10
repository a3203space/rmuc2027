"""Conservative sweep acceptance against the original exhaustive height sampler."""
import unittest
import numpy as np
import warp as wp
from rmuc2027_core.sdk import _REFERENCE_ARENA
from rmuc2027_research.rl.warp_kinematic.terrain import Terrain,TerrainData,surface_height,smooth_sweep,limit_terrain


@wp.kernel
def exhaustive(scene:TerrainData,previous:wp.array(dtype=wp.vec2),positions:wp.array(dtype=wp.vec2),
               velocities:wp.array(dtype=wp.vec2),radii:wp.array(dtype=wp.vec3),steps:wp.array(dtype=float),
               heights:wp.array(dtype=float),flight:wp.array(dtype=wp.int32)):
    i=wp.tid();start=previous[i];end=positions[i];delta=end-start;accepted=start
    old_height=surface_height(scene,start);blocked=int(0)
    for j in range(1,13):
        p=start+delta*(float(j)/12.0);h=surface_height(scene,p)
        if p[0]<radii[i][2] or p[0]>scene.width-radii[i][2] or p[1]<radii[i][2] or p[1]>scene.height-radii[i][2] or (flight[i]==0 and wp.abs(h-old_height)>steps[i]):blocked=1
        if blocked==0:accepted=p;old_height=h
    positions[i]=accepted;heights[i]=surface_height(scene,accepted)
    if blocked!=0:velocities[i]=wp.vec2(0.0,0.0)


@wp.kernel
def classify(scene:TerrainData,start:wp.array(dtype=wp.vec2),end:wp.array(dtype=wp.vec2),
             radii:wp.array(dtype=wp.vec3),steps:wp.array(dtype=float),flags:wp.array(dtype=wp.int32)):
    i=wp.tid();flags[i]=smooth_sweep(scene,start[i],end[i]-start[i],radii[i][2],steps[i])


class SmoothSweepTests(unittest.TestCase):
    def compare(self,arena,start,delta,step=None,flight=None):
        scene=Terrain(arena);n=len(start)
        start=np.asarray(start,np.float32);end=start+np.asarray(delta,np.float32)
        step=np.full(n,.04,np.float32) if step is None else np.asarray(step,np.float32)
        flight=np.zeros(n,np.int32) if flight is None else np.asarray(flight,np.int32)
        a=wp.array(start,dtype=wp.vec2,device=scene.device)
        r=wp.array([[1,1,.1]]*n,dtype=wp.vec3,device=scene.device)
        s=wp.array(step,dtype=float,device=scene.device);f=wp.array(flight,dtype=wp.int32,device=scene.device)
        results=[]
        for kernel in (exhaustive,limit_terrain):
            p=wp.array(end,dtype=wp.vec2,device=scene.device);v=wp.array([[1.,2.]]*n,dtype=wp.vec2,device=scene.device)
            h=wp.empty(n,dtype=float,device=scene.device)
            wp.launch(kernel,n,inputs=[scene.data,a,p,v,r,s,h,f],device=scene.device)
            results.append([x.numpy() for x in (p,v,h)])
        for old,new in zip(*results):self.assertEqual(old.tobytes(),new.tobytes())
        flags=wp.empty(n,dtype=wp.int32,device=scene.device)
        wp.launch(classify,n,inputs=[scene.data,a,wp.array(end,dtype=wp.vec2,device=scene.device),r,s,flags],device=scene.device)
        return flags.numpy()

    def test_stock_map_random_and_boundary_sweeps_are_bit_exact(self):
        rng=np.random.default_rng(319)
        start=rng.uniform([-.2,-.2],[28.2,15.2],(50000,2)).astype(np.float32)
        delta=rng.normal(0,.035,(len(start),2)).astype(np.float32)
        delta[::17]*=30;delta[::23]=0
        # Bias additional samples at every polygon edge/vertex and the map ridge.
        points=[]
        for t in _REFERENCE_ARENA['terrain']:
            for p in t.get('polygon',[]):
                points.extend(np.asarray(p)+rng.normal(0,1e-5,(40,2)))
        points=np.asarray(points,np.float32)
        start=np.concatenate([start,points]);delta=np.concatenate([delta,rng.normal(0,.02,points.shape).astype(np.float32)])
        flags=self.compare(_REFERENCE_ARENA,start,delta,flight=np.arange(len(start),dtype=np.int32)%11==0)
        self.assertGreater(flags[:50000].mean(),.5)
        self.assertTrue((flags==0).any())

    def test_thin_steps_concavity_overlap_steep_and_fast_waves_fall_back(self):
        arena=dict(width=10,height=10,floor_slope_deg=3,terrain=[
            dict(x=3.,y=1.,w=.0002,h=8.,kind='platform',elevation=.4),
            dict(x=4.,y=1.,w=2.,h=3.,polygon=[[4,1],[6,1],[4.1,2],[6,4],[4,4]],kind='platform',elevation=.2),
            dict(x=7.,y=1.,w=1.,h=8.,kind='bumps',elevation=.2,bump_height=.1,bump_pitch=.0001),
            dict(x=1.,y=6.,w=1.,h=2.,kind='ramp',elevation=.9,base_elevation=0.,ramp_start=[1,6],ramp_end=[1.001,6]),
            dict(x=1.,y=1.,w=1.,h=2.,kind='platform',elevation=.2,edge_ramp_m=.1)])
        rng=np.random.default_rng(921);start=rng.uniform(.2,9.8,(10000,2));delta=rng.normal(0,.1,(10000,2))
        self.compare(arena,start,delta,step=rng.choice([.04,.0001,.5],10000))
        flags=self.compare(arena,[[2.9999,5],[5,2],[7.5,5],[1.0002,7]],[[.002,0],[.5,.1],[.02,0],[.005,0]])
        np.testing.assert_array_equal(flags,np.zeros(4))

    def test_flat_and_continuous_bevel_paths_can_skip_samples(self):
        arena=dict(width=10,height=10,terrain=[dict(x=2.,y=2.,w=4.,h=4.,kind='platform',elevation=.2,base_elevation=0.,edge_ramp_m=.2)])
        flags=self.compare(arena,[[1,1],[1.999,3],[3,3],[5.999,3]],[[.03,0],[.02,0],[0,.04],[.02,0]])
        np.testing.assert_array_equal(flags,np.ones(4))

    def test_large_height_quantization_keeps_exhaustive_samples(self):
        # Even a gentle floor slope can quantize into >4 cm jumps after
        # addition to a large float32 elevation. Coordinate error alone is insufficient.
        arena=dict(width=10,height=10,floor_slope_deg=3,terrain=[dict(x=0,y=0,w=10,h=10,kind='platform',elevation=1.e6)])
        flags=self.compare(arena,[[2,.59],[3,1.78]],[[0,.04],[0,.04]])
        np.testing.assert_array_equal(flags,np.zeros(2))

    def test_bow_tie_corner_set_does_not_imply_a_solid_rectangle(self):
        arena=dict(width=10,height=10,terrain=[dict(x=2,y=2,w=2,h=2,kind='platform',elevation=.3,polygon=[[2,2],[4,4],[2,4],[4,2]])])
        flags=self.compare(arena,[[2.79,2.8]],[[.03,0]])
        np.testing.assert_array_equal(flags,[0])

    def test_stationary_boundary_and_negative_step_keep_original_behavior(self):
        arena=dict(width=10,height=10,terrain=[dict(x=2,y=2,w=2,h=2,kind='platform',elevation=.3)])
        flags=self.compare(arena,[[2,3],[2,3],[.1,1]],[[0,0]]*3,step=[.04,-.01,.04])
        np.testing.assert_array_equal(flags,[1,0,1])


if __name__=='__main__':unittest.main()
