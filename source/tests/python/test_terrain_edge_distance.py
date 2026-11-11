import copy
import unittest
import numpy as np
from rmuc2027_research.rl.warp_kinematic.terrain import Terrain


class TerrainEdgeDistanceTests(unittest.TestCase):
    def test_ramp_and_bumps_ignore_bevel_distance_parameters(self):
        rng=np.random.default_rng(281)
        points=rng.uniform([1.5,1.5],[4.5,4.5],(2000,2)).astype(np.float32)
        for kind in ('ramp','bumps'):
            field=dict(x=2,y=2,w=2,h=2,kind=kind,elevation=.3,base_elevation=0.,
                       ramp_start=[2,2],ramp_end=[4,4],bump_pitch=.24,bump_height=.07)
            heights=[]
            for edge in (0.,.5):
                field['edge_ramp_m']=edge
                heights.append(Terrain(dict(width=10,height=10,terrain=[copy.deepcopy(field)])).heights(points))
            self.assertEqual(heights[0].tobytes(),heights[1].tobytes())

    def test_bevel_still_uses_polygon_edge_distance(self):
        points=[[2.02,3],[2.08,3],[3,3]]
        heights=[]
        for edge in (.1,.4):
            scene=Terrain(dict(width=10,height=10,terrain=[dict(x=2,y=2,w=2,h=2,kind='bevel',elevation=.2,base_elevation=0.,edge_ramp_m=edge)]))
            heights.append(scene.heights(points))
        np.testing.assert_allclose(heights[0],[.04,.16,.2],atol=2e-6)
        np.testing.assert_allclose(heights[1],[.01,.04,.2],atol=2e-6)


if __name__=='__main__':unittest.main()
