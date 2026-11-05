import unittest
import numpy as np
from rmuc2027_research.rl.warp_kinematic.protocol import MotionFrame

class FrameTests(unittest.TestCase):
    def data(self):
        return dict(ids=('red-3',), positions=np.zeros((6,1,3)), velocities=np.zeros((6,1,3)),
                    headings=np.zeros((6,1)), folded=np.zeros((6,1)), grounded=np.ones((6,1),dtype=bool))

    def test_invalid_whole_frame_rejected(self):
        for field,value in [('positions',np.zeros((5,1,3))), ('velocities',np.full((6,1,3),np.nan)),
                            ('folded',np.full((6,1),2.)), ('grounded',np.ones((6,1)))]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                MotionFrame(**{**self.data(),field:value})

    def test_owned_samples_and_sequence(self):
        data=self.data();frame=MotionFrame(**data)
        data['positions'][:]=42
        self.assertEqual(frame.response(12)['positions'][0][0],[0.,0.,0.])
        self.assertEqual(frame.response(12)['sequence'],12)
        self.assertFalse(frame.positions.flags.writeable)
        with self.assertRaises(ValueError):frame.response(-1)
