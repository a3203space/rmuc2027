import unittest
from unittest.mock import patch
import json
import numpy as np
from rmuc2027_research.rl.warp_kinematic.protocol import MotionFrame
from rmuc2027_research.rl.warp_kinematic import protocol

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

    def test_native_and_standard_encoding_preserve_double_bits_and_framing(self):
        data=self.data()
        values=np.resize([np.nextafter(0.,1.),np.finfo(np.float64).max,-0.,1.0000000000000002,1.e-250,-1.e200],18).reshape(6,1,3)
        data.update(ids=['蓝方\n4'],positions=values)
        frame=MotionFrame(**data)
        native=frame.encoded_response(17)
        with patch.object(protocol,'_native_json',None):standard=frame.encoded_response(17)
        for payload in (native,standard):
            self.assertEqual(payload.count(b'\n'),1)
            decoded=json.loads(payload)
            self.assertEqual(decoded,frame.response(17))
            self.assertEqual(np.asarray(decoded['positions'],dtype=np.float64).tobytes(),values.tobytes())

    def test_validated_storage_cannot_be_made_writable(self):
        data=self.data();ids=['red-3'];data['ids']=ids
        frame=MotionFrame(**data);ids[0]='changed';data['positions'][:]=np.nan
        self.assertEqual(json.loads(frame.encoded_response(0))['ids'],['red-3'])
        for name in ('positions','velocities','headings','folded','grounded'):
            with self.subTest(field=name),self.assertRaises(ValueError):getattr(frame,name).setflags(write=True)
        for sequence in (True,-1,np.int64(1)):
            with self.assertRaises(ValueError):frame.encoded_response(sequence)
