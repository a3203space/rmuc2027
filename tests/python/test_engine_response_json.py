import json
import unittest
from unittest.mock import patch
from rmuc2027_core import engine


class ResponseDecoderTests(unittest.TestCase):
    def test_native_decoder_matches_standard_library_frames(self):
        try:
            import orjson
        except ImportError:
            self.skipTest('Install research/requirements-rl.txt for native decoder comparison')
        results=[]
        for decoder in (json,orjson):
            with patch.object(engine,'_response_json',decoder),engine.Match(seed=42,battle_settings={'drones_enabled':False}) as match:
                frames=[match.snapshot()]
                for tick in range(60):
                    frames.append(match.step({'red':{'4':{'move':[.1,.2],'aim':tick*.01,'transform':tick>20}}}))
                observations=[match.observation('red',[4]),match.observation('blue',[3])]
                results.append((frames,observations,match.digest()))
        self.assertEqual(results[0],results[1])
