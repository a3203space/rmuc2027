"""Same-state reuse must preserve exact bytes, viewer scopes and failure handling."""
import unittest
import threading
from unittest.mock import patch
from rmuc2027_core import engine


class ObservationReuseTests(unittest.TestCase):
    def match(self):
        match=engine.Match(seed=42,battle_settings={'drones_enabled':False})
        self.addCleanup(match.close)
        return match

    def test_cached_batches_are_byte_exact_and_single_returns_fresh_values(self):
        match=self.match();viewers=[('red',[3]),('blue',[7]),('red',None),('red',[4,3])]
        reference=match.observations(viewers)
        self.assertNotEqual(reference[1][0],reference[1][2])
        with patch.object(match,'_request',wraps=match._request) as request:
            self.assertEqual(match.observations(iter(viewers)),reference)
            value=match.observation('red',[3]);value['robots'].clear()
            value['arena'].clear();value['controlled_ids'].append(7)
            self.assertTrue(match.observation('red',[3])['robots'])
            self.assertTrue(match.observation('red',[3])['arena'])
            self.assertEqual(match.observation('red',[3])['controlled_ids'],['3'])
            self.assertEqual(request.call_count,0)
        with patch.object(engine,'_CACHE_POLICY_OBSERVATIONS',False):
            self.assertEqual(match.observations(viewers),reference)
        self.assertEqual(match.digest(),match.digest())

    def test_single_to_partial_batch_reuse_preserves_order_and_scope(self):
        match=self.match();slots=[3]
        own=match.observation('red',slots);slots[0]=4
        with patch.object(match,'_request',wraps=match._request) as request:
            ids,lines=match.observations([('blue',[3]),('red',[3]),('red',[4])])
            self.assertEqual(request.call_args.args[0]['requests'],[
                {'team':'blue','slots':(3,)},{'team':'red','slots':(4,)}])
        self.assertEqual(ids,[['3'],['3'],['4']])
        self.assertEqual(engine.decode_response_json(lines[1]),own)
        with patch.object(engine,'_CACHE_POLICY_OBSERVATIONS',False):
            self.assertEqual(match.observations([('blue',[3]),('red',[3]),('red',[4])]),(ids,lines))
        self.assertEqual(match._observation_cache[('red',(3,))][0],('3',))

    def test_steps_finish_and_unknown_requests_invalidate_even_on_error(self):
        match=self.match();before=match.observation('red',[3])
        match.step({'red':{'3':{'move':[.1,0]}}})
        self.assertFalse(match._observation_cache)
        after=match.observation('red',[3]);self.assertNotEqual(before['tick'],after['tick'])
        with self.assertRaises(ValueError):match._request({'op':'unknown'})
        self.assertFalse(match._observation_cache)
        match.observation('red',[3]);match.finish('red','test')
        self.assertFalse(match._observation_cache)
        with patch.object(match,'_request',wraps=match._request) as request:
            match.observation('red',[3]);self.assertEqual(request.call_count,1)
        self.assertEqual(match.status,'finished')

    def test_compact_operator_and_invalid_viewers_cannot_hit_policy_cache(self):
        match=self.match();own=match.observation('red',[3])
        with patch.object(match,'_request',wraps=match._request) as request:
            match.compact_observation('red',[3]);match.operator_observation('red',3)
            self.assertEqual(request.call_count,2)
            self.assertEqual(match.observation('red',[True])['controlled_ids'],[])
            self.assertEqual(match.observation('red',[3.0])['controlled_ids'],[])
            with self.assertRaises(ValueError):match.observation('invalid',[3])
        self.assertEqual(match.observation('red',[3]),own)
        self.assertEqual(len(match._observation_cache),1)

    def test_hits_cannot_bypass_broken_or_closed_transport(self):
        match=self.match();match.observation('red',[3])
        for operation in (lambda:match.observation('red',[3]),lambda:match.observations([('red',[3])]),lambda:match.observations([])):
            match._broken=True
            with self.assertRaisesRegex(RuntimeError,'interrupted'):operation()
        match.close()
        with self.assertRaisesRegex(RuntimeError,'closed'):match.observation('red',[3])

    def test_cache_is_bounded_and_interrupted_batch_stays_unusable(self):
        import itertools
        match=self.match()
        for n in range(1,4):
            for slots in itertools.permutations((1,3,4,7),n):match.observation('red',slots)
        self.assertEqual(len(match._observation_cache),32)
        read=match._read_line;calls=0
        def interrupt():
            nonlocal calls
            calls+=1
            if calls==3:raise KeyboardInterrupt
            return read()
        with patch.object(match,'_read_line',side_effect=interrupt):
            with self.assertRaises(KeyboardInterrupt):match.observations([('blue',[3]),('blue',[7])])
        with self.assertRaisesRegex(RuntimeError,'interrupted'):match.observation('red',[7,4,3])

    def test_concurrent_step_cannot_be_followed_by_stale_cache_insertion(self):
        match=self.match();ready=threading.Event();release=threading.Event()
        read=match._read_line;errors=[];observed=[];once=True
        def paused_read():
            nonlocal once
            line=read()
            if once:
                once=False;ready.set()
                if not release.wait(2):raise TimeoutError('test release missing')
            return line
        def observe():
            try:observed.append(match.observation('blue',[7]))
            except BaseException as error:errors.append(error)
        def advance():
            try:match.step({})
            except BaseException as error:errors.append(error)
        with patch.object(match,'_read_line',side_effect=paused_read):
            query=threading.Thread(target=observe);query.start()
            try:
                self.assertTrue(ready.wait(2))
                step=threading.Thread(target=advance);step.start()
            finally:release.set();query.join(2)
            step.join(2)
            self.assertFalse(query.is_alive());self.assertFalse(step.is_alive())
        self.assertFalse(errors)
        self.assertEqual(observed[0]['tick'],0)
        self.assertFalse(match._observation_cache)
        self.assertEqual(match.observation('blue',[7])['tick'],1)


if __name__=='__main__':unittest.main()
