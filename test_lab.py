import json
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import lab


class LabTests(unittest.TestCase):
    def test_resume_after_interruption_skips_completed_runs(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            path = Path(directory) / 'runs.jsonl'
            argv = ['lab.py', 'run', '--runs', '1', '--log', str(path)]

            def record(config, index, condition):
                return dict(config=config, config_id=lab.config_id(config), index=index,
                            condition=condition, status='failed', rounds=8,
                            duration_seconds=0)

            saved = record(dict(lab.DEFAULTS), 0, 'no_memory')
            # Emulate a crash after the first condition was persisted.
            with patch('sys.argv', argv), patch('sys.stdout', new_callable=io.StringIO), \
                    patch('urllib.request.urlopen', side_effect=AssertionError('Network forbidden')) as http:
                with patch.object(lab, 'simulate', side_effect=[saved, RuntimeError('crash')]):
                    with self.assertRaisesRegex(RuntimeError, 'crash'):
                        lab.main()
                self.assertEqual(len(lab.read_logs(path)), 1)
                # Also emulate an incomplete trailing JSON write.
                with path.open('a') as stream:
                    stream.write('{"incomplete":')
                with patch.object(lab, 'simulate', side_effect=record) as simulate:
                    lab.main()
                    simulate.assert_called_once_with(dict(lab.DEFAULTS), 0, 'memory')
                self.assertEqual([(r['index'], r['condition']) for r in lab.read_logs(path)],
                                 [(0, 'no_memory'), (0, 'memory')])
                before = path.read_bytes()
                # Same command again: no simulation and no file modification.
                with patch.object(lab, 'simulate', side_effect=AssertionError('Must skip')) as simulate:
                    lab.main()
                    simulate.assert_not_called()
                self.assertEqual(path.read_bytes(), before)
                http.assert_not_called()

    def test_resume_requires_matching_configuration_and_completed_status(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            path = Path(directory) / 'runs.jsonl'
            old_config = dict(lab.DEFAULTS, version=2)
            old = dict(config=old_config, config_id=lab.config_id(old_config), index=0,
                       condition='no_memory', status='agreed')
            unfinished = dict(config=dict(lab.DEFAULTS), config_id=lab.config_id(lab.DEFAULTS),
                              index=0, condition='memory', status='running')
            path.write_text(json.dumps(old) + '\n' + json.dumps(unfinished) + '\n')

            def record(config, index, condition):
                return dict(config=config, config_id=lab.config_id(config), index=index,
                            condition=condition, status='agreed', rounds=1, duration_seconds=0)

            with patch('sys.argv', ['lab.py', 'run', '--runs', '1', '--log', str(path)]), \
                    patch('sys.stdout', new_callable=io.StringIO), \
                    patch('urllib.request.urlopen', side_effect=AssertionError('Network forbidden')) as http, \
                    patch.object(lab, 'simulate', side_effect=record) as simulate:
                lab.main()
                self.assertEqual([call.args[2] for call in simulate.call_args_list],
                                 ['no_memory', 'memory'])
                simulate.reset_mock()
                lab.main()
                simulate.assert_not_called()
                http.assert_not_called()

    def test_validation_categories(self):
        cases = [('bad', 'json'), ('{}', 'schema'),
                 (json.dumps(dict(allocation=dict(A=30, B=40, C=29), reason='ok')), 'game'),
                 (json.dumps(dict(allocation=dict(A=True, B=40, C=59), reason='ok')), 'schema')]
        for raw, category in cases:
            with self.subTest(raw=raw), self.assertRaises(lab.Invalid) as caught:
                lab.validate(raw, 'proposal', 'A')
            self.assertEqual(caught.exception.category, category)
        with self.assertRaises(lab.Invalid):
            lab.validate('{"vote":"accept","reason":"ok"}', 'vote', 'B', dict(A=34, B=33, C=33))

    def test_context_isolation(self):
        history = [dict(round=1, proposal=dict(allocation=dict(A=30, B=40, C=30)))]
        for agent in lab.AGENTS:
            for memory in (False, True):
                msgs = lab.messages(agent, 2, 'vote', history, memory, dict(A=30, B=40, C=30))
                self.assertEqual(msgs[0]['content'], lab.ROLES[agent] + '\n' + lab.RULES)
                context = json.loads(msgs[1]['content'])
                self.assertEqual('ИСТОРИЯ (только для справки)' in context, memory)
                self.assertEqual(context['rounds_remaining_including_current'], 7)
                self.assertEqual(context['ТЕКУЩЕЕ ПРЕДЛОЖЕНИЕ'], dict(A=30, B=40, C=30))
                self.assertIn('Голосуй только', context['задание'])

    def test_current_proposal_is_separate_from_history(self):
        old = dict(A=80, B=10, C=10)
        current = dict(A=30, B=40, C=30)
        history = [dict(round=1, proposal=dict(allocation=old))]
        for memory in (False, True):
            context = json.loads(lab.messages('C', 2, 'vote', history, memory, current)[1]['content'])
            self.assertEqual(context['ТЕКУЩЕЕ ПРЕДЛОЖЕНИЕ'], current)
            self.assertIn('Не голосуй за предложения из истории', context['задание'])
            if memory:
                self.assertEqual(context['ИСТОРИЯ (только для справки)'], history)
            else:
                self.assertNotIn('ИСТОРИЯ (только для справки)', context)
            proposal_context = json.loads(lab.messages('A', 2, 'proposal', history, memory)[1]['content'])
            self.assertNotIn('ТЕКУЩЕЕ ПРЕДЛОЖЕНИЕ', proposal_context)
            self.assertEqual('ИСТОРИЯ (только для справки)' in proposal_context, memory)

    def test_retry_and_failed_run(self):
        def invalid(*args):
            return 'invalid'
        row = lab.simulate(lab.DEFAULTS, 0, 'no_memory', invalid)
        self.assertEqual(row['status'], 'failed')
        self.assertEqual(row['rounds'], 8)
        self.assertEqual(len(row['calls']), 16)
        self.assertEqual(row['invalid_actions'], 8)
        self.assertEqual([r['proposer'] for r in row['history']], list('ABCABCAB'))
        self.assertEqual(lab.metrics([row])['invalid_response_rate'], 1)
        self.assertIsNone(lab.metrics([row])['gini'])

    def test_agreement_and_history(self):
        seen = []
        def transport(config, seed, msgs, kind):
            seen.append((seed, msgs))
            return json.dumps(dict(allocation=dict(A=30, B=40, C=30), reason='ok')
                              if kind == 'proposal' else dict(vote='accept', reason='ok'))
        row = lab.simulate(lab.DEFAULTS, 3, 'memory', transport)
        self.assertEqual(row['status'], 'agreed')
        self.assertEqual(row['rounds'], 1)
        self.assertEqual(len(row['calls']), 3)
        self.assertEqual(lab.metrics([row])['accept_rate'], 1)
        self.assertEqual(lab.metrics([row])['almost_agreement_count'], 0)
        self.assertTrue(all(seed == lab.DEFAULTS['base_seed'] + 3 for seed, _ in seen))
        self.assertTrue(all(json.loads(m[1]['content'])['ИСТОРИЯ (только для справки)'] == [] for _, m in seen))
        self.assertAlmostEqual(lab.gini(row['allocation']), 1/15)

    def test_retry_recovers(self):
        answers = iter(['oops', '{"vote":"reject","reason":"ok"}'])
        calls = []
        value = lab.action(lab.DEFAULTS, 1, 'C', 1, 'vote', [], False, calls,
                           dict(A=30, B=40, C=30), lambda *args: next(answers))
        self.assertEqual(value['vote'], 'reject')
        self.assertEqual([c['valid'] for c in calls], [False, True])

    def test_long_reason_is_valid_and_trimmed_only_for_logging(self):
        reason = 'Первое предложение. Второе предложение! ' * 30
        raw = json.dumps(dict(vote='reject', reason=reason))
        value = lab.validate(raw, 'vote', 'C')
        self.assertEqual(value['reason'], reason)
        logged = lab.log_value(dict(raw=raw, result=value))
        self.assertEqual(len(logged['result']['reason']), 200)
        self.assertEqual(len(json.loads(logged['raw'])['reason']), 200)
        self.assertEqual(value['reason'], reason)
        for vote in (True, 'maybe', 0, []):
            with self.assertRaises(lab.Invalid):
                lab.validate(json.dumps(dict(vote=vote, reason='ok')), 'vote', 'C')

    def test_ollama_payload(self):
        class Response:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def read(self): return b'{"message":{"content":"{}"}}'
        with patch('urllib.request.urlopen', return_value=Response()) as request:
            lab.chat(dict(lab.DEFAULTS, model='qwen3:1.7b'), 7, [], 'vote')
        payload = json.loads(request.call_args.args[0].data)
        self.assertFalse(payload['think'])
        self.assertFalse(payload['stream'])
        self.assertEqual(payload['format'], lab.SCHEMAS['vote'])
        self.assertEqual(payload['options']['seed'], 7)

    def test_diagnostics_without_agreements(self):
        def turn(number, shares, votes):
            return dict(round=number, proposer='A', proposal=dict(allocation=shares),
                        votes=dict(A=dict(vote='accept', automatic=True), **votes))
        row = dict(index=0, status='failed', history=[
            turn(1, dict(A=50, B=40, C=10),
                 dict(B=dict(vote='accept'), C=dict(vote='reject'))),
            dict(round=2, proposer='B', proposal=None, votes={}),
            turn(4, dict(A=30, B=40, C=30),
                 dict(B=dict(vote='reject', invalid=True), C=dict(vote='reject')))])
        result = lab.diagnostics([row])
        self.assertEqual(result['by_round'][1]['accept_rate'], .5)
        self.assertAlmostEqual(result['by_round'][1]['accept_rate_including_proposer'], 2/3)
        self.assertIsNone(result['by_round'][2]['proposal_gini'])
        self.assertEqual(result['blockers']['C']['sole_rejects'], 1)
        self.assertEqual(result['almost_agreements_by_run'], [dict(index=0, count=1)])
        self.assertEqual(result['by_round'][4]['invalid_reject_count'], 1)
        self.assertEqual(result['concessions']['A']['mean_net_concession'], 20)
        self.assertEqual(result['concessions']['A']['mean_step_concession'], 20)
        self.assertIsNone(result['concessions']['B']['mean_net_concession'])
        self.assertAlmostEqual(result['proposals'][0]['gini'], lab.gini(dict(A=50, B=40, C=10)))


if __name__ == '__main__':
    unittest.main()
