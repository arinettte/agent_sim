import unittest

import analyze_paired as analysis


def run(index, allocation, a_vote, c_vote):
    return dict(index=index, seed=1729+index, condition='test', history=[
        dict(round=1, proposer='B', proposal=dict(allocation=allocation, reason='test'),
             votes=dict(B=dict(vote='accept', automatic=True),
                        A=dict(vote=a_vote, reason='test'),
                        C=dict(vote=c_vote, reason='test')))])


class PairedTests(unittest.TestCase):
    def test_reason_search_rules(self):
        self.assertTrue(analysis.reason_flags('B и C получают по 50, что больше 40.')['mentions_B_share'])
        self.assertTrue(analysis.reason_flags('Доля Б равна 40.')['mentions_B_share'])
        self.assertTrue(analysis.reason_flags('Доля C — меньше 40.')['mentions_40'])
        self.assertFalse(analysis.reason_flags('Доля C равна 140 или 400.')['mentions_40'])
        self.assertFalse(analysis.reason_flags('B отвергает предложение.')['mentions_B_share'])
        self.assertTrue(analysis.reason_flags('Меньше 40, не согласен.')['mentions_40_bound'])
        self.assertFalse(analysis.reason_flags('C получает 40, что больше 33.')['mentions_40_bound'])

    def test_thresholds_and_automatic_vote_exclusion(self):
        row = run(0, dict(A=0, B=100, C=0), 'accept', 'accept')
        values = analysis.cluster(row)
        self.assertEqual(values['accept_rate'], (2, 2))
        self.assertEqual(values['conflict_per_AC_accept'], (2, 2))
        self.assertEqual(values['B_accepted_share'], (0, 0))
        self.assertEqual(len(analysis.extract(row)), 3)
        self.assertFalse(analysis.conflict('A', dict(A=20, B=40, C=40)))
        self.assertFalse(analysis.conflict('C', dict(A=0, B=40, C=60)))  # Gini exactly .4

    def test_identical_paired_conditions_have_zero_difference(self):
        # Two correlated clusters with completely different acceptance levels.
        rows = {i: run(i, dict(A=30, B=40, C=30), vote, vote)
                for i, vote in enumerate(('accept', 'reject'))}
        report = analysis.paired_bootstrap(dict(no_memory=rows, memory=rows), repeats=100)
        self.assertEqual(report['metrics']['accept_rate']['difference']['ci95'], [0, 0])
        self.assertEqual(report['metrics']['accept_rate']['no_memory']['ci95'], [0, 1])

    def test_missing_pairs_fail(self):
        with self.assertRaises(ValueError):
            analysis.paired_bootstrap(dict(no_memory={0: {}}, memory={}))


if __name__ == '__main__':
    unittest.main()
