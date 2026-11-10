import copy
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

import fusion_audit_statistics as statistics
from fusion_continuation_diagnostics import probability_result


class FusionAuditStatisticsTests(unittest.TestCase):
    def results(self, n=120):
        truth = np.arange(n, dtype=np.int64) % 3
        ids = [f'val/images/{index}.JPEG' for index in range(n)]
        names = ['zero', 'one', 'two']

        def record(correct):
            labels = np.where(np.arange(n) < correct, truth, (truth+1) % 3)
            probabilities = np.full((n, 3), .05, dtype=np.float32)
            probabilities[np.arange(n), labels] = .9
            return probability_result(probabilities, truth.copy(), ids, names)

        result = dict(source_cnn=record(60), source_vit=record(50),
            existing_joint=record(100), random_vit_cls_linear=record(65),
            reset_cnn_head_cls_linear=record(80), patch_mean_linear=record(85))
        result['probability_90_10'] = probability_result(
            .9*result['source_cnn']['probabilities']+.1*result['source_vit']['probabilities'],
            truth, ids, names)
        return result

    def test_exact_mcnemar_matches_integer_binomial_tails(self):
        for n in range(61):
            for corrected in range(n+1):
                harmed = n-corrected
                expected = 1. if n == 0 else min(1.,
                    2*sum(math.comb(n, k) for k in range(min(corrected, harmed)+1))/(2**n))
                result = statistics.exact_mcnemar(corrected, harmed)
                self.assertAlmostEqual(result['pvalue'], expected, places=12)
                self.assertEqual(result['discordant'], n)
        self.assertAlmostEqual(statistics.exact_mcnemar(8, 0)['pvalue'], .0078125, places=14)
        underflow = statistics.exact_mcnemar(10000, 0)
        self.assertEqual(underflow['pvalue'], 0.)
        self.assertTrue(underflow['numeric_underflow'])
        self.assertTrue(math.isfinite(underflow['log_pvalue']))
        for a, b in ((-1, 0), (True, 1), (1.5, 2)):
            with self.assertRaises(ValueError):
                statistics.exact_mcnemar(a, b)

    def test_holm_adjustment_is_monotone_in_sorted_pvalues_and_fixed_family(self):
        actual = statistics.holm_adjust(dict(a=.01, b=.04, c=.03, d=.002))
        expected = dict(a=.03, b=.06, c=.06, d=.008)
        for key in expected:
            self.assertAlmostEqual(actual[key], expected[key])
        self.assertEqual(list(actual), ['a', 'b', 'c', 'd'])
        with self.assertRaises(ValueError):
            statistics.holm_adjust(dict(invalid=float('nan')))

    def test_stratified_bootstrap_preserves_class_counts_not_pooled_variability(self):
        # All improvements are in class0 and all harms in class1. Resampling
        # within the fixed classes must retain exactly zero net difference.
        truth = np.array([0]*50+[1]*50, dtype=np.int64)
        a = truth == 0
        b = truth == 1
        np.random.seed(246)
        before = np.random.get_state()
        first = statistics.stratified_accuracy_interval(a, b, truth, seed=20261109)
        second = statistics.stratified_accuracy_interval(a, b, truth, seed=20261109)
        self.assertEqual(first, second)
        self.assertEqual(first['low_pp'], 0.)
        self.assertEqual(first['high_pp'], 0.)
        self.assertEqual([x['n'] for x in first['strata']], [50, 50])
        self.assertTrue(first['class_counts_preserved'])
        self.assertFalse(first['training_seed_resampling'])
        after = np.random.get_state()
        self.assertEqual(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])
        with self.assertRaises(ValueError):
            statistics.stratified_accuracy_interval(a.astype(int), b, truth, seed=1)

    def test_registered_primary_secondary_and_image_confidence_are_applied(self):
        results = self.results()
        plan = statistics.statistical_plan()
        actual = statistics.evaluate_audit_results(results, plan=plan)
        self.assertEqual(actual['n'], 120)
        self.assertEqual(actual['primary']['id'], 'joint_vs_cnn')
        self.assertEqual(actual['primary']['corrected'], 40)
        self.assertEqual(actual['primary']['harmed'], 0)
        self.assertTrue(actual['primary']['positive_improvement_established'])
        self.assertEqual(len(actual['secondary']), 4)
        for row in actual['secondary']:
            self.assertTrue(row['confirmatory_gate_open'])
            self.assertGreaterEqual(row['multiplicity_adjusted_pvalue'], row['exact_mcnemar']['pvalue'])
        self.assertEqual([x['model'] for x in actual['metrics']], list(statistics.MODELS))
        self.assertIsNone(actual['conclusions']['selected_best_model'])
        self.assertFalse(actual['conclusions']['training_seed_variability_estimated'])
        self.assertEqual(actual['statistical_plan'], plan)
        plan['alpha'] = 1.
        self.assertEqual(statistics.statistical_plan()['alpha'], .05)
        with self.assertRaisesRegex(ValueError, 'plan changed'):
            statistics.evaluate_audit_results(results, plan=plan)

    def test_primary_failure_closes_secondary_confirmatory_gate(self):
        results = self.results()
        results['existing_joint'] = copy.deepcopy(results['source_cnn'])
        actual = statistics.evaluate_audit_results(results)
        self.assertFalse(actual['primary']['positive_improvement_established'])
        self.assertEqual(actual['primary']['exact_mcnemar']['pvalue'], 1.)
        for row in actual['secondary']:
            self.assertFalse(row['confirmatory_gate_open'])
            self.assertFalse(row['positive_improvement_established'])

    def test_input_alignment_nll_and_fixed_mixture_are_required(self):
        for mutation in ('ids', 'targets', 'nll', 'extra', 'mixture'):
            results = self.results()
            if mutation == 'ids':
                results['existing_joint']['image_ids'] = list(reversed(results['existing_joint']['image_ids']))
            elif mutation == 'targets':
                results['existing_joint']['targets'][0] = 2
            elif mutation == 'nll':
                results['existing_joint']['nll'][0] = float('nan')
            elif mutation == 'extra':
                results['new_candidate'] = results['existing_joint']
            elif mutation == 'mixture':
                results['probability_90_10'] = copy.deepcopy(results['source_vit'])
            with self.assertRaises(ValueError, msg=mutation):
                statistics.evaluate_audit_results(results)
        results = self.results()
        results['existing_joint']['metrics']['accuracy'] = -50
        self.assertGreater(statistics.evaluate_audit_results(results)['metrics'][2]['accuracy'], 0)

    def test_report_preserves_fixed_order_and_explicit_provenance(self):
        directory = Path(__file__).resolve().parents[1]/'runs'/'fusion_audit_statistics_tests_tmp'
        directory.mkdir(parents=True, exist_ok=True)
        summary = statistics.evaluate_audit_results(self.results())
        summary['evaluation_provenance'] = dict(target='official_validation',
            legacy_internal_audit_previously_evaluated=True,
            prior_official_use='not detected in provided records; global history unverified')
        with tempfile.TemporaryDirectory(dir=directory) as temporary:
            output = Path(temporary)
            statistics.write_report(output, summary)
            report = (output/'report.md').read_text(encoding='utf-8')
            self.assertIn('Holm', report)
            self.assertIn('class-stratified', report)
            self.assertIn('not proof of equivalence', report)
            self.assertIn('legacy_internal_audit_previously_evaluated', report)
            self.assertLess(report.index('|source_cnn|'), report.index('|existing_joint|'))
            saved = json.loads((output/'statistics.json').read_text(encoding='utf-8'))
            self.assertEqual(saved, summary)
            self.assertTrue((output/'metrics.csv').is_file())
            self.assertTrue((output/'comparisons.csv').is_file())
            statistics.write_report(output, summary)
            self.assertEqual((output/'report.md').read_text(encoding='utf-8'), report)


if __name__ == '__main__':
    unittest.main()
