import copy
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

import pairwise_resolver_diagnostics as diagnostics
from fusion_continuation_diagnostics import probability_result


def cache_fixture(n=12):
    logits = torch.tensor([[4., 3., 2., 1., 0., -1.]]).repeat(n, 1)
    return dict(features=torch.arange(n*3).reshape(n, 3).float(),
        patch_tokens=torch.arange(n*4*2).reshape(n, 4, 2).float(),
        anchor_logits=logits, labels=torch.arange(n)%6,
        image_ids=[f'image_{i:03d}' for i in range(n)], class_names=[f'c{i}' for i in range(6)],
        identity=dict(split='selection'))


class FakeResolver(nn.Module):
    def __init__(self):
        super().__init__(); self.scale = nn.Parameter(torch.tensor(.1)); self.seen_patches = []
        self.resolver = nn.Module()
        self.resolver.register_buffer('train_patch_mean_ready', torch.tensor(True))
        self.register_buffer('train_patch_mean', cache_fixture()['patch_tokens'].mean(0))
    def forward_cached(self, features, patches, anchor_logits=None, collect=False):
        self.seen_patches.append(patches.detach().cpu())
        candidates = torch.argsort(anchor_logits, descending=True, stable=True)[:, :5]
        delta = torch.zeros_like(anchor_logits)
        # A valid bounded antisymmetric correction that keeps the top candidate.
        delta[:, 0] = self.scale; delta[:, 1] = -self.scale
        return dict(logits=anchor_logits+delta, anchor_logits=anchor_logits,
            delta=delta, candidates=candidates, eta=torch.tensor(.2))


def result(predictions, labels=None):
    n = len(predictions); labels = np.zeros(n, dtype=np.int64) if labels is None else labels
    p = np.full((n, 2), .1, dtype=np.float32)
    p[np.arange(n), predictions] = .9
    return probability_result(p, labels, [str(i) for i in range(n)], ['a', 'b'])


class PairwiseDiagnosticsTests(unittest.TestCase):
    def test_complete_selection_records_bounds_and_paired_counts(self):
        cache = cache_fixture(); model = FakeResolver()
        observed, stats = diagnostics.evaluate_cached_resolver(model, cache, 5, 'cpu')
        self.assertEqual(observed['metrics']['n'], 12)
        self.assertEqual(stats['predicted_outside_candidates'], 0)
        self.assertEqual(stats['protected_prediction_changes'], 0)
        self.assertAlmostEqual(stats['candidate_target_coverage'], 10/12)
        self.assertEqual(stats['paired_vs_anchor']['corrected'], 0)
        self.assertLess(stats['max_abs_delta_sum'], 1e-6)
        self.assertTrue(all(not x.requires_grad for x in model.seen_patches))

    def test_expired_deadline_never_returns_a_partial_split_as_final_score(self):
        with self.assertRaises(diagnostics.AnalysisDeadline):
            diagnostics.evaluate_cached_resolver(FakeResolver(), cache_fixture(), 4, 'cpu', deadline_unix=time.time()-1)

    def test_global_mismatch_is_label_blind_and_crosses_batch_boundaries(self):
        cache = cache_fixture(20); model = FakeResolver()
        _, stats = diagnostics.evaluate_cached_resolver(model, cache, 4, 'cpu', patch_mode='global_mismatch', mismatch_seed=42)
        supplied = torch.cat(model.seen_patches)
        self.assertTrue(bool((supplied[:, 0, 0] != cache['patch_tokens'][:, 0, 0]).all()))
        policy = stats['mismatch_policy']
        self.assertFalse(policy['pairing_uses_labels']); self.assertFalse(policy['pairing_uses_batch_boundaries'])
        self.assertEqual(policy['same_image_pairs'], 0)

    def test_train_mean_replaces_only_the_extra_patch_information(self):
        cache = cache_fixture(); model = FakeResolver()
        mean = cache['patch_tokens'].mean(0)
        diagnostics.evaluate_cached_resolver(model, cache, 4, 'cpu', patch_mode='train_mean', train_mean=mean)
        self.assertTrue(torch.equal(torch.cat(model.seen_patches), mean[None].repeat(12, 1, 1)))
        with self.assertRaises(ValueError):
            diagnostics.evaluate_cached_resolver(model, cache, 4, 'cpu', patch_mode='train_mean')

    def test_invalid_zero_sum_or_topk_details_are_rejected(self):
        model = FakeResolver(); original = model.forward_cached
        def broken(*args, **kwargs):
            values = original(*args, **kwargs)
            values['delta'][:, 2] = .01
            values['logits'] = values['anchor_logits']+values['delta']
            return values
        model.forward_cached = broken
        with self.assertRaises(RuntimeError): diagnostics.evaluate_cached_resolver(model, cache_fixture(), 4, 'cpu')

    def test_comparison_requires_all_three_controls_and_strict_gains(self):
        values = dict(pairwise=result([0, 0, 0, 1]), static_query=result([0, 0, 1, 1]),
            mlp=result([0, 1, 1, 1]), mean_tokens=result([0, 0, 1, 1]))
        anchor = result([0, 1, 1, 1])
        comparisons = diagnostics.control_comparisons(values, anchor, all_training_complete=True, full_best_is_trained=True)
        self.assertTrue(comparisons['full_beats_all_three_controls_on_selection'])
        self.assertTrue(comparisons['eligible_for_held_out_confirmation'])
        self.assertFalse(diagnostics.control_comparisons(values, values['pairwise'],
            all_training_complete=True, full_best_is_trained=True)['eligible_for_held_out_confirmation'])
        self.assertFalse(diagnostics.control_comparisons(values, anchor,
            all_training_complete=False, full_best_is_trained=True)['eligible_for_held_out_confirmation'])
        self.assertFalse(diagnostics.control_comparisons(values, anchor,
            all_training_complete=True, full_best_is_trained=False)['eligible_for_held_out_confirmation'])
        values['mlp'] = values['pairwise']
        self.assertFalse(diagnostics.control_comparisons(values)['full_beats_all_three_controls_on_selection'])
        del values['mlp']
        self.assertEqual(diagnostics.control_comparisons(values)['status'], 'incomplete')

    def test_cache_validation_rejects_duplicates_and_nonfinite_features(self):
        cache = cache_fixture(); cache['image_ids'][1] = cache['image_ids'][0]
        with self.assertRaises(ValueError): diagnostics.anchor_result(cache)
        cache = cache_fixture(); cache['features'][0, 0] = torch.nan
        with self.assertRaises(ValueError): diagnostics.anchor_result(cache)

    def test_report_runs_all_controls_and_preserves_development_only_status(self):
        import pairwise_resolver_experiment as experiment
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); training = root/'training'; training.mkdir()
            for name in ('selection_cache.pt', 'train_probe_cache.pt'): (training/name).write_bytes(b'fixture')
            for job in diagnostics.JOBS:
                directory = training/job; directory.mkdir()
                (directory/'best.pt').write_bytes(b'fixture')
                (directory/'summary.json').write_text(json.dumps(dict(status='complete', completed_epochs=30,
                    best_trained_selection=dict(accuracy=1/6), best_trained_epoch=1, best_trained_source='raw')))
            selection = cache_fixture(); probe = copy.deepcopy(selection); probe['identity']['split'] = 'train_probe'
            plan = dict(epochs=30, seed=42, config=dict(dataset=dict(batch_size=4)))
            with patch.object(experiment, 'load_feature_cache', side_effect=lambda p, path:probe if 'train_probe' in path.name else selection), \
                    patch.object(experiment, 'load_resolver_model', side_effect=lambda *a, **k:(FakeResolver(), dict(epoch=1, source='raw'))), \
                    patch('torch.cuda.is_available', return_value=False):
                report = diagnostics.run_pairwise_diagnostics(plan, training, root/'analysis')
            self.assertEqual(report['status'], 'complete')
            self.assertEqual(len(report['jobs']), 4)
            self.assertFalse(report['audit_partition_evaluated'])
            self.assertFalse(report['controls']['eligible_for_held_out_confirmation'])
            self.assertIn('global_mismatch', report['jobs']['pairwise']['interventions'])
            self.assertTrue((root/'analysis'/'report.md').is_file())

    def test_report_rejects_unknown_partition_even_if_it_is_not_named_audit(self):
        import pairwise_resolver_experiment as experiment
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); (root/'training').mkdir()
            cache = cache_fixture(); cache['identity']['split'] = 'unverified'
            with patch.object(experiment, 'load_feature_cache', return_value=cache):
                with self.assertRaises(ValueError):
                    diagnostics.run_pairwise_diagnostics(dict(), root/'training', root/'analysis')


if __name__ == '__main__': unittest.main()
