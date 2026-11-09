import copy
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch

import fusion_mechanism_diagnostics as diagnostics
from fusion_continuation_diagnostics import probability_result
from models.attention_correction import state_digest
from test_vit_augreg_diagnostics import Images, ToyCNN, ToyDual, ToyViT, NAMES, config


ROOT = Path(__file__).resolve().parents[1]/'runs'/'fusion_mechanism_diagnostic_tests_tmp'


class Concat(ToyDual):
    variant = 'concat'
    readout_policy = 'CLS'


class FusionMechanismDiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_comparisons_are_paired_fixed_budget_and_never_audit_confirmation(self):
        def result(pred):
            probabilities = np.full((4, 3), .05, dtype=np.float32)
            probabilities[np.arange(4), pred] = .9
            return probability_result(probabilities, [0, 1, 2, 0], list('abcd'), NAMES)
        best = dict(source_cnn=result([0, 0, 2, 1]), existing_joint=result([0, 1, 2, 0]),
            random_vit_cls_linear=result([0, 0, 2, 0]),
            reset_cnn_head_cls_linear=result([0, 1, 1, 0]), patch_mean_linear=result([0, 1, 2, 0]))
        summaries = {job: dict(status='complete', completed_epochs=30) for job in diagnostics.JOBS}
        actual = diagnostics.interpret_comparisons(best, summaries, 30)
        self.assertTrue(actual['independently_trained_vit_advantage_on_selection'])
        self.assertTrue(actual['inherited_cnn_head_advantage_on_selection'])
        self.assertFalse(actual['patch_mean_advantage_on_selection'])
        self.assertEqual(actual['comparisons']['winner_vs_random_vit']['corrected'], 1)
        self.assertEqual(actual['comparisons']['winner_vs_random_vit']['harmed'], 0)
        self.assertTrue(actual['selection_is_exploratory'])
        self.assertFalse(actual['independent_confirmation'])
        summaries[diagnostics.JOBS[0]]['completed_epochs'] = 29
        partial = diagnostics.interpret_comparisons(best, summaries, 30)
        self.assertIsNone(partial['independently_trained_vit_advantage_on_selection'])
        self.assertIn('30-epoch', partial['scope'])

    def test_only_disjoint_training_probe_and_selection_can_be_evaluated(self):
        manifest = dict(train_clean=['a'], selection=['b'])
        diagnostics._allowed_data(Images(n=1), Images('selection', 1), 'x', 'x', manifest)
        with self.assertRaisesRegex(ValueError, 'Only clean'):
            diagnostics._allowed_data(Images(n=1), Images('audit', 1), 'x', 'x', manifest)
        with self.assertRaisesRegex(ValueError, 'overlap'):
            diagnostics._allowed_data(Images(n=1), Images('selection', 1), 'x', 'x',
                dict(train_clean=['a'], selection=['a']))
        with self.assertRaisesRegex(ValueError, 'partition'):
            diagnostics._allowed_data(Images(n=1), Images('selection', 1), 'x', 'y', manifest)

    def _fixture(self, action):
        ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            root = Path(temporary)
            training = root/'training'
            training.mkdir()
            for directory in ('cnn_source', 'vit_source', 'reference'):
                (root/directory).mkdir()
            marker = root/'cnn_source'/'marker.json'
            marker.write_text('{}', encoding='utf-8')
            cfg = config()
            plan = dict(cnn_suite=str(root/'cnn_source'), vit_suite=str(root/'vit_source'),
                reference=dict(suite=str(root/'reference'), checkpoint_sha256='winner'),
                config=cfg, seed=42, split_hash='same', source_hashes={'new': 'source'},
                source_provenance={'parent_checkpoint_sha256': 'cnn'},
                selected_vit={'checkpoint_sha256': 'vit'}, epochs=30,
                source_files={str(marker): diagnostics.file_hash(marker)})
            (training/'suite.json').write_text(json.dumps(plan), encoding='utf-8')
            torch.manual_seed(19)
            cnn, vit, dual = ToyCNN().eval(), ToyViT().eval(), Concat().eval()
            for job in diagnostics.JOBS:
                run = training/job
                run.mkdir()
                for checkpoint, epoch in [('best.pt', 0), ('best_trained.pt', 1), ('last.pt', 30)]:
                    torch.save(dict(model=dual.state_dict(), epoch=epoch,
                        source='raw', ema=dual.state_dict()), run/checkpoint)
                (run/'summary.json').write_text(json.dumps(dict(status='complete', completed_epochs=30,
                    best_epoch=0, total_train_seconds=83)), encoding='utf-8')
                (run/'metadata.json').write_text('{}', encoding='utf-8')
                (run/'epochs.csv').write_text('epoch,accuracy\n1,.5\n', encoding='utf-8')
            sources = dict(source_files=plan['source_files'], selected=dict(checkpoint_sha256='vit'))
            module = types.ModuleType('fusion_mechanism_experiment')
            module.load_sources = lambda plan: sources
            load_calls = []
            def load_model(plan, training, job, checkpoint='best.pt', sources=None):
                load_calls.append((job, checkpoint))
                state = torch.load(Path(training)/job/('last.pt' if checkpoint == 'last_ema' else checkpoint),
                    weights_only=False)
                model = copy.deepcopy(dual)
                model.load_state_dict(state['model'])
                if job == 'patch_mean_linear':
                    model.readout_policy = 'normalized patch mean'
                if checkpoint == 'last_ema':
                    state['source'] = 'ema'
                return model, state
            module.load_model = load_model
            probe, selection = Images(n=6), Images('selection', 5)
            manifest = dict(train_clean=[probe[i][2] for i in range(len(probe))],
                selection=[selection[i][2] for i in range(len(selection))])
            def source_models(*args):
                return iter([('source_cnn', copy.deepcopy(cnn), {'epoch': 83, 'source': 'source_best'}),
                    ('source_vit', copy.deepcopy(vit), {'epoch': 273, 'source': 'source_best'}),
                    ('existing_joint', copy.deepcopy(dual), {'epoch': 7, 'source': 'ema'})])
            with patch.dict(sys.modules, {'fusion_mechanism_experiment': module}), \
                    patch.object(diagnostics, 'clean_datasets', return_value=(probe, selection, None, 'same', manifest)), \
                    patch.object(diagnostics, 'class_labels', return_value=NAMES), \
                    patch.object(diagnostics, '_source_models', side_effect=source_models), \
                    patch('torch.cuda.is_available', return_value=False):
                action(root, plan, training, load_calls, dual)

    def test_complete_report_caches_every_checkpoint_and_preserves_branch_math(self):
        def action(root, plan, training, calls, dual):
            before = state_digest(dual)
            result = diagnostics.run_fusion_mechanism_diagnostics(plan, training, root/'analysis')
            self.assertEqual(result['status'], 'complete', result['failures'])
            self.assertEqual(len(result['jobs']), 15)
            self.assertEqual(len(calls), 12)
            self.assertEqual(state_digest(dual), before)
            self.assertEqual(len(result['probability_mixtures']), 16)
            self.assertTrue(all(row['corrected_vs_existing_joint'] is not None
                and row['harmed_vs_existing_joint'] is not None for row in result['probability_mixtures']))
            self.assertFalse(result['audit_partition_evaluated'])
            self.assertFalse(result['official_validation_evaluated'])
            record = result['jobs']['random_vit_cls_linear/best.pt']
            self.assertEqual(record['epoch'], 0)
            self.assertEqual(result['jobs']['random_vit_cls_linear/best_trained.pt']['epoch'], 1)
            self.assertEqual(result['jobs']['random_vit_cls_linear/last_ema']['source'], 'ema')
            for row in record['contributions']:
                if row['n']:
                    self.assertAlmostEqual(row['cnn_head_true_margin']+row['vit_true_margin'],
                        row['full_true_margin'], places=5)
            path = root/'analysis'/result['analysis_directory']/'random_vit_cls_linear'/'best'
            policy = json.loads((path/'selection_intervention_policy.json').read_text())
            self.assertEqual(policy['same_image_pairs'], 0)
            self.assertFalse(policy['pairing_uses_labels'])
            mean = json.loads((path/'training_mean_provenance.json').read_text())
            self.assertFalse(mean['uses_selection_images'])
            self.assertTrue((root/'analysis'/'report.md').is_file())
            calls.clear()
            with patch.object(diagnostics, 'collect_concat_features', side_effect=AssertionError('Recomputed cache')), \
                    patch.object(diagnostics, 'evaluate_plain', side_effect=AssertionError('Recomputed source')):
                repeated = diagnostics.run_fusion_mechanism_diagnostics(plan, training, root/'analysis')
            self.assertEqual(repeated['status'], 'complete', repeated['failures'])
            self.assertEqual(repeated['analysis_directory'], result['analysis_directory'])
            self.assertEqual(calls, [])
        self._fixture(action)

    def test_incomplete_owned_checkpoint_write_is_retained_and_rebuilt(self):
        def action(root, plan, training, calls, dual):
            first = diagnostics.run_fusion_mechanism_diagnostics(plan, training, root/'analysis')
            path = root/'analysis'/first['analysis_directory']/diagnostics.JOBS[0]/'best'
            (path/'record.json').unlink()
            calls.clear()
            repeated = diagnostics.run_fusion_mechanism_diagnostics(plan, training, root/'analysis')
            self.assertEqual(repeated['status'], 'complete', repeated['failures'])
            self.assertEqual(calls, [(diagnostics.JOBS[0], 'best.pt')])
            self.assertEqual(len(list(path.parent.glob('best__incomplete_*'))), 1)
        self._fixture(action)

    def test_modified_prediction_cache_is_rejected_and_originals_stay_unchanged(self):
        def action(root, plan, training, calls, dual):
            result = diagnostics.run_fusion_mechanism_diagnostics(plan, training, root/'analysis')
            path = root/'analysis'/result['analysis_directory']/diagnostics.JOBS[0]/'best'
            record = json.loads((path/'record.json').read_text())
            cache = path/'selection'/'learned'/'predictions.npz'
            cache.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'changed'):
                diagnostics._load_cache(path, record['identity'])
            old = diagnostics.file_hash(training/diagnostics.JOBS[0]/'best.pt')
            repeated = diagnostics.run_fusion_mechanism_diagnostics(plan, training, root/'analysis')
            self.assertEqual(repeated['status'], 'partial')
            self.assertIn(diagnostics.JOBS[0]+'/best.pt', repeated['failures'])
            self.assertEqual(diagnostics.file_hash(training/diagnostics.JOBS[0]/'best.pt'), old)
        self._fixture(action)

    def test_real_training_checkpoint_loader_and_diagnostics_complete_together(self):
        """Exercise the actual new engine/loader, not hand-authored checkpoints."""
        import fusion_mechanism_experiment as study
        from test_fusion_mechanism_experiment import FusionMechanismExperimentTests
        ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            root = Path(temporary)
            training = root/'training'; training.mkdir()
            for name in ('cnn_source', 'vit_source', 'reference'):
                (root/name).mkdir()
            plan = FusionMechanismExperimentTests().fixture(1)
            plan.update(cnn_suite=str(root/'cnn_source'), vit_suite=str(root/'vit_source'))
            plan['reference']['suite'] = str(root/'reference')
            plan['config']['dataset']['batch_size'] = 2
            plan['config_recipe_hash'] = study.recipe_hash(plan['config'])
            (training/'suite.json').write_text(json.dumps(plan), encoding='utf-8')
            torch.manual_seed(19)
            cnn, vit, dual = ToyCNN().eval(), ToyViT().eval(), Concat().eval()
            probe, selection, train = Images(n=6), Images('selection', 5), Images('train', 6)
            manifest = dict(train_clean=[probe[i][2] for i in range(len(probe))],
                selection=[selection[i][2] for i in range(len(selection))])
            sources = dict(source_files={}, selected=dict(checkpoint_sha256='vit'))
            def source_models(*args):
                return iter([('source_cnn', copy.deepcopy(cnn), {'epoch': 83, 'source': 'source_best'}),
                    ('source_vit', copy.deepcopy(vit), {'epoch': 273, 'source': 'source_best'}),
                    ('existing_joint', copy.deepcopy(dual), {'epoch': 7, 'source': 'ema'})])
            def cpu_runtime(cfg, seed):
                torch.manual_seed(seed)
                return 'cpu', False, torch.float32, torch.amp.GradScaler('cpu', enabled=False)
            with patch.object(study, 'build_model', side_effect=lambda *a: copy.deepcopy(dual)), \
                    patch.object(study, 'load_sources', return_value=sources), \
                    patch.object(study, 'runtime', side_effect=cpu_runtime), \
                    patch.object(study, 'mix_batch', side_effect=lambda x, y, *a:
                        (x, torch.nn.functional.one_hot(y, 3).float())), \
                    patch.object(diagnostics, 'clean_datasets', return_value=(probe, selection, None, 'split', manifest)), \
                    patch.object(diagnostics, 'class_labels', return_value=NAMES), \
                    patch.object(diagnostics, '_source_models', side_effect=source_models), \
                    patch('torch.cuda.is_available', return_value=False):
                for job in study.JOBS:
                    trained = study.train_job(plan, job, training, sources,
                        dict(train=train, selection=selection, train_probe=probe))
                    self.assertEqual(trained['status'], 'complete')
                completed = diagnostics.run_fusion_mechanism_diagnostics(plan, training, root/'analysis')
            self.assertEqual(completed['status'], 'complete', completed['failures'])
            self.assertEqual(len(completed['jobs']), 15)
            self.assertTrue(completed['comparisons']['all_training_horizons_complete'])
            self.assertTrue(any(name.endswith('epochs.csv') for name in completed['identity']['checkpoints']))


if __name__ == '__main__':
    unittest.main()
