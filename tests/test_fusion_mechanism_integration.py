"""Verify the new control loop preserves the winning training operations."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

import fusion_mechanism_experiment as controls
import fusion_mechanism_diagnostics as evidence
import vit_augreg_experiment as original
from test_vit_augreg_diagnostics import Images, ToyCNN, ToyDual, ToyViT, NAMES


class SmallFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Flatten(), nn.Linear(48, 6),
                                     nn.BatchNorm1d(6), nn.Dropout(.15))
        self.vit = nn.Sequential(nn.Flatten(), nn.Linear(48, 4), nn.Dropout(.2))
        self.fusion_classifier = nn.Linear(10, 200)

    def forward(self, images):
        return self.fusion_classifier(torch.cat((self.encoder(images), self.vit(images)), 1))


class FusionMechanismIntegrationTests(unittest.TestCase):
    def test_control_epoch_matches_original_joint_optimizer_ema_and_rng(self):
        torch.manual_seed(991)
        baseline = SmallFusion()
        candidate = copy.deepcopy(baseline)
        averaged = copy.deepcopy(baseline).eval().requires_grad_(False)
        candidate_ema = {name: value.detach().clone()
                         for name, value in baseline.state_dict().items()}
        batches = [(torch.randn(size, 3, 4, 4), torch.arange(size),
                    [f'image-{size}-{index}' for index in range(size)])
                   for size in (4, 2)]
        config = dict(training=dict(learning_rate=5e-5, encoder_learning_rate=5e-5,
            vit_learning_rate=5e-5, fusion_learning_rate=5e-4, weight_decay=.05,
            gradient_clip_norm=5., accumulation_steps=1, mixup_alpha=.4,
            cutmix_alpha=0., mix_probability=1., label_smoothing=.1))
        first_optimizer = original.make_optimizer(baseline, config)
        second_optimizer = controls.make_optimizer(candidate, config)

        def fresh_batches():
            # timm Mixup modifies image tensors in place.
            return [(x.clone(), y.clone(), list(ids)) for x, y, ids in batches]

        torch.manual_seed(783)
        expected = original.train_one_epoch(baseline, averaged, fresh_batches(), config,
            42, 101, 'cpu', False, torch.float16, first_optimizer,
            torch.amp.GradScaler('cpu', enabled=False))
        expected_rng = torch.get_rng_state().clone()
        torch.manual_seed(783)
        actual = controls.train_one_epoch(candidate, fresh_batches(), config,
            42, 101, 'cpu', False, torch.float16, second_optimizer,
            torch.amp.GradScaler('cpu', enabled=False), candidate_ema)
        self.assertEqual(actual['optimizer_steps'], expected['optimizer_steps'])
        self.assertEqual(actual['skipped_amp_steps'], expected['skipped_amp_steps'])
        self.assertAlmostEqual(actual['train_soft_target_loss'],
                               expected['train_soft_target_loss'], places=7)
        self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))
        for name, value in baseline.state_dict().items():
            torch.testing.assert_close(candidate.state_dict()[name], value, rtol=0, atol=0)
            torch.testing.assert_close(candidate_ema[name], averaged.state_dict()[name],
                                       rtol=0, atol=0)
        first = first_optimizer.state_dict()
        second = second_optimizer.state_dict()
        self.assertEqual(first['param_groups'], second['param_groups'])
        for key, state in first['state'].items():
            for field, value in state.items():
                torch.testing.assert_close(second['state'][key][field], value, rtol=0, atol=0)

    def test_real_plan_training_diagnostics_and_completed_resume_share_schema(self):
        class Concat(ToyDual):
            variant = 'concat'

        directory = Path(__file__).resolve().parents[1]/'runs'/'fusion_mechanism_integration_tmp'
        directory.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=directory) as temporary:
            root = Path(temporary)
            for name in ('cnn', 'vit', 'reference'):
                (root/name).mkdir()
            marker = root/'cnn'/'immutable.json'
            marker.write_text('{}', encoding='utf-8')
            config = dict(dataset=dict(batch_size=64, num_workers=0, data_dir='data', split_seed=7),
                augmentation=dict(crop_scale=[.6, 1.], randaugment_ops=2, randaugment_magnitude=5),
                training=dict(epochs=100, seed=42, probe_per_class=20, mixed_precision=False,
                    learning_rate=5e-4, weight_decay=.05, scheduler={}, gradient_clip_norm=5.,
                    mixup_alpha=.2, cutmix_alpha=1., mix_probability=.5, label_smoothing=.1))
            sources = dict(cnn_suite=str(root/'cnn'), vit_suite=str(root/'vit'), config=config,
                seed=42, split_hash='same', parent_completed_epochs=213,
                provenance=dict(source_cnn_accuracy=.648, source_vit_accuracy=.5018,
                    parent_checkpoint_sha256='cnn', source_execution_environment={}),
                selected=dict(profile='vit_small_p8_augreg', checkpoint_sha256='vit', state={'model': {}}),
                reference=dict(record=dict(suite=str(root/'reference'), checkpoint_sha256='winner',
                    selection=dict(accuracy=.6545, nll=1.79), epoch=7, source='ema')),
                source_files={str(marker): controls.file_hash(marker)})
            torch.manual_seed(19)
            prototype, cnn, vit = Concat(), ToyCNN(), ToyViT()
            train, probe, selection = Images('train', 6), Images('train_probe', 6), Images('selection', 5)
            datasets = dict(train=train, train_probe=probe, selection=selection)
            training_manifest = dict(train=[train[i][2] for i in range(len(train))],
                                     selection=[selection[i][2] for i in range(len(selection))])
            clean_manifest = dict(train_clean=[probe[i][2] for i in range(len(probe))],
                                  selection=training_manifest['selection'])

            def runtime(config, seed):
                torch.manual_seed(seed)
                return 'cpu', False, torch.float32, torch.amp.GradScaler('cpu', enabled=False)

            def references(*args):
                return iter([('source_cnn', copy.deepcopy(cnn), dict(epoch=83, source='source_best')),
                    ('source_vit', copy.deepcopy(vit), dict(epoch=273, source='source_best')),
                    ('existing_joint', copy.deepcopy(prototype), dict(epoch=7, source='ema'))])

            args = controls.arguments().parse_args(['--output', str(root/'new'), '--epochs', '1'])
            with patch.object(controls, 'load_sources', return_value=sources), \
                    patch.object(controls, '_validate_data', return_value={}), \
                    patch.object(controls, 'source_hashes', return_value={'new': 'hash'}), \
                    patch.object(controls, 'runtime', side_effect=runtime), \
                    patch.object(controls, 'prepare_datasets', return_value=(datasets, training_manifest, 'same')), \
                    patch.object(controls, 'build_model', side_effect=lambda *a: copy.deepcopy(prototype)), \
                    patch.object(controls, 'mix_batch', side_effect=lambda x, y, *a:
                        (x, torch.nn.functional.one_hot(y, 3).float())), \
                    patch.object(evidence, 'clean_datasets', return_value=(probe, selection, None, 'same', clean_manifest)), \
                    patch.object(evidence, 'class_labels', return_value=NAMES), \
                    patch.object(evidence, '_source_models', side_effect=references), \
                    patch('torch.cuda.is_available', return_value=False):
                plan = controls.make_plan(args)
                suite = controls.create_suite(args, plan)
                first = controls.run_suite(suite)
                self.assertEqual(first['status'], 'complete', first)
                self.assertEqual(json.loads((suite/'training'/'suite.json').read_text()), plan)
                metrics = json.loads((suite/'feature_analysis'/'completion.json').read_text())
                self.assertEqual(metrics['status'], 'complete', metrics['failures'])
                self.assertEqual(len(metrics['jobs']), 15)
                self.assertTrue(metrics['probability_mixtures'])
                self.assertEqual(metrics['failures'], {})
                before = {str(path.relative_to(suite)): controls.file_hash(path)
                          for path in (suite/'training').rglob('*.pt')}
                with patch.object(controls, 'train_one_epoch', side_effect=AssertionError('Repeated training')), \
                        patch.object(evidence, 'collect_concat_features', side_effect=AssertionError('Repeated analysis')), \
                        patch.object(evidence, 'evaluate_plain', side_effect=AssertionError('Repeated references')):
                    second = controls.run_suite(suite)
                self.assertEqual(second['status'], 'complete', second)
                after = {str(path.relative_to(suite)): controls.file_hash(path)
                         for path in (suite/'training').rglob('*.pt')}
                self.assertEqual(after, before)


if __name__ == '__main__':
    unittest.main()
