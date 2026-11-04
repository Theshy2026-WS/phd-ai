import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

import mechanism_diagnostics as diagnostics


class SmallImages(Dataset):
    split = 'train_clean'

    def __init__(self, count=8, prefix='train'):
        self.count, self.prefix = count, prefix

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        image = torch.rand((3, 8, 8), generator=torch.Generator().manual_seed(20+index))+.7
        return image, index % 3, f'{self.prefix}/image{index}.png'


class SmallSpatial(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 4, 1), nn.BatchNorm2d(4), nn.ReLU(), nn.Dropout(.8))
        self.stages = nn.Identity()
        self.head = nn.Sequential(nn.BatchNorm2d(4), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
                                  nn.Flatten(), nn.Linear(4, 3))

    def forward(self, images):
        features = self.stages(self.stem(images))
        return self.head(features[..., ::2, ::2])


class ReadoutTests(unittest.TestCase):
    def test_dense_logits_equal_mean_phase_logits_with_nontrivial_bn(self):
        model = SmallSpatial().eval()
        with torch.no_grad():
            model.head[0].running_mean.copy_(torch.tensor([.1, -.2, .4, .8]))
            model.head[0].running_var.copy_(torch.tensor([.3, .8, 1.4, 2.]))
            model.head[0].bias.copy_(torch.tensor([.5, .1, -.1, .2]))
        features = torch.randn(3, 4, 8, 8)
        values, error = diagnostics.readout_logits(model, features)
        self.assertEqual(tuple(values), diagnostics.READOUTS)
        torch.testing.assert_close(values['full8'], torch.stack(list(values.values())[:4]).mean(0))
        self.assertLess(error, 1e-5)
        self.assertFalse(torch.equal(values['phase00'], values['phase11']))

    def test_readout_rejects_train_mode_or_wrong_feature_size(self):
        model = SmallSpatial()
        with self.assertRaisesRegex(ValueError, 'evaluation-mode'):
            diagnostics.readout_logits(model, torch.randn(2, 4, 8, 8))
        model.eval()
        with self.assertRaisesRegex(ValueError, '8x8'):
            diagnostics.readout_logits(model, torch.randn(2, 4, 4, 4))

    def test_head_precision_is_strict_and_restored_even_on_failure(self):
        model = SmallSpatial().eval()
        previous = torch.get_float32_matmul_precision()
        observed = []
        handle = model.head.register_forward_pre_hook(
            lambda *args: observed.append(torch.get_float32_matmul_precision()))
        try:
            torch.set_float32_matmul_precision('high')
            diagnostics.readout_logits(model, torch.randn(2, 4, 8, 8))
            self.assertEqual(observed, ['highest']*5)
            self.assertEqual(torch.get_float32_matmul_precision(), 'high')
            with patch.object(model.head, 'forward', side_effect=RuntimeError('head failed')):
                with self.assertRaisesRegex(RuntimeError, 'head failed'):
                    diagnostics.readout_logits(model, torch.randn(2, 4, 8, 8))
            self.assertEqual(torch.get_float32_matmul_precision(), 'high')
        finally:
            handle.remove()
            torch.set_float32_matmul_precision(previous)

    def test_one_encoder_pass_exports_all_readouts_and_exact_counts(self):
        model, dataset = SmallSpatial().eval(), SmallImages()
        calls = []
        handle = model.stem.register_forward_hook(lambda *args: calls.append(1))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'readouts'
            result = diagnostics.evaluate_readouts(model, dataset, output_dir=output,
                class_names=['a', 'b', 'c'], batch_size=4, num_workers=0, device='cpu')
            self.assertEqual(len(calls), 2)
            self.assertEqual(result['image_ids'], [dataset[i][2] for i in range(8)])
            for name in diagnostics.READOUTS:
                self.assertEqual(result['records'][name]['metrics']['n'], 8)
                self.assertEqual(int(result['records'][name]['confusion'].sum()), 8)
                with np.load(output/name/'predictions.npz', allow_pickle=False) as arrays:
                    np.testing.assert_array_equal(arrays['targets'], [i % 3 for i in range(8)])
                self.assertTrue((output/name/'per_class.csv').is_file())
            self.assertLess(result['sensitivity']['mean_phase_full8_max_logit_error'], 1e-5)
            with self.assertRaisesRegex(ValueError, 'overwrite'):
                diagnostics.evaluate_readouts(model, dataset, output_dir=output,
                    class_names=['a', 'b', 'c'], batch_size=4, num_workers=0, device='cpu')
        handle.remove()


class CalibrationTests(unittest.TestCase):
    def test_calibration_changes_only_copy_bn_buffers_and_ignores_labels(self):
        model, dataset = SmallSpatial().eval(), SmallImages()
        before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
        dropout_modes = []
        # Deepcopy also copies the hook, allowing the calibration copy's mode to be observed.
        hook = model.stem[3].register_forward_pre_hook(lambda module, inputs: dropout_modes.append(module.training))
        allowed = [dataset[i][2] for i in range(len(dataset))]
        calibrated, meta = diagnostics.recalibrate_bn(model, dataset, allowed_ids=allowed,
            batch_size=4, num_workers=0, device='cpu')
        hook.remove()
        self.assertEqual(meta['training_images'], 8)
        self.assertEqual(meta['batches'], 2)
        self.assertFalse(meta['labels_used'])
        self.assertFalse(meta['optimizer_used'])
        self.assertTrue(meta['changed_buffers'])
        self.assertEqual(set(meta['calibration_ids']), set(allowed))
        self.assertTrue(all(not value for value in dropout_modes))
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]), name)
        for name, parameter in calibrated.named_parameters():
            self.assertTrue(torch.equal(parameter, dict(model.named_parameters())[name]), name)
        for module in calibrated.modules():
            self.assertFalse(module.training)
        self.assertEqual(calibrated.stem[1].momentum, model.stem[1].momentum)
        self.assertEqual(int(calibrated.stem[1].num_batches_tracked), 2)

    def test_calibration_authorizes_training_ids_and_rejects_selection(self):
        train = SmallImages()
        selection = SmallImages(prefix='selection')
        with self.assertRaisesRegex(ValueError, 'unauthorized'):
            diagnostics.recalibrate_bn(SmallSpatial().eval(), selection,
                allowed_ids=[train[i][2] for i in range(len(train))], batch_size=4,
                num_workers=0, device='cpu')

    def test_calibration_rejects_augmented_view_and_unequal_batches(self):
        dataset = SmallImages()
        ids = [dataset[i][2] for i in range(len(dataset))]
        dataset.split = 'train'
        with self.assertRaisesRegex(ValueError, 'without augmentation'):
            diagnostics.recalibrate_bn(SmallSpatial().eval(), dataset, allowed_ids=ids,
                batch_size=4, num_workers=0, device='cpu')
        dataset.split = 'train_clean'
        with self.assertRaisesRegex(ValueError, 'equal batches'):
            diagnostics.recalibrate_bn(SmallSpatial().eval(), dataset, allowed_ids=ids,
                batch_size=3, num_workers=0, device='cpu')

    def test_calibration_rejects_selection_even_when_caller_supplies_its_ids(self):
        dataset = SmallImages()
        dataset.split = 'selection'
        with self.assertRaisesRegex(ValueError, 'clean training'):
            diagnostics.recalibrate_bn(SmallSpatial().eval(), dataset,
                allowed_ids=[dataset[i][2] for i in range(len(dataset))],
                batch_size=4, num_workers=0, device='cpu')

    def test_runtime_relocation_cannot_change_training_recipe(self):
        original = dict(dataset=dict(data_dir='old', num_workers=4, batch_size=64,
                                    download=False, split_seed=42),
                        training=dict(epochs=100, learning_rate=.0005))
        relocated = copy.deepcopy(original)
        relocated['dataset'].update(data_dir='new', num_workers=0)
        diagnostics._relocation_only(original, relocated)
        relocated['training']['epochs'] = 30
        with self.assertRaisesRegex(ValueError, 'relocation'):
            diagnostics._relocation_only(original, relocated)


class ParentIdentityTests(unittest.TestCase):
    def _parent(self, root):
        run = root/'run'
        run.mkdir()
        config = dict(dataset=dict(data_dir='data', split_seed=42, batch_size=64, num_workers=0),
                      training=dict(epochs=100, learning_rate=.0005))
        metadata = dict(protocol=diagnostics.SPATIAL_PROTOCOL, variant='cnn8', seed=42,
            split_hash='split', source_hashes={name: f'hash-{name}' for name in diagnostics.MODEL_SOURCES},
            recipe_hash=diagnostics.recipe_hash(config), initial_state_sha256='initial',
            environment=dict(device='cpu'), params=1)
        selection = dict(accuracy=.64, nll=1.72, top5_accuracy=.83)
        summary = dict(**metadata, best_epoch=100, completed_epochs=100, best_selection=selection)
        (run/'summary.json').write_text(json.dumps(summary), encoding='utf-8')
        torch.save(dict(metadata=metadata, config=config, epoch=100, selection=selection,
                        model=dict(weight=torch.ones(1))), run/'best.pt')
        return run

    def test_parent_accepts_new_tooling_but_rejects_changed_model_source(self):
        with tempfile.TemporaryDirectory() as directory:
            run = self._parent(Path(directory))

            def source_hash(path):
                path = Path(path)
                if path.name == 'best.pt':
                    return 'checkpoint'
                return f'hash-{path.relative_to(diagnostics.ROOT).as_posix()}'

            with patch.object(diagnostics, 'file_hash', side_effect=source_hash):
                _, summary, state, digest = diagnostics.load_spatial_parent(run)
                self.assertEqual(summary['best_epoch'], 100)
                self.assertEqual(state['epoch'], 100)
                self.assertEqual(digest, 'checkpoint')
            with patch.object(diagnostics, 'file_hash', return_value='changed'):
                with self.assertRaisesRegex(ValueError, 'implementation changed'):
                    diagnostics.load_spatial_parent(run)

    def test_parent_rejects_recipe_drift_and_non_cnn8_run(self):
        with tempfile.TemporaryDirectory() as directory:
            run = self._parent(Path(directory))
            state = torch.load(run/'best.pt', weights_only=False)
            state['config']['training']['learning_rate'] = .005
            torch.save(state, run/'best.pt')
            with self.assertRaisesRegex(ValueError, 'recipe hash mismatch'):
                diagnostics.load_spatial_parent(run)
            summary = json.loads((run/'summary.json').read_text(encoding='utf-8'))
            summary['variant'] = 'cnn4'
            (run/'summary.json').write_text(json.dumps(summary), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, '100-epoch spatial cnn8'):
                diagnostics.load_spatial_parent(run)


class FullDiagnosticTests(unittest.TestCase):
    def test_complete_api_two_bn_states_five_readouts_and_strict_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent = root/'parent'
            parent.mkdir()
            (parent/'best.pt').write_bytes(b'immutable parent checkpoint fixture')
            model = SmallSpatial().eval()
            config = dict(dataset=dict(data_dir='fixture', split_seed=42, batch_size=4,
                                       num_workers=0, download=False), training=dict(epochs=100))
            summary = dict(seed=42, split_hash='fixture-split',
                           best_selection=dict(accuracy=.64, nll=1.72, top5_accuracy=.83))
            state = dict(config=config, model=model.state_dict())
            digest = diagnostics.file_hash(parent/'best.pt')
            train, selection = SmallImages(8, 'train'), SmallImages(4, 'selection')
            selection.split = 'selection'
            manifest = dict(train_clean=[train[i][2] for i in range(8)],
                            selection=[selection[i][2] for i in range(4)])
            output = root/'mechanism'
            with patch.object(diagnostics, 'TRAIN_IMAGES', 8), \
                 patch.object(diagnostics, 'SELECTION_IMAGES', 4), \
                 patch.object(diagnostics, 'load_spatial_parent', return_value=(parent, summary, state, digest)), \
                 patch.object(diagnostics, 'build_spatial_model', side_effect=lambda *args: copy.deepcopy(model)), \
                 patch.object(diagnostics, 'clean_datasets', return_value=(train, selection, ['a','b','c'], 'fixture-split', manifest)), \
                 patch.object(diagnostics, 'class_labels', return_value=['a','b','c']), \
                 patch.object(diagnostics.torch.cuda, 'is_available', return_value=False):
                result = diagnostics.run_diagnostics(parent, output, examples=0)
                self.assertEqual(result['status'], 'complete')
                self.assertEqual(set(result['scores']), {'original', 'recalibrated'})
                for label in result['scores']:
                    self.assertEqual(tuple(result['scores'][label]), diagnostics.READOUTS)
                    for scores in result['scores'][label].values():
                        self.assertEqual(scores['train_clean']['n'], 8)
                        self.assertEqual(scores['selection']['n'], 4)
                self.assertTrue((output/'report.md').is_file())
                self.assertTrue((output/result['analysis_directory']/'comparison.csv').is_file())
                again = diagnostics.run_diagnostics(parent, output, examples=0)
                self.assertEqual(result, again)
                self.assertEqual(len(list(output.glob('attempt_*'))), 1)
                self.assertEqual(diagnostics.file_hash(parent/'best.pt'), digest)
                self.assertFalse(result['audit_partition_evaluated'])
                self.assertFalse(result['official_validation_evaluated'])
                changed = copy.deepcopy(config)
                changed['dataset']['num_workers'] = 1
                with self.assertRaisesRegex(ValueError, 'does not match'):
                    diagnostics.run_diagnostics(parent, output, config=changed, examples=0)


if __name__ == '__main__':
    unittest.main()
