import copy
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
import torch
from torch import nn
from torchvision.datasets import ImageFolder

import baseline_diagnostics as runner
from datasets.utility_tiny import UtilityDataset


class ForbiddenEvaluation:
    """The final/audit source must never be read by this diagnosis."""
    def __init__(self):
        self.reads = 0

    def __getitem__(self, index):
        self.reads += 1
        raise AssertionError('Audit/final images were accessed')


def toy_model():
    model = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 3))
    with torch.no_grad():
        model[-1].weight.copy_(torch.eye(3))
        model[-1].bias.zero_()
    return model


def config():
    return dict(dataset=dict(data_dir='data', download=True, split_seed=42,
                             batch_size=64, num_workers=0),
                augmentation=dict(crop_scale=[.6, 1.], randaugment_ops=2,
                                  randaugment_magnitude=5),
                training=dict(seed=42, epochs=100, enable_tf32=False,
                              early_stopping_patience=10, learning_rate=.0005,
                              weight_decay=.05, label_smoothing=.1,
                              mix_probability=.5, mixup_alpha=.2, cutmix_alpha=1.,
                              scheduler=dict(name='cosine', warmup_epochs=2,
                                             min_lr=.00001)))


def write_parent(root):
    root.mkdir(parents=True, exist_ok=True)
    selection = dict(accuracy=2/3, top5_accuracy=1., nll=.75)
    metadata = dict(variant='resnet50_preact', protocol=runner.PROTOCOL,
                    seed=42, split_hash='fixture-split', recipe_hash=runner.recipe_hash(config()),
                    source_hashes={name: runner.file_hash(runner.ROOT/name)
                                   for name in runner.MODEL_SOURCES})
    summary = dict(metadata, completed_epochs=100, best_epoch=94,
                   best_selection=selection)
    state = dict(metadata=copy.deepcopy(metadata), epoch=94,
                 selection=copy.deepcopy(selection), config=config(),
                 model=toy_model().state_dict())
    (root/'summary.json').write_text(json.dumps(summary), encoding='utf-8')
    torch.save(state, root/'best.pt')
    return summary, state


def fixture_datasets(root):
    dataset_root = root/'tiny'
    train_root = dataset_root/'train'
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    for label in range(3):
        directory = train_root/f'n{label:03d}'/'images'
        directory.mkdir(parents=True)
        for i in range(4):
            # One deliberately difficult selection image: red, labeled green.
            color = colors[0] if (label, i) == (1, 2) else colors[label]
            Image.new('RGB', (16, 16), color).save(directory/f'{i}.png')
    (dataset_root/'words.txt').write_text('n000\tred object\nn001\tgreen object\n',
                                          encoding='utf-8')
    folder = ImageFolder(str(train_root))
    train = [0, 1, 4, 5, 8, 9]
    selection = [2, 6, 10]
    audit = ForbiddenEvaluation()
    datasets = dict(train=UtilityDataset(folder, train, 'train', 42),
                    selection=UtilityDataset(folder, selection, 'selection', 42),
                    audit=audit, official_validation=ForbiddenEvaluation())
    manifest = {name: [Path(folder.samples[i][0]).relative_to(folder.root).as_posix()
                       for i in indices]
                for name, indices in [('train', train), ('selection', selection)]}
    manifest['audit'] = ['must-not-be-read']
    return datasets, manifest, 'fixture-split'


class BaselineDiagnosticCliTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_parent_identity_and_mismatch_rejection(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)/'parent'
            summary, state = write_parent(run)
            loaded_run, loaded_summary, loaded_state, digest = runner.load_run(run)
            self.assertEqual(loaded_run, run.resolve())
            self.assertEqual(loaded_summary, summary)
            self.assertEqual(loaded_state['epoch'], 94)
            self.assertEqual(digest, runner.file_hash(run/'best.pt'))

            changed = copy.deepcopy(summary)
            changed['seed'] = 43
            (run/'summary.json').write_text(json.dumps(changed), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'mismatch: seed'):
                runner.load_run(run)

            changed = copy.deepcopy(summary)
            changed['best_epoch'] = 99
            (run/'summary.json').write_text(json.dumps(changed), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'best epoch mismatch'):
                runner.load_run(run)

            changed = copy.deepcopy(summary)
            changed['completed_epochs'] = 80
            (run/'summary.json').write_text(json.dumps(changed), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'chronology'):
                runner.load_run(run)

            (run/'summary.json').write_text(json.dumps(summary), encoding='utf-8')
            tampered = copy.deepcopy(state)
            tampered['config']['training']['learning_rate'] = .9
            torch.save(tampered, run/'best.pt')
            with self.assertRaisesRegex(ValueError, 'recipe hash'):
                runner.load_run(run)

            changed = copy.deepcopy(summary)
            changed['source_hashes'][runner.MODEL_SOURCES[0]] = 'different-source'
            state['metadata']['source_hashes'] = copy.deepcopy(changed['source_hashes'])
            (run/'summary.json').write_text(json.dumps(changed), encoding='utf-8')
            torch.save(state, run/'best.pt')
            with self.assertRaisesRegex(ValueError, 'implementation differs'):
                runner.load_run(run)

    def test_clean_views_and_nested_fixed_probe(self):
        with tempfile.TemporaryDirectory() as temp:
            supplied = fixture_datasets(Path(temp))
            with patch.object(runner, 'make_datasets', return_value=supplied):
                full, selection, classes, fingerprint, manifest = runner.clean_datasets(
                    config(), 42, 'fixture-split', 0)
                probe, _, _, _, probe_manifest = runner.clean_datasets(
                    config(), 42, 'fixture-split', 1)
                repeated, _, _, _, _ = runner.clean_datasets(
                    config(), 99, 'fixture-split', 1)
                self.assertEqual(full.split, 'train_clean')
                self.assertEqual(supplied[0]['train'].split, 'train')
                self.assertEqual(full.indices, [0, 1, 4, 5, 8, 9])
                self.assertEqual(len(probe), 3)
                self.assertEqual(probe.indices, repeated.indices)
                self.assertTrue(set(probe_manifest['train_clean']) <= set(manifest['train_clean']))
                self.assertEqual(selection.split, 'selection')
                self.assertEqual(classes, ['n000', 'n001', 'n002'])
                self.assertEqual(fingerprint, 'fixture-split')
                self.assertEqual(set(manifest), {'train_clean', 'selection'})
                with patch('datasets.utility_tiny.TF.pad', side_effect=AssertionError('Augmentation')):
                    first = full[0]
                    second = full[0]
                    torch.testing.assert_close(first[0], second[0])
                self.assertEqual(supplied[0]['audit'].reads, 0)
                self.assertEqual(supplied[0]['official_validation'].reads, 0)
                with self.assertRaisesRegex(ValueError, 'split/class mapping'):
                    runner.clean_datasets(config(), 42, 'wrong-split', 0)

    def test_evaluation_options_do_not_mutate_archived_config(self):
        original = config()
        state = dict(config=original)
        args = runner.arguments().parse_args(['evaluate', '--device', 'cpu',
                    '--batch-size', '16', '--num-workers', '2'])
        evaluated = runner.evaluation_config(state, args)
        self.assertFalse(evaluated['dataset']['download'])
        self.assertEqual(evaluated['dataset']['batch_size'], 16)
        self.assertEqual(evaluated['dataset']['num_workers'], 2)
        self.assertTrue(Path(evaluated['dataset']['data_dir']).is_absolute())
        self.assertEqual(original, config())
        args.batch_size = 0
        with self.assertRaisesRegex(ValueError, 'batch_size'):
            runner.evaluation_config(state, args)

    def test_training_controls_change_only_declared_factors(self):
        original = config()
        expected = copy.deepcopy(original)
        expected['training'].update(epochs=200, early_stopping_patience=0)
        self.assertEqual(runner.control_config(original, 'longer200'), expected)
        expected = copy.deepcopy(original)
        expected['training'].update(epochs=100, early_stopping_patience=0,
                                    mix_probability=0.)
        self.assertEqual(runner.control_config(original, 'no_mix100'), expected)
        self.assertEqual(original, config())
        with self.assertRaisesRegex(ValueError, 'Unknown control'):
            runner.control_config(original, 'arbitrary')

    def test_controls_preflight_then_fresh_runs_and_clean_evaluation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = root/'parent'
            write_parent(run)
            supplied = fixture_datasets(root)
            args = runner.arguments().parse_args(['controls', '--base-run', str(run),
                '--variants', 'longer200', 'no_mix100', '--output', str(root/'controls')])
            with patch.object(runner.torch.cuda, 'is_available', return_value=True), \
                    patch.object(runner, 'make_datasets', return_value=supplied) as datasets, \
                    patch.object(runner, 'train_run', return_value=True) as train, \
                    patch.object(runner, 'evaluate_run') as evaluate:
                suite = runner.run_controls(args)
            datasets.assert_called_once()
            self.assertEqual(train.call_count, 2)
            self.assertEqual(train.call_args_list[0].args[0]['training']['epochs'], 200)
            self.assertEqual(train.call_args_list[1].args[0]['training']['mix_probability'], 0.)
            for call in train.call_args_list:
                self.assertEqual(call.args[1:3], ('resnet50_preact', 42))
                self.assertEqual(call.kwargs, {})  # No resume or parent-weight factory.
                self.assertEqual(call.args[0]['training']['early_stopping_patience'], 0)
            self.assertEqual(evaluate.call_count, 2)
            self.assertEqual(evaluate.call_args_list[0].args[0].base_run, str(suite/'longer200'))
            self.assertTrue(json.loads((suite/'plan.json').read_text())['fresh_training'])
            self.assertEqual(supplied[0]['audit'].reads, 0)

    def test_controls_reject_wrong_split_before_training(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = root/'parent'
            write_parent(run)
            supplied = fixture_datasets(root)
            args = runner.arguments().parse_args(['controls', '--base-run', str(run),
                '--variants', 'longer200', '--output', str(root/'controls')])
            with patch.object(runner.torch.cuda, 'is_available', return_value=True), \
                    patch.object(runner, 'make_datasets', return_value=(*supplied[:2], 'wrong-split')), \
                    patch.object(runner, 'train_run') as train:
                with self.assertRaisesRegex(ValueError, 'split/class mapping'):
                    runner.run_controls(args)
            train.assert_not_called()
            self.assertFalse((root/'controls').exists())

    def test_end_to_end_clean_evaluation_and_portable_bundle(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = root/'parent'
            write_parent(run)
            supplied = fixture_datasets(root)
            before = runner.file_hash(run/'best.pt')
            model = toy_model()
            weights = {name: value.clone() for name, value in model.state_dict().items()}
            args = runner.arguments().parse_args(['evaluate', '--base-run', str(run),
                    '--device', 'cpu', '--output', str(root/'diagnostics'), '--examples', '0'])
            with patch.object(runner, 'make_datasets', return_value=supplied), \
                    patch.object(runner, 'build_model', return_value=model) as build:
                output = runner.evaluate_run(args)
            build.assert_called_once_with('resnet50_preact', 42)
            report = json.loads((output/'summary.json').read_text(encoding='utf-8'))
            self.assertEqual(report['status'], 'complete')
            self.assertEqual(report['checkpoint_epoch'], 94)
            self.assertEqual(report['scores']['train_clean']['n'], 6)
            self.assertEqual(report['scores']['selection']['n'], 3)
            self.assertAlmostEqual(report['scores']['train_clean']['accuracy'], 1.)
            self.assertAlmostEqual(report['scores']['selection']['accuracy'], 2/3)
            self.assertAlmostEqual(report['generalization_gap']['accuracy_pp'], 100/3)
            self.assertAlmostEqual(report['selection_replay']['accuracy_delta_pp'], 0.)
            self.assertFalse(report['evaluation']['official_validation_evaluated'])
            self.assertFalse(report['evaluation']['audit_partition_evaluated'])
            self.assertTrue(report['model_state_unchanged'])
            self.assertEqual(runner.file_hash(run/'best.pt'), before)
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, weights[name])
            self.assertEqual(supplied[0]['audit'].reads, 0)
            self.assertEqual(supplied[0]['official_validation'].reads, 0)
            manifest = json.loads((output/'evaluation_manifest.json').read_text())
            self.assertEqual(set(manifest), {'train_clean', 'selection'})
            self.assertTrue((output/'report.md').exists())
            self.assertTrue((output/'train_clean/predictions.npz').exists())
            with tarfile.open(output/'diagnostics_report.tar.gz') as archive:
                names = archive.getnames()
                self.assertIn('summary.json', names)
                self.assertIn('selection/per_class.csv', names)
                self.assertFalse(any(name.endswith('.npz') for name in names))

    def test_output_inside_original_training_run_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = root/'parent'
            write_parent(run)
            supplied = fixture_datasets(root)
            args = runner.arguments().parse_args(['evaluate', '--base-run', str(run),
                    '--device', 'cpu', '--output', str(run/'diagnosis'), '--examples', '0'])
            with patch.object(runner, 'make_datasets', return_value=supplied), \
                    patch.object(runner, 'build_model', return_value=toy_model()), \
                    patch.object(runner, 'evaluate_clean') as evaluate:
                with self.assertRaisesRegex(ValueError, 'outside the original'):
                    runner.evaluate_run(args)
                evaluate.assert_not_called()


if __name__ == '__main__':
    unittest.main()
