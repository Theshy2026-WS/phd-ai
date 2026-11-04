import copy
import csv
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch import nn

import generalization_audit as audit
from datasets.resnet_tiny import AugmentedTrainingDataset
from datasets.utility_tiny import UtilityDataset


AUGMENTATION = dict(crop_scale=[.6, 1.], randaugment_ops=2, randaugment_magnitude=5)


def supplied(root):
    data = Path(root)/'images'
    samples, targets = [], []
    for label, name in enumerate(('class_a', 'class_b')):
        directory = data/name
        directory.mkdir(parents=True)
        for i in range(3):
            path = directory/f'image{i}.png'
            pixels = np.random.default_rng(label*30+i).integers(0, 256, (64, 64, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(path)
            samples.append((str(path), label)); targets.append(label)

    def image_loader(path):
        with Image.open(path) as image:
            return image.convert('RGB')

    folder = SimpleNamespace(root=str(data), samples=samples, targets=targets,
        classes=['class_a', 'class_b'], class_to_idx={'class_a': 0, 'class_b': 1}, loader=image_loader)
    train = UtilityDataset(folder, [0, 1, 3, 4], 'train_clean', 42)
    selection = UtilityDataset(folder, [2, 5], 'selection', 42)
    probe = UtilityDataset(folder, [0, 3], 'train_clean', 42)
    return train, selection, probe


def model():
    return nn.Sequential(nn.Conv2d(3, 4, 1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
                         nn.Flatten(), nn.Linear(4, 2))


class SourceAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_shared_source_balance_and_read_only_inventory(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            train, selection, _ = supplied(root)
            output = root/'audit'; output.mkdir()
            before = {path: Path(path).read_bytes() for path, _ in train.folder.samples}
            result = audit.scan_sources(train, selection, output)
            self.assertTrue(result['shared_class_mapping'])
            self.assertTrue(result['ids_disjoint'])
            self.assertTrue(result['same_image_pipeline_replay']['all_equal'])
            self.assertEqual(result['same_image_pipeline_replay']['max_absolute_tensor_difference'], 0.)
            self.assertEqual(result['counts'], {'train_clean': 4, 'selection': 2})
            self.assertEqual(result['class_counts']['train_clean'], {0: 2, 1: 2})
            self.assertTrue((output/'distribution_effect_sizes.csv').is_file())
            for path, contents in before.items():
                self.assertEqual(Path(path).read_bytes(), contents)
            with (output/'image_inventory.csv').open(newline='') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 6)
            self.assertTrue(all(len(row['rgb_pixel_sha256']) == 64 for row in rows))

    def test_exact_content_duplicates_are_reported_not_removed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            train, selection, _ = supplied(root)
            original = Path(train.folder.samples[0][0])
            target = Path(selection.folder.samples[5][0])
            target.write_bytes(original.read_bytes())
            output = root/'audit'; output.mkdir()
            result = audit.scan_sources(train, selection, output)
            self.assertEqual(result['duplicate_counts']['file_sha256']['cross_split_groups'], 1)
            self.assertEqual(result['duplicate_counts']['rgb_pixel_sha256']['selection_images'], 1)
            with (output/'hash_duplicates.csv').open(newline='') as stream:
                rows = list(csv.DictReader(stream))
            self.assertTrue(any(row['kind'] == 'file_sha256' and row['label_conflict'] == 'True' for row in rows))
            self.assertEqual(target.read_bytes(), original.read_bytes())

    def test_mapping_mismatch_and_corrupt_dimensions_are_not_hidden(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            train, selection, _ = supplied(root)
            output = root/'audit'; output.mkdir()
            train.folder.targets[0] = 1
            Image.new('RGB', (32, 32)).save(train.folder.samples[1][0])
            with self.assertRaisesRegex(ValueError, 'provenance errors'):
                audit.scan_sources(train, selection, output)
            status = json.loads((output/'source_scan.json').read_text())
            self.assertEqual(status['failures'], 2)
            self.assertTrue((output/'source_failures.csv').is_file())

    def test_coarse_hash_is_explicitly_not_exact_duplicate_proof(self):
        self.assertEqual(audit.average_hash(Image.new('RGB', (64, 64), 'black')),
                         audit.average_hash(Image.new('RGB', (64, 64), 'white')))
        # Flat images collide in average-hash space even with different pixels.
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            train, selection, _ = supplied(root)
            Image.new('RGB', (64, 64), 'black').save(train.folder.samples[0][0])
            Image.new('RGB', (64, 64), 'white').save(selection.folder.samples[2][0])
            output = root/'audit'; output.mkdir()
            result = audit.scan_sources(train, selection, output)
            candidate = result['duplicate_counts']['average_hash64']
            self.assertGreaterEqual(candidate['cross_split_groups'], 1)
            self.assertFalse(candidate['exact_duplicate_proof'])
            self.assertEqual(result['duplicate_counts']['rgb_pixel_sha256']['pair_count'], 0)

    def test_same_image_replay_detects_normalization_disagreement(self):
        class ChangedNormalization(UtilityDataset):
            def __getitem__(self, position):
                tensor, target, identifier = super().__getitem__(position)
                return tensor+.1, target, identifier

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            train, selection, _ = supplied(root)
            wrong = ChangedNormalization(train.folder, train.indices, 'train_clean', train.seed)
            replay = audit.clean_pipeline_replay(wrong)
            self.assertFalse(replay['all_equal'])
            self.assertFalse(replay['same_image_tensor_equal'])
            self.assertTrue(replay['same_target_equal'])
            self.assertTrue(replay['same_image_id_equal'])
            output = root/'audit'; output.mkdir()
            with self.assertRaisesRegex(ValueError, 'Same-image'):
                audit.scan_sources(wrong, selection, output)
            self.assertEqual(json.loads((output/'source_scan.json').read_text())['status'], 'issues_found')


class ViewTests(unittest.TestCase):
    def test_full_augmentation_matches_existing_recipe_and_preserves_rng(self):
        with tempfile.TemporaryDirectory() as temp:
            train, _, _ = supplied(temp)
            original = AugmentedTrainingDataset(train, AUGMENTATION)
            original.epoch = 101
            audited = audit.AuditView(train, 'inherited_aug', AUGMENTATION, epoch=101)
            torch.manual_seed(938)
            expected = torch.rand(5)
            torch.manual_seed(938)
            for i in range(len(train)):
                left, y_left, id_left = original[i]
                right, y_right, id_right = audited[i]
                torch.testing.assert_close(left, right, rtol=0, atol=0)
                self.assertEqual((y_left, id_left), (y_right, id_right))
            torch.testing.assert_close(torch.rand(5), expected, rtol=0, atol=0)

    def test_flip_and_reflected_shifts_preserve_shape_labels_and_ids(self):
        with tempfile.TemporaryDirectory() as temp:
            train, _, _ = supplied(temp)
            clean, truth, identifier = train[0]
            flip, target, image_id = audit.AuditView(train, 'flip', AUGMENTATION)[0]
            torch.testing.assert_close(flip, clean.flip(-1), rtol=0, atol=0)
            self.assertEqual((truth, identifier), (target, image_id))
            for name in audit.VIEWS:
                image, target, image_id = audit.AuditView(train, name, AUGMENTATION)[0]
                self.assertEqual(tuple(image.shape), (3, 64, 64))
                self.assertEqual((truth, identifier), (target, image_id))
            right = audit.AuditView(train, 'shift_x_plus2', AUGMENTATION)[0][0]
            torch.testing.assert_close(right[:, :, 2:], clean[:, :, :-2], rtol=0, atol=0)

    def test_fixed_model_evaluation_has_exact_counts_and_no_weight_change(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, selection, _ = supplied(root)
            network = model().eval()
            before = audit.state_digest(network)
            result = audit.evaluate_view(network, audit.AuditView(selection, 'clean', AUGMENTATION),
                root/'evaluation', batch_size=2, num_workers=0, device='cpu', classes=2)
            self.assertEqual(result['metrics']['n'], 2)
            self.assertEqual(result['targets'].tolist(), [0, 1])
            self.assertEqual(audit.state_digest(network), before)
            self.assertTrue((root/'evaluation/per_class.csv').exists())
            with self.assertRaises(FileExistsError):
                audit.evaluate_view(network, selection, root/'evaluation', batch_size=2,
                    num_workers=0, device='cpu', classes=2)

    def test_deadline_ends_before_loading_or_training(self):
        with patch.object(audit, 'load_spatial_parent') as load:
            with self.assertRaises(TimeoutError):
                audit.run_data_audit('unused', 'unused', deadline=time.monotonic()-1)
            load.assert_not_called()


class FullAuditTests(unittest.TestCase):
    def test_full_api_both_partitions_views_immutable_and_strict_reuse(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            train, selection, probe = supplied(root)
            network = model().eval()
            config = dict(dataset=dict(data_dir='fixture', download=False, num_workers=0, batch_size=2,
                                       split_seed=42), augmentation=AUGMENTATION,
                training=dict(seed=42, epochs=100, enable_tf32=False, deterministic=True))
            parent = root/'parent'; parent.mkdir()
            (parent/'best.pt').write_bytes(b'immutable checkpoint fixture')
            loaded = (parent, dict(seed=42, split_hash='fixture-split'),
                dict(config=config, model=network.state_dict()), audit.file_hash(parent/'best.pt'))

            def datasets(config, seed, expected, per_class):
                chosen = probe if per_class else train
                manifest = {name: [dataset[i][2] for i in range(len(dataset))]
                    for name, dataset in (('train_clean', chosen), ('selection', selection))}
                return chosen, selection, ['class_a', 'class_b'], 'fixture-split', manifest

            with patch.object(audit, 'load_spatial_parent', return_value=loaded), \
                    patch.object(audit, 'clean_datasets', side_effect=datasets), \
                    patch.object(audit, 'build_spatial_model', side_effect=lambda *args: copy.deepcopy(network)), \
                    patch.object(audit, 'TRAIN_IMAGES', 4), patch.object(audit, 'SELECTION_IMAGES', 2), \
                    patch.object(audit, 'PROBE_PER_CLASS', 1), \
                    patch.object(audit.torch.cuda, 'is_available', return_value=False):
                output = root/'audit'
                result = audit.run_data_audit(parent, output, examples=0)
                self.assertEqual(result['status'], 'complete')
                self.assertTrue(result['training_allowed'])
                self.assertEqual(set(result['view_scores']), {'train_probe', 'selection'})
                self.assertTrue(all(tuple(values) == audit.VIEWS for values in result['view_scores'].values()))
                self.assertFalse(result['audit_partition_evaluated'])
                self.assertFalse(result['official_validation_evaluated'])
                self.assertEqual(result, audit.run_data_audit(parent, output, examples=0))
                self.assertEqual(len(list(output.glob('attempt_*'))), 1)
                with self.assertRaisesRegex(ValueError, 'does not match'):
                    audit.run_data_audit(parent, output, examples=1)
            self.assertEqual((parent/'best.pt').read_bytes(), b'immutable checkpoint fixture')
            self.assertTrue((output/'report.md').exists())

    def test_concrete_source_error_returns_training_block_without_view_evaluation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            train, selection, _ = supplied(root)
            train.folder.targets[0] = 1
            config = dict(dataset=dict(data_dir='fixture', num_workers=0, batch_size=2),
                          augmentation=AUGMENTATION, training=dict(seed=42, enable_tf32=False))
            parent = root/'parent'; parent.mkdir()
            (parent/'best.pt').write_bytes(b'checkpoint')
            loaded = (parent, dict(seed=42, split_hash='fixture-split'),
                dict(config=config, model={}), audit.file_hash(parent/'best.pt'))
            with patch.object(audit, 'load_spatial_parent', return_value=loaded), \
                    patch.object(audit, 'TRAIN_IMAGES', 4), patch.object(audit, 'SELECTION_IMAGES', 2), \
                    patch.object(audit, 'clean_datasets', return_value=(train, selection,
                        ['class_a', 'class_b'], 'fixture-split', {})), \
                    patch.object(audit, 'evaluate_view') as evaluation, \
                    patch.object(audit.torch.cuda, 'is_available', return_value=False):
                result = audit.run_data_audit(parent, root/'audit', examples=0)
            self.assertFalse(result['training_allowed'])
            self.assertEqual(result['status'], 'complete')
            self.assertTrue(result['fatal_issues'])
            evaluation.assert_not_called()


if __name__ == '__main__':
    unittest.main()
