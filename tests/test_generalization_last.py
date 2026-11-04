import copy
import csv
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import Dataset

import generalization_last as runner
from test_baseline_diagnostics import config as original_config, fixture_datasets, toy_model
from datasets.utility_tiny import UtilityDataset


class Vectors(Dataset):
    split = 'selection'
    epoch = 0

    def __len__(self):
        return 4

    def __getitem__(self, index):
        x = torch.tensor([[2., 0.], [0., 2.], [1., -1.], [-1., 1.]])[index]
        return x, [0, 1, 1, 0][index], f'image/{index}'


class Branch(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.head = nn.Linear(2, 2, bias=False)
        self.correction = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.encoder.head.weight.copy_(torch.eye(2))
            self.correction.weight.copy_(-2*torch.eye(2))

    def forward(self, x):
        return self.encoder.head(x)+self.correction(x)


class PreviousLastTests(unittest.TestCase):
    def test_branch_ablation_reports_offsetting_corrections_and_cleans_hooks(self):
        with tempfile.TemporaryDirectory() as temp:
            model = Branch()
            before = runner.state_digest(model)
            result = runner.branch_contribution(model, Vectors(),
                dict(dataset=dict(batch_size=2, num_workers=0)), Path(temp), 'cpu')
            self.assertEqual(result['n'], 4)
            self.assertEqual(result['corrected'], 2)
            self.assertEqual(result['harmed'], 2)
            self.assertEqual(result['prediction_changes'], 4)
            self.assertEqual(result['delta_top1_pp'], 0.)
            self.assertAlmostEqual(result['rms_ratio'], 2.)
            self.assertEqual(before, runner.state_digest(model))
            self.assertFalse(model.correction._forward_hooks)
            self.assertFalse(model.encoder.head._forward_hooks)
            self.assertTrue((Path(temp)/'branch_contribution.json').is_file())

    def test_deadline_restores_hooks_and_never_updates_weights(self):
        with tempfile.TemporaryDirectory() as temp:
            model = Branch()
            before = runner.state_digest(model)
            with self.assertRaises(TimeoutError):
                runner.branch_contribution(model, Vectors(),
                    dict(dataset=dict(batch_size=2, num_workers=0)), Path(temp), 'cpu', time.monotonic()-1)
            self.assertEqual(before, runner.state_digest(model))
            self.assertFalse(model.correction._forward_hooks)
            self.assertFalse(model.encoder.head._forward_hooks)

    def test_actual_last_api_outputs_three_distinct_reports_and_reuses_completion(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            parent = root/'parent'; parent.mkdir()
            (parent/'best.pt').write_bytes(b'original parent')
            previous = root/'old_fusion'; previous.mkdir()
            output = root/'last_reports'
            config = original_config()
            config['dataset'].update(batch_size=2, num_workers=0)
            config['training']['mixed_precision'] = False
            datasets, manifest, fingerprint = fixture_datasets(root/'images')
            base = datasets['train']
            clean = UtilityDataset(base.folder, base.indices, 'train_clean', 42)
            selection = datasets['selection']
            source_model = toy_model()
            parent_hash = runner.file_hash(parent/'best.pt')
            old_plan = dict(protocol='tiny-cnn8-fusion30-v1', smoke=False,
                parent_checkpoint_sha256=parent_hash, split_hash=fingerprint,
                source_hashes={name: runner.file_hash(runner.ROOT/name)
                    for name in ('fusion30_experiment.py', 'models/spatial_fusion.py')})
            (previous/'suite.json').write_text(json.dumps(old_plan))
            for variant in runner.FUSION_VARIANTS:
                directory = previous/variant; directory.mkdir()
                torch.save(dict(epoch=30, model=source_model.state_dict(), best=.6,
                    history=[dict(epoch=i, selection_accuracy=.6) for i in range(31)]), directory/'last.pt')
            loaded = (parent, dict(seed=42, split_hash=fingerprint),
                      dict(config=config, model=source_model.state_dict()), parent_hash)
            with patch.object(runner, 'load_spatial_parent', return_value=loaded), \
                    patch.object(runner, 'clean_datasets', return_value=(clean, selection, [], fingerprint, manifest)), \
                    patch.object(runner, 'build_spatial_model', side_effect=lambda *args: copy.deepcopy(source_model)), \
                    patch.object(runner, 'build_fusion_model', side_effect=lambda *args: copy.deepcopy(source_model)), \
                    patch.object(runner, '_checkpoint_matches') as guard:
                result = runner.run_previous_last(parent, previous, output, config=config, examples=0)
                self.assertEqual(result['status'], 'complete')
                self.assertEqual(guard.call_count, 3)
                for variant in runner.FUSION_VARIANTS:
                    self.assertEqual(result['scores'][variant]['train_clean']['n'], len(clean))
                    self.assertEqual(result['scores'][variant]['selection']['n'], len(selection))
                again = runner.run_previous_last(parent, previous, output, config=config, examples=0)
                self.assertEqual(result, again)
                self.assertEqual(guard.call_count, 3)
            table = output/result['analysis_directory']/'comparison.csv'
            with table.open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 3)
            self.assertTrue(all(row['checkpoint'] == 'last.pt' and row['added_epoch'] == '30' for row in rows))
            self.assertFalse(result['audit_partition_evaluated'])
            self.assertEqual((parent/'best.pt').read_bytes(), b'original parent')


if __name__ == '__main__':
    unittest.main()
