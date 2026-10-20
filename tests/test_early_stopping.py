import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
import yaml
import literature_experiment as experiment


class EarlyStoppingTests(unittest.TestCase):
    def test_ties_regressions_and_recovery(self):
        rows = lambda values: [dict(selection_accuracy=v) for v in values]
        self.assertEqual(experiment.stale_selection_epochs(rows([.6]*5), .6), 5)
        self.assertEqual(experiment.stale_selection_epochs(rows([.5, .55, .59]), .6), 3)
        self.assertEqual(experiment.stale_selection_epochs(rows([.6, .59, .61]), .6), 0)
        self.assertEqual(experiment.stale_selection_epochs(rows([.6]*6)), 5)

    def test_stop_saves_summary_and_best(self):
        torch.set_num_threads(2)
        config = yaml.safe_load(experiment.DEFAULT_CONFIG.read_text())
        config['training'].update(epochs=20, early_stopping_patience=5, mixed_precision=False)
        config['dataset'].update(batch_size=2, num_workers=0)
        config['profiling'].update(warmup_iterations=0, benchmark_iterations=1)
        batch = [(torch.ones(2, 3, 64, 64), torch.tensor([0, 1]), ['a', 'b'])]
        def factory(*args):
            model = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 200))
            return model, dict(architecture={'test': True})
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(experiment, 'prepare_datasets', return_value=(
                    {'train': batch, 'selection': batch, 'audit': batch}, {}, 'test')), \
                patch.object(experiment, 'loader', side_effect=lambda data, *a: data), \
                patch.object(experiment, 'evaluate', return_value=dict(accuracy=.5, top5_accuracy=.8, nll=1.)):
            run = Path(tmp)/'run'
            self.assertTrue(experiment.train_run(config, 'test', 42, run, model_factory=factory))
            summary = json.loads((run/'summary.json').read_text())
            self.assertEqual(summary['completed_epochs'], 6)
            self.assertEqual(summary['best_epoch'], 1)
            self.assertTrue(summary['stopped_early'])
            self.assertEqual(summary['planned_epochs'], 20)


if __name__ == '__main__':
    unittest.main()
