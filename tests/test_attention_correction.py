import json
from pathlib import Path
import tarfile
import tempfile
import time
import unittest
from unittest.mock import patch

import torch
import yaml
from torch import nn

from models.attention_correction import FrozenAttentionCorrection, TinyViTFP32
from models.literature_resnet import build_model
from utils.literature_profiling import estimate_flops
from overnight_correction import bundle_results, finish
from correction_experiment import write_comparison, factory
import literature_experiment as experiment


class CorrectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_identity_frozen_buffers_and_gradient_flow(self):
        torch.manual_seed(42)
        base = build_model('resnet50_preact').eval()
        model = FrozenAttentionCorrection(base)
        x = torch.randn(2, 3, 64, 64)
        with torch.no_grad():
            reference = base(x)
        model.train()
        self.assertFalse(base.training)
        self.assertTrue(all(not m.training for m in base.modules() if isinstance(m, nn.BatchNorm2d)))
        torch.testing.assert_close(model(x), reference, atol=0, rtol=0)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001)
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(x), torch.tensor([1, 2]))
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            optimizer.step()
        self.assertIsNotNone(model.readout.in_proj_weight.grad)
        self.assertGreater(model.readout.in_proj_weight.grad.abs().sum().item(), 0.)
        self.assertTrue(all(p.grad is None for p in base.parameters()))
        model.verify_frozen()
        with torch.no_grad():
            self.assertFalse(torch.equal(model(x), reference))
        base_bn = next(m for m in base.modules() if isinstance(m, nn.BatchNorm2d))
        base_bn.running_mean.add_(1)
        with self.assertRaisesRegex(RuntimeError, 'Frozen CNN'):
            model.verify_frozen()

    def test_pooled_and_vit(self):
        model = FrozenAttentionCorrection(build_model('resnet50_preact'), mode='pooled')
        self.assertEqual(model(torch.randn(2, 3, 64, 64)).shape, (2, 200))
        vit = TinyViTFP32()
        logits = vit(torch.randn(2, 3, 64, 64))
        nn.functional.cross_entropy(logits, torch.tensor([0, 1])).backward()
        self.assertTrue(torch.isfinite(logits).all())
        self.assertGreater(estimate_flops(vit, 'cpu'), 0)
        self.assertGreater(estimate_flops(model, 'cpu'), 0)
        model.verify_frozen()

    def test_split_mismatch_rejected_before_loading(self):
        with self.assertRaisesRegex(ValueError, 'split'):
            factory({'split_hash': 'original'})({}, 'attention_correction', 42, 'wrong')

    def test_training_checkpoint_resume_and_epoch_zero_candidate(self):
        class Toy(nn.Module):
            def __init__(self):
                super().__init__()
                self.base = nn.Linear(3, 200).requires_grad_(False)
                self.delta = nn.Linear(3, 200)
                nn.init.zeros_(self.delta.weight)
                nn.init.zeros_(self.delta.bias)

            def forward(self, x):
                x = x.mean((2, 3))
                return self.base(x).detach() + self.delta(x)

            def verify_frozen(self):
                assert all(p.grad is None for p in self.base.parameters())

        def create(config, variant, seed, fingerprint):
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                return Toy(), dict(architecture={'test': True})

        batch = [(torch.ones(2, 3, 64, 64), torch.tensor([0, 1]), ['a', 'b'])]
        config = yaml.safe_load(experiment.DEFAULT_CONFIG.read_text())
        config['dataset'].update(batch_size=2, num_workers=0)
        config['training'].update(epochs=2, mixed_precision=False, enable_tf32=False)
        config['training']['scheduler']['warmup_epochs'] = 1
        config['profiling'].update(warmup_iterations=0, benchmark_iterations=1)
        with tempfile.TemporaryDirectory() as temp, \
                patch.object(experiment, 'prepare_datasets', return_value=(
                    {'train': batch, 'selection': batch, 'audit': batch}, {}, 'test')), \
                patch.object(experiment, 'loader', side_effect=lambda data, *args: data):
            run = Path(temp)/'run'
            self.assertFalse(experiment.train_run(config, 'test', 42, run,
                             deadline=time.perf_counter()-1, model_factory=create))
            self.assertTrue((run/'best.pt').exists())
            self.assertTrue(experiment.train_run(None, None, None, run, resume=True, model_factory=create))
            summary = json.loads((run/'summary.json').read_text())
            self.assertEqual(summary['completed_epochs'], 2)
            self.assertEqual(summary['trainable_params'], 800)
            self.assertIn('initial_selection', summary)
            saved = torch.load(run/'last.pt', weights_only=False)
            self.assertEqual(sum(len(g['params']) for g in saved['optimizer']['param_groups']), 2)

    def test_bundle_excludes_checkpoints_and_datasets(self):
        with tempfile.TemporaryDirectory() as temp:
            suite = Path(temp)/'suite_test'
            suite.mkdir()
            (suite/'summary.json').write_text('{}')
            (suite/'best.pt').write_bytes(b'dummy')
            (suite/'image.JPEG').write_bytes(b'dummy')
            log = suite/'overnight.log'
            log.write_text('finished')
            archive = bundle_results(suite, log)
            with tarfile.open(archive) as f:
                names = f.getnames()
            self.assertIn('suite_test/summary.json', names)
            self.assertIn('suite_test/training.log', names)
            self.assertFalse(any(n.endswith(('.pt', '.JPEG')) for n in names))

    @patch('overnight_correction.stop_workbench')
    @patch('overnight_correction.bundle_results', side_effect=OSError('disk full'))
    def test_shutdown_even_if_bundle_fails(self, bundle, stop):
        with self.assertRaises(OSError):
            finish('suite', 'log', 'url', True)
        stop.assert_called_once_with('url', True)

    @patch('overnight_correction.stop_workbench')
    @patch('overnight_correction.subprocess.run', side_effect=RuntimeError('upload failure'))
    @patch('overnight_correction.bundle_results', return_value=Path('test.tar.gz'))
    def test_shutdown_even_if_upload_fails(self, bundle, run, stop):
        with self.assertRaises(RuntimeError):
            finish('suite', 'log', 'url', True, 'gs://test/results')
        stop.assert_called_once_with('url', True)

    def test_comparison_ignores_parent_copy(self):
        with tempfile.TemporaryDirectory() as temp:
            suite = Path(temp)
            (suite/'parent_provenance').mkdir()
            (suite/'parent_provenance'/'summary.json').write_text('{}')
            summary = dict(variant='frozen_base', heldout={'accuracy': .6, 'top5_accuracy': .8},
                           params=100, trainable_params=0, forward_flops_estimate=1000,
                           total_train_seconds=0, parent_train_seconds=600,
                           inference={'1': {'latency_per_batch_ms': 1},
                                      '64': {'throughput_images_per_second': 100}},
                           inference_batch_size=64)
            (suite/'frozen_base.json').write_text(json.dumps(summary))
            write_comparison(suite)
            self.assertTrue((suite/'comparison.csv').exists())


if __name__ == '__main__':
    unittest.main()
