import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import yaml
from torch import nn

import dual_image_experiment as runner
import literature_experiment as experiment
from models.dual_image_fusion import DualImageFusion, AACNNTiny
from models.literature_resnet import build_model
from utils.literature_profiling import estimate_flops


class DualImageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_frozen_branches_and_gate_learning(self):
        for mode in ('prediction', 'feature'):
            torch.manual_seed(42)
            base = build_model('resnet50_preact')
            model = DualImageFusion(base, mode).train()
            x = torch.randn(2, 3, 64, 64)
            target = torch.nn.functional.one_hot(torch.tensor([1, 2]), 200).float()
            seen = []
            handle = model.vit.patch_embedding.register_forward_pre_hook(
                lambda m, args: seen.append(args[0]))
            loss = model.training_loss(x, target)
            handle.remove()
            self.assertIs(seen[0], x)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertGreater(model.vit.patch_embedding.weight.grad.abs().sum().item(), 0)
            self.assertGreater(model.gate[-1].weight.grad.abs().sum().item(), 0)
            self.assertTrue(all(p.grad is None for p in base.parameters()))
            self.assertFalse(base.training)
            torch.optim.SGD([p for p in model.parameters() if p.requires_grad], .001).step()
            model.verify_frozen()
            if mode == 'prediction':
                model.eval()
                with torch.no_grad():
                    out = model(x)
                torch.testing.assert_close(out.exp().sum(-1), torch.ones(2))
            self.assertGreater(estimate_flops(model, 'cpu'), 0)

    def test_aacnn_scratch_both_branches(self):
        model = AACNNTiny()
        x = torch.randn(2, 3, 64, 64)
        out = model(x)
        self.assertEqual(out.shape, (2, 200))
        out.square().mean().backward()
        self.assertGreater(model.cnn[0].weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.vit.patch_embedding.weight.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.requires_grad for p in model.parameters()))

    def test_plan_horizons_and_smoke(self):
        config = yaml.safe_load(experiment.DEFAULT_CONFIG.read_text())
        summary = dict(split_hash='x', total_train_seconds=10)
        with tempfile.TemporaryDirectory() as temp, \
                patch.object(runner, 'ROOT', Path(temp)), \
                patch.object(runner, 'file_hash', return_value='hash'), \
                patch.object(runner, 'find_parent', return_value=(Path(temp), summary, {'config': config})):
            for smoke in (False, True):
                args = runner.arguments().parse_args(['--smoke'] if smoke else [])
                suite = runner.create_suite(args)
                plan = json.loads((suite/'suite.json').read_text())
                self.assertEqual([j['epochs'] for j in plan['jobs']],
                                 [1]*4 if smoke else [100, 100, 25, 100])

    def test_custom_loss_training_checkpoint_integration(self):
        batch = [(torch.randn(2, 3, 64, 64), torch.tensor([0, 1]), ['a', 'b'])]
        config = yaml.safe_load(experiment.DEFAULT_CONFIG.read_text())
        config['dataset'].update(batch_size=2, num_workers=0)
        config['training'].update(epochs=1, mixed_precision=False, enable_tf32=False)
        config['training']['scheduler']['warmup_epochs'] = 0
        config['profiling'].update(warmup_iterations=0, benchmark_iterations=1)
        with tempfile.TemporaryDirectory() as temp, \
                patch.object(experiment, 'prepare_datasets', return_value=(
                    {'train': batch, 'selection': batch, 'audit': batch}, {}, 'test')), \
                patch.object(experiment, 'loader', side_effect=lambda data, *args: data):
            for variant in ('prediction100', 'feature100', 'aacnn100'):
                def create(c, v, s, fingerprint):
                    model = (AACNNTiny() if v == 'aacnn100' else
                             DualImageFusion(build_model('resnet50_preact'),
                                             'feature' if v == 'feature100' else 'prediction'))
                    return model, dict(architecture={'test': True}, parent_train_seconds=0)
                run = Path(temp)/variant
                self.assertTrue(experiment.train_run(copy.deepcopy(config), variant, 42, run,
                                                     model_factory=create))
                s = json.loads((run/'summary.json').read_text())
                self.assertEqual(s['completed_epochs'], 1)
                self.assertGreater(s['forward_flops_estimate'], 0)
                self.assertTrue((run/'best.pt').exists())
            (Path(temp)/'frozen_base.json').write_text(json.dumps(
                dict(heldout={'accuracy': .5}, forward_flops_estimate=2596438016)))
            plan = dict(jobs=[dict(variant=v, intended_full_epochs=100)
                              for v in ('prediction100', 'feature100', 'aacnn100')])
            runner.comparison(Path(temp), plan)
            self.assertTrue((Path(temp)/'comparison.csv').exists())


if __name__ == '__main__':
    unittest.main()
