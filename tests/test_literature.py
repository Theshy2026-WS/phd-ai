import argparse
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from PIL import Image
import torch
from torch import nn
from torchvision.datasets import ImageFolder
import yaml

from datasets.utility_tiny import UtilityDataset
from models.literature_alternet import AttentionBlockB, LocalAttention, Attention2d
from models.literature_resnet import build_model, BoTAttention, VARIANTS
from utils.literature_profiling import estimate_flops
from utils.profiling import estimate_model_flops
import literature_experiment as experiment


class LiteratureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_all_architectures_forward_backward_and_layout(self):
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                model = build_model(variant)
                shapes = []
                hooks = [getattr(model, f'layer{i}').register_forward_hook(
                    lambda m, inputs, output: shapes.append(output.shape[-1])) for i in range(1, 5)]
                output = model(torch.randn(2, 3, 64, 64))
                for hook in hooks:
                    hook.remove()
                self.assertEqual(shapes, [32, 16, 8, 4])
                self.assertEqual(tuple(output.shape), (2, 200))
                output.square().mean().backward()
                for name, p in model.named_parameters():
                    self.assertIsNotNone(p.grad, name)
                    self.assertTrue(torch.isfinite(p.grad).all(), name)
                if variant == 'alternet50':
                    self.assertEqual(sum(isinstance(m, AttentionBlockB) for m in model.modules()), 6)
                    self.assertEqual([len(getattr(model, f'layer{i}')) for i in range(1, 5)], [3, 4, 6, 4])
                    for layer in (model.layer2, model.layer3, model.layer4):
                        self.assertIsInstance(layer[-1], AttentionBlockB)
                    for m in model.modules():
                        if isinstance(m, LocalAttention):
                            self.assertEqual(m.window_size, 4)
                if variant == 'botnet50':
                    self.assertEqual(sum(isinstance(m, BoTAttention) for m in model.modules()), 3)
                del model, output

    def test_bot_matches_explicit_relative_attention_and_gradients(self):
        m = BoTAttention(8, 3, heads=2).double()
        x = torch.randn(2, 8, 3, 3, dtype=torch.float64, requires_grad=True)
        q, k, v = m.qkv(x).reshape(2, 3, 2, 4, 9).unbind(1)
        q = q.transpose(-2, -1) / 2
        relative = m.rel_h[m.relative_h] + m.rel_w[m.relative_w]
        logits = q @ k + torch.einsum('bhid,ijd->bhij', q, relative)
        weights = logits.float().softmax(-1).double()
        reference = (weights @ v.transpose(-2, -1)).transpose(-2, -1).reshape_as(x)
        actual = m(x)
        torch.testing.assert_close(actual, reference, rtol=1e-6, atol=1e-7)
        actual.square().mean().backward()
        self.assertGreater(m.rel_h.grad.abs().sum().item(), 0)
        self.assertGreater(m.rel_w.grad.abs().sum().item(), 0)

    def test_alternet_window_partition_matches_independent_windows(self):
        m = LocalAttention(8, 12, window_size=4, heads=3, dim_head=4).eval()
        x = torch.randn(2, 8, 8, 8)
        actual, _ = m(x)
        expected = torch.empty_like(actual)
        bias = m.pos_embedding[m.rel_index[:, :, 0], m.rel_index[:, :, 1]]
        for y in (0, 4):
            for z in (0, 4):
                expected[:, :, y:y+4, z:z+4] = m.attn(x[:, :, y:y+4, z:z+4], bias)[0]
        torch.testing.assert_close(actual, expected)

    def test_attention_autocast_is_finite(self):
        # CPU bf16 checks mixed-dtype paths, not CUDA fp16 performance/stability.
        for m in (BoTAttention(16, 4), LocalAttention(16, 16, window_size=4, heads=2, dim_head=8)):
            with torch.autocast('cpu', dtype=torch.bfloat16):
                result = m(torch.randn(2, 16, 4, 4))
                result = result[0] if isinstance(result, tuple) else result
                loss = result.float().square().mean()
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            for p in m.parameters():
                self.assertTrue(torch.isfinite(p.grad).all())

    def test_attention_flops_include_functional_matmuls(self):
        for variant in ('botnet50', 'alternet50'):
            model = build_model(variant)
            extra = [0]
            def count(m, args, out):
                b, _, h, w = args[0].shape
                n = h*w
                if isinstance(m, Attention2d):
                    extra[0] += 4*b*n*n*m.to_q.out_channels
                else:
                    d = m.heads*m.dim_head
                    extra[0] += 4*b*n*n*d + 4*b*n*(2*m.size-1)*d
            handles = [m.register_forward_hook(count) for m in model.modules()
                       if isinstance(m, (BoTAttention, Attention2d))]
            base = estimate_model_flops(model, (3, 64, 64), 'cpu')
            for h in handles:
                h.remove()
            self.assertGreater(extra[0], 0)
            self.assertEqual(estimate_flops(model, 'cpu'), base+extra[0])

    def test_synthetic_training_time_budget_resume_and_comparison(self):
        def toy_model(variant, seed):
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                return nn.Sequential(nn.Conv2d(3, 4, 1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
                                     nn.Flatten(), nn.Linear(4, 200))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = root/'images'/'class0'
            images.mkdir(parents=True)
            for i in range(8):
                Image.new('RGB', (64, 64), (i*20, 40, 60)).save(images/f'{i}.png')
            folder = ImageFolder(root/'images')
            def fake_data(config, seed):
                return {k: UtilityDataset(folder, ids, k, seed) for k, ids in
                        [('train', [0, 1, 2, 3]), ('selection', [4, 5]), ('audit', [6, 7])]}, {}, 'synthetic'
            config = yaml.safe_load(experiment.DEFAULT_CONFIG.read_text())
            config['dataset'].update(batch_size=2, num_workers=0)
            config['training'].update(epochs=2, mixed_precision=False, enable_tf32=False)
            config['training']['scheduler']['warmup_epochs'] = 1
            config['profiling'].update(warmup_iterations=0, benchmark_iterations=1)
            with patch('datasets.resnet_tiny.make_datasets', fake_data), patch.object(experiment, 'build_model', toy_model):
                partial = root/'partial'
                self.assertFalse(experiment.train_run(config, 'resnet50', 42, partial, deadline=time.perf_counter()-1))
                self.assertFalse((partial/'summary.json').exists())
                self.assertEqual(torch.load(partial/'last.pt', weights_only=False)['epoch'], 1)
                self.assertTrue(experiment.train_run(None, None, None, partial, resume=True))
                full = root/'full'
                self.assertTrue(experiment.train_run(config, 'resnet50', 42, full))
                a = torch.load(partial/'last.pt', weights_only=False)
                b = torch.load(full/'last.pt', weights_only=False)
                for k in a['model']:
                    torch.testing.assert_close(a['model'][k], b['model'][k], atol=0, rtol=0)
                initial = toy_model('resnet50', 42).state_dict()
                self.assertTrue(any(not torch.equal(initial[k], b['model'][k]) for k in initial))
                self.assertEqual(a['history'][-1]['lr'], b['history'][-1]['lr'])
                bot = root/'bot'
                experiment.train_run(config, 'botnet50', 42, bot)
                experiment.compare([full, bot], root/'comparison.csv')
                self.assertIn('delta_vs_control_pp', (root/'comparison.csv').read_text())
                with self.assertRaises(ValueError):
                    experiment.compare([full, partial], root/'invalid.csv')
                summary = json.loads((bot/'summary.json').read_text())
                summary['recipe_hash'] = 'different'
                experiment.save_json(bot/'summary.json', summary)
                with self.assertRaisesRegex(ValueError, 'recipe_hash'):
                    experiment.compare([full, bot], root/'invalid.csv')
                with patch.object(experiment, 'source_hashes', return_value={}):
                    with self.assertRaisesRegex(ValueError, 'Source changed'):
                        experiment.train_run(None, None, None, partial, resume=True)

    def test_suite_resume_skips_completed_and_preserves_schedule(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = argparse.Namespace(output=tmp, config=str(experiment.DEFAULT_CONFIG), epochs=15,
                                      variants=['resnet50', 'botnet50'], seeds=[42], resume=None)
            def fake_train(config, variant, seed, run, resume=False, deadline=None):
                Path(run).mkdir(parents=True, exist_ok=True)
                experiment.save_json(Path(run)/'summary.json', {'test': True})
                return True
            with patch.object(experiment, 'train_run', side_effect=fake_train) as mocked, patch.object(experiment, 'compare'):
                experiment.run_suite(args, None)
                self.assertEqual(mocked.call_count, 2)
                args.resume = str(next(Path(tmp).iterdir()))
                mocked.reset_mock()
                experiment.run_suite(args, None)
                mocked.assert_not_called()
                args.epochs = 30
                with self.assertRaisesRegex(ValueError, 'epoch schedule'):
                    experiment.run_suite(args, None)


if __name__ == '__main__':
    unittest.main()
