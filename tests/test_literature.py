import argparse
import copy
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
from utils.literature_legacy import LEGACY_SOURCES, is_legacy_source
from utils.literature_eval_migration import SOURCES_85DD1E7


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

    def test_fp16_overflow_regression(self):
        m = BoTAttention(8, 4, heads=2).eval()
        with torch.no_grad():
            m.qkv.weight.fill_(0.25)
        x = torch.full((2, 8, 4, 4), 200.)
        with torch.autocast('cpu', dtype=torch.float16):
            q, k, _ = m.qkv(x).reshape(2, 3, 2, 4, 16).unbind(1)
            old_logits = (q.transpose(-2, -1) / 2) @ k
            self.assertFalse(torch.isfinite(old_logits).all())
            self.assertFalse(torch.isfinite(old_logits.float().softmax(-1)).all())
            fixed = m(x)
        self.assertEqual(fixed.dtype, torch.float32)
        self.assertTrue(torch.isfinite(fixed).all())
        torch.testing.assert_close(fixed, m(x))
        fixed.square().mean().backward()
        self.assertTrue(torch.isfinite(m.qkv.weight.grad).all())
        a = Attention2d(8, 8, heads=2, dim_head=4).eval()
        with torch.no_grad():
            a.to_q.weight.fill_(0.25)
            a.to_kv.weight.fill_(0.25)
        with torch.autocast('cpu', dtype=torch.float16):
            fixed_a, weights = a(x)
        self.assertTrue(torch.isfinite(fixed_a).all())
        self.assertTrue(torch.isfinite(weights).all())
        torch.testing.assert_close(fixed_a, a(x)[0])

    def test_evaluation_disables_amp_for_entire_model(self):
        model = nn.Sequential(nn.Flatten(), nn.Linear(3*4*4, 200, bias=False))
        with torch.no_grad():
            model[1].weight.fill_(100.)
        x = torch.full((2, 3, 4, 4), 100.)
        with torch.autocast('cpu', dtype=torch.float16):
            self.assertFalse(torch.isfinite(model(x)).all())
        batches = [(x, torch.zeros(2, dtype=torch.long), ['a', 'b'])]
        scores = experiment.evaluate(model, batches, 'cpu', True, torch.float16)
        self.assertTrue(torch.isfinite(torch.tensor(scores['nll'])))
        with torch.no_grad():
            model[1].weight.fill_(float('nan'))
        with self.assertRaisesRegex(RuntimeError, 'Non-finite FP32'):
            experiment.evaluate(model, batches, 'cpu', True, torch.float16)

    def test_legacy_compatibility_is_cnn_only_and_exact(self):
        legacy = {path: choices[0] for path, choices in LEGACY_SOURCES.items()}
        self.assertTrue(is_legacy_source(legacy))
        summary = dict(variant='resnet50', source_hashes=legacy)
        self.assertEqual(experiment.comparable_source(summary), experiment.source_hashes())
        summary['variant'] = 'botnet50'
        self.assertNotEqual(experiment.comparable_source(summary), experiment.source_hashes())
        changed = dict(legacy, **{'main.py': 'unknown modification'})
        self.assertFalse(is_legacy_source(changed))
        self.assertFalse(is_legacy_source({}))

    def test_repair_creates_new_suite_without_overwriting_old_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            original = Path(tmp)/'old'
            cnn = original/'resnet50_seed42'
            bot = original/'botnet50_seed42'
            cnn.mkdir(parents=True)
            bot.mkdir()
            config = yaml.safe_load(experiment.DEFAULT_CONFIG.read_text())
            legacy = {path: choices[0] for path, choices in LEGACY_SOURCES.items()}
            old = dict(config=config, source_hashes=legacy,
                       jobs=[dict(variant=v, seed=42, directory=f'{v}_seed42') for v in VARIANTS])
            experiment.save_json(original/'suite.json', old)
            summary = dict(protocol=experiment.PROTOCOL, variant='resnet50', seed=42,
                           source_hashes=legacy, recipe_hash=experiment.recipe_hash(config),
                           completed_epochs=15, environment={'synthetic': True}, split_hash='test')
            experiment.save_json(cnn/'summary.json', summary)
            (bot/'last.pt').write_bytes(b'old partial checkpoint must stay untouched')
            before = {p: p.read_bytes() for p in original.rglob('*') if p.is_file()}
            args = argparse.Namespace(from_suite=str(original), output=str(Path(tmp)/'repaired'))
            with patch.object(experiment, 'runtime', return_value=('cpu', False, torch.float16, None)), \
                 patch.object(experiment, 'environment', return_value={'synthetic': True}), \
                 patch.object(experiment, 'prepare_datasets', return_value=({}, {}, 'test')), \
                 patch.object(experiment, 'run_suite') as runner:
                experiment.repair_suite(args, None)
                runner.assert_called_once()
            repaired = json.loads((Path(args.resume)/'suite.json').read_text())
            self.assertEqual(repaired['jobs'][0]['reuse_run'], str(cnn.resolve()))
            self.assertNotIn('reuse_run', repaired['jobs'][1])
            self.assertEqual(repaired['source_hashes'], experiment.source_hashes())
            for path, contents in before.items():
                self.assertEqual(contents, path.read_bytes())
            bad = copy.deepcopy(old)
            bad['source_hashes']['main.py'] = 'unknown modification'
            experiment.save_json(original/'suite.json', bad)
            with self.assertRaisesRegex(ValueError, 'exact ff14146'):
                experiment.repair_suite(args, None)

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

    def test_evaluation_migration_resumes_optimizer_without_retraining(self):
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
            original = root/'original'
            partial = original/'botnet50_seed42'
            legacy = {p: choices[0] for p, choices in SOURCES_85DD1E7.items()}
            with patch('datasets.resnet_tiny.make_datasets', fake_data), patch.object(experiment, 'build_model', toy_model):
                experiment.train_run(config, 'botnet50', 42, partial, deadline=time.perf_counter()-1)
                cnn = root/'completed_cnn'
                experiment.train_run(config, 'resnet50', 42, cnn)
                old_cnn_last = torch.load(cnn/'last.pt', weights_only=False)
                for name in ('best.pt', 'last.pt'):
                    state = torch.load(cnn/name, weights_only=False)
                    state['metadata']['source_hashes'] = {p: choices[0] for p, choices in LEGACY_SOURCES.items()}
                    state['metadata'].pop('evaluation_precision', None)
                    state['metadata'].pop('inference_precision', None)
                    experiment.atomic_save(cnn/name, state)
                for name in ('best.pt', 'last.pt'):
                    state = torch.load(partial/name, weights_only=False)
                    state['metadata']['source_hashes'] = legacy
                    state['metadata'].pop('evaluation_precision', None)
                    state['metadata'].pop('inference_precision', None)
                    experiment.atomic_save(partial/name, state)
                experiment.save_json(original/'suite.json', dict(config=config, source_hashes=legacy,
                    jobs=[dict(variant='resnet50', seed=42, directory='resnet50_seed42', reuse_run=str(cnn)),
                          dict(variant='botnet50', seed=42, directory='botnet50_seed42')]))
                old_bytes = (partial/'last.pt').read_bytes()
                args = argparse.Namespace(from_suite=str(original), output=str(root/'new'))
                experiment.migrate_fp32_evaluation(args, None)
                migrated = Path(args.resume)/'botnet50_seed42'
                self.assertEqual((partial/'last.pt').read_bytes(), old_bytes)
                migrated_state = torch.load(migrated/'last.pt', weights_only=False)
                self.assertEqual(migrated_state['epoch'], 2)
                self.assertEqual(migrated_state['metadata']['evaluation_migration']['completed_epochs_before_migration'], 1)
                full = root/'full'
                experiment.train_run(config, 'botnet50', 42, full)
                complete = torch.load(full/'last.pt', weights_only=False)
                for k, value in complete['model'].items():
                    torch.testing.assert_close(value, migrated_state['model'][k], atol=0, rtol=0)
                summary = json.loads((migrated/'summary.json').read_text())
                self.assertEqual(summary['evaluation_precision'], 'float32')
                self.assertEqual(summary['inference_precision'], 'float32')
                migrated_cnn = torch.load(Path(args.resume)/'resnet50_seed42'/'last.pt', weights_only=False)
                self.assertEqual(migrated_cnn['epoch'], 2)
                for key, value in old_cnn_last['model'].items():
                    torch.testing.assert_close(value, migrated_cnn['model'][key], atol=0, rtol=0)
                self.assertEqual([r['train_seconds'] for r in old_cnn_last['history']],
                                 [r['train_seconds'] for r in migrated_cnn['history']])

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
