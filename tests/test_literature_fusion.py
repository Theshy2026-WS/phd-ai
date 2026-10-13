import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import yaml
from PIL import Image
from torchvision.datasets import ImageFolder

from models.literature_resnet import build_model
from models.literature_acmix import ACmix
from models.literature_conformer import TinyConformer, ConformerAttentionFP32
from models.conformer_author import Conformer
from utils.literature_training import soft_target_loss
from utils.literature_profiling import estimate_flops
from utils.profiling import estimate_model_flops
from utils.literature_fusion_compat import BASELINE_EC0D06A
from datasets.utility_tiny import UtilityDataset
import literature_experiment as experiment


class FusionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_acmix_shapes_gradients_and_checkpoint_equivalence(self):
        model = build_model('acmix50')
        self.assertEqual(sum(p.numel() for p in model.parameters()), 16687096)
        modules = [m for m in model.modules() if isinstance(m, ACmix)]
        self.assertEqual(len(modules), 16)
        shapes = []
        handles = [getattr(model, f'layer{i}').register_forward_hook(
            lambda m, x, y: shapes.append(tuple(y.shape[-2:]))) for i in range(1, 5)]
        x = torch.randn(2, 3, 64, 64)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            y = model(x)
        self.assertEqual(y.shape, (2, 200))
        self.assertEqual(shapes, [(32, 32), (16, 16), (8, 8), (4, 4)])
        y.float().square().mean().backward()
        for name, p in model.named_parameters():
            self.assertIsNotNone(p.grad, name)
            self.assertTrue(torch.isfinite(p.grad).all(), name)
        for handle in handles:
            handle.remove()
        for stride in (1, 2):
            a = ACmix(8, stride=stride)
            b = copy.deepcopy(a)
            b.checkpoint_attention = False
            z = torch.randn(2, 8, 8, 8)
            ya, yb = a(z), b(z)
            torch.testing.assert_close(ya, yb, rtol=0, atol=0)
            ya.square().mean().backward()
            yb.square().mean().backward()
            for pa, pb in zip(a.parameters(), b.parameters()):
                torch.testing.assert_close(pa.grad, pb.grad, rtol=0, atol=0)

    def test_conformer_matches_author_and_shapes(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            reference = Conformer(patch_size=16, channel_ratio=1, embed_dim=384, depth=12,
                                  num_heads=6, mlp_ratio=4, qkv_bias=True, num_classes=200).eval()
        model = build_model('conformer_tiny', 42).eval()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 23011104)
        x = torch.randn(2, 3, 64, 64)
        sizes = []
        handles = [m.register_forward_pre_hook(lambda m, a: sizes.append(tuple(a[0].shape)))
                   for m in model.modules() if isinstance(m, ConformerAttentionFP32)]
        with torch.no_grad():
            heads = reference(x)
            torch.testing.assert_close(model(x), heads[0] + heads[1], atol=0, rtol=0)
        self.assertEqual(sizes, [(2, 17, 384)] * 12)
        for h in handles:
            h.remove()
        model.train()
        with torch.autocast('cpu', dtype=torch.bfloat16):
            outputs = model(x)
            self.assertEqual(len(outputs), 2)
            targets = torch.full((2, 200), 1 / 200)
            loss = soft_target_loss(outputs, targets)
        loss.backward()
        for name, p in model.named_parameters():
            self.assertIsNotNone(p.grad, name)
            self.assertTrue(torch.isfinite(p.grad).all(), name)

    def test_dual_head_loss_is_mean_of_losses_not_loss_of_mean(self):
        a = torch.tensor([[5., -2.]], requires_grad=True)
        b = torch.tensor([[-4., 3.]], requires_grad=True)
        target = torch.tensor([[0.8, 0.2]])
        expected = (soft_target_loss(a, target) + soft_target_loss(b, target)) / 2
        actual = soft_target_loss([a, b], target)
        torch.testing.assert_close(actual, expected)
        self.assertGreater(float((actual - soft_target_loss((a + b) / 2, target)).detach()), 1.)
        actual.backward()
        self.assertIsNotNone(a.grad)
        self.assertIsNotNone(b.grad)

    def test_flops_account_for_functional_attention(self):
        for variant, extra in [('acmix50', 48570368),
                               ('conformer_tiny', 4 * 12 * 17**2 * 384)]:
            # ACmix: 4*C*49*Hout*Wout for each of 16 modules.
            model = build_model(variant)
            if variant == 'acmix50':
                extra = 4*49*(3*64*32**2 + 4*128*16**2 + 6*256*8**2 + 3*512*4**2)
            base = estimate_model_flops(model, (3, 64, 64), 'cpu')
            self.assertEqual(estimate_flops(model, 'cpu'), base + extra)

    def test_checkpoint_roundtrip(self):
        x = torch.randn(1, 3, 64, 64)
        for variant in ('acmix50', 'conformer_tiny'):
            model = build_model(variant).eval()
            clone = build_model(variant, 43).eval()
            clone.load_state_dict(model.state_dict(), strict=True)
            with torch.no_grad():
                torch.testing.assert_close(clone(x), model(x), rtol=0, atol=0)

    def test_previous_five_sources_are_comparison_only(self):
        for variant in ('resnet50', 'botnet50', 'preact50', 'alternet50', 'coatnet0'):
            summary = dict(variant=variant, source_hashes=dict(BASELINE_EC0D06A))
            self.assertEqual(experiment.comparable_source(summary), experiment.source_hashes())
        for variant in ('acmix50', 'conformer_tiny'):
            summary = dict(variant=variant, source_hashes=dict(BASELINE_EC0D06A))
            self.assertNotEqual(experiment.comparable_source(summary), experiment.source_hashes())
        summary['variant'] = 'coatnet0'
        summary['source_hashes']['main.py'] = 'unknown'
        self.assertNotEqual(experiment.comparable_source(summary), experiment.source_hashes())

    def test_shared_100_epoch_config(self):
        old = yaml.safe_load((experiment.ROOT/'configs/tiny_imagenet_coatnet.yaml').read_text())
        new = yaml.safe_load((experiment.ROOT/'configs/tiny_imagenet_fusion.yaml').read_text())
        self.assertEqual(old, new)

    def test_fp16_attention_islands(self):
        from models.conformer_author import Attention
        inner = Attention(8, num_heads=2, qkv_bias=True).eval()
        with torch.no_grad():
            inner.qkv.weight.fill_(0.25)
            inner.qkv.bias.zero_()
        safe = ConformerAttentionFP32(inner)
        x = torch.full((2, 17, 8), 200.)
        with torch.autocast('cpu', dtype=torch.float16):
            self.assertFalse(torch.isfinite(inner(x)).all())
            self.assertTrue(torch.isfinite(safe(x)).all())
        acmix = ACmix(8, checkpoint_attention=False).eval()
        with torch.autocast('cpu', dtype=torch.float16):
            y = acmix(torch.full((2, 8, 8, 8), 200.))
        self.assertEqual(y.dtype, torch.float32)
        self.assertTrue(torch.isfinite(y).all())

    def test_conformer_real_training_evaluation_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = root/'images'/'class0'
            images.mkdir(parents=True)
            for i in range(6):
                Image.new('RGB', (64, 64), (i*30, 50, 80)).save(images/f'{i}.png')
            folder = ImageFolder(root/'images')
            def fake_data(config, seed):
                return {k: UtilityDataset(folder, ids, k, seed) for k, ids in
                        [('train', [0, 1]), ('selection', [2, 3]), ('audit', [4, 5])]}, {}, 'synthetic'
            config = yaml.safe_load(experiment.DEFAULT_CONFIG.read_text())
            config['dataset'].update(batch_size=2, num_workers=0)
            config['training'].update(epochs=2, mixed_precision=False, enable_tf32=False)
            config['training']['scheduler']['warmup_epochs'] = 1
            config['profiling'].update(warmup_iterations=0, benchmark_iterations=1)
            with patch('datasets.resnet_tiny.make_datasets', fake_data):
                self.assertFalse(experiment.train_run(config, 'conformer_tiny', 42, root/'run', deadline=0))
                self.assertTrue(experiment.train_run(None, None, None, root/'run', resume=True))
                self.assertTrue((root/'run'/'summary.json').exists())
                state = torch.load(root/'run'/'last.pt', weights_only=False)
                self.assertEqual(state['epoch'], 2)
                self.assertEqual(state['metadata']['architecture']['prediction'],
                                 'sum of both head logits; NLL computed on summed logits')


if __name__ == '__main__':
    unittest.main()
