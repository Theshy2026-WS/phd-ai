import argparse
from copy import deepcopy
import unittest
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import torch
from torch import nn
import timm
import yaml
from PIL import Image
from torchvision.datasets import ImageFolder
from datasets.utility_tiny import UtilityDataset
from timm.layers import trunc_normal_
from timm.layers.grn import GlobalResponseNorm
from timm.models.convmixer import ConvMixer
from timm.models.metaformer import Attention, SepConv

from models.literature_resnet import build_model, architecture_description
from models.literature_mixers import MIXER_VARIANTS, MetaFormerAttentionFP32
from utils.literature_profiling import estimate_flops
from utils.profiling import estimate_model_flops
from utils.literature_fusion_compat import BASELINE_0C160B3, BASELINE_392A385
import literature_experiment as experiment
from overnight_twostage import training_command


def reference(variant, seed=42):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        if variant == 'convmixer768_tiny64':
            return ConvMixer(768, 32, kernel_size=7, patch_size=4, act_layer=nn.ReLU,
                             num_classes=200, drop_rate=0.)
        name = dict(convnextv2_tiny64='convnextv2_tiny', convformer_s18_tiny64='convformer_s18',
                    caformer_s18_tiny64='caformer_s18')[variant]
        model = timm.create_model(name, pretrained=False, num_classes=200, drop_rate=0., drop_path_rate=0.)
        if variant == 'convnextv2_tiny64':
            model.stem[0] = nn.Conv2d(3, 96, 2, stride=2)
            trunc_normal_(model.stem[0].weight, std=.02)
            nn.init.zeros_(model.stem[0].bias)
        else:
            model.stem.conv.stride, model.stem.conv.padding = (2, 2), (3, 3)
            for m in model.modules():
                if isinstance(m, Attention):
                    m.fused_attn = False
        return model


class MixerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_shapes_gradients_optimizer_and_reload(self):
        expected = [20419016, 28016840, 25135248, 24702456]
        for variant, params in zip(MIXER_VARIANTS, expected):
            with self.subTest(variant=variant):
                model = build_model(variant)
                self.assertEqual(sum(p.numel() for p in model.parameters()), params)
                modules = [model.blocks] if variant.startswith('convmixer') else list(model.stages)
                sizes, tokens = [], []
                hooks = [m.register_forward_hook(lambda m, inp, out: sizes.append(out.shape[-1])) for m in modules]
                hooks += [m.register_forward_pre_hook(lambda m, inp: tokens.append(inp[0].shape[1]))
                          for m in model.modules() if isinstance(m, MetaFormerAttentionFP32)]
                optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005)
                with torch.autocast('cpu', dtype=torch.bfloat16):
                    logits = model(torch.randn(2, 3, 64, 64))
                    loss = nn.functional.cross_entropy(logits, torch.tensor([0, 199]))
                loss.backward()
                for name, p in model.named_parameters():
                    if p.requires_grad:
                        self.assertIsNotNone(p.grad, name)
                        self.assertTrue(torch.isfinite(p.grad).all(), name)
                optimizer.step()
                for h in hooks:
                    h.remove()
                meta = architecture_description(variant)
                self.assertEqual(logits.shape, (2, 200))
                self.assertEqual(sizes, meta['stage_spatial_sizes'])
                self.assertEqual(tokens, [s*s for s in meta['attention_spatial_sizes']])
                if variant == 'convnextv2_tiny64':
                    self.assertEqual(sum(isinstance(m, GlobalResponseNorm) for m in model.modules()), 18)
                if variant == 'convformer_s18_tiny64':
                    self.assertEqual(sum(isinstance(m, SepConv) for m in model.modules()), 18)
                if variant == 'caformer_s18_tiny64':
                    self.assertEqual(sum(isinstance(m, SepConv) for m in model.modules()), 6)
                model.eval()
                clone = build_model(variant, 43).eval()
                clone.load_state_dict(model.state_dict(), strict=True)
                with torch.no_grad():
                    x = torch.randn(1, 3, 64, 64)
                    torch.testing.assert_close(model(x), clone(x), rtol=0, atol=0)
                del optimizer, clone, model, logits, loss

    def test_fp32_matches_unwrapped_timm_and_never_downloads(self):
        for variant in MIXER_VARIANTS:
            with self.subTest(variant=variant), patch('torch.hub.download_url_to_file', side_effect=AssertionError('download')):
                model, ref = build_model(variant).eval(), reference(variant).eval()
                with torch.no_grad():
                    x = torch.randn(1, 3, 64, 64)
                    torch.testing.assert_close(model(x), ref(x), rtol=0, atol=0)

    def test_flops_include_attention(self):
        model = build_model('caformer_s18_tiny64')
        base = estimate_model_flops(model, (3, 64, 64), 'cpu')
        extra = 4 * (9 * 64**2 * 320 + 3 * 16**2 * 512)
        self.assertEqual(estimate_flops(model, 'cpu'), base + extra)
        for variant in MIXER_VARIANTS[:-1]:
            model = build_model(variant)
            self.assertEqual(estimate_flops(model, 'cpu'), estimate_model_flops(model, (3, 64, 64), 'cpu'))

    def test_fp16_attention_overflow_protection(self):
        inner = Attention(8, head_dim=4).eval()
        inner.fused_attn = False
        with torch.no_grad():
            inner.qkv.weight.fill_(0.25)
        safe = MetaFormerAttentionFP32(deepcopy(inner))
        x = torch.full((2, 64, 8), 200.)
        with torch.autocast('cpu', dtype=torch.float16):
            self.assertFalse(torch.isfinite(inner(x)).all())
            self.assertTrue(torch.isfinite(safe(x)).all())

    def test_old_comparison_whitelist_and_overnight_command(self):
        summary = dict(variant='botnet50_preact_cnn', source_hashes=BASELINE_0C160B3)
        self.assertEqual(experiment.comparable_source(summary), experiment.source_hashes())
        for v in MIXER_VARIANTS:
            for hashes in (BASELINE_0C160B3, BASELINE_392A385):
                self.assertNotEqual(experiment.comparable_source(dict(variant=v, source_hashes=hashes)),
                                    experiment.source_hashes())
        args = argparse.Namespace(resume=None, variants=list(MIXER_VARIANTS), epochs=100)
        command = training_command(args)
        self.assertEqual(command[command.index('--variants')+1:command.index('--config')], list(MIXER_VARIANTS))
        self.assertEqual(command[-4:], ['--seeds', '42', '--epochs', '100'])

    def test_caformer_training_evaluation_and_resume(self):
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
                self.assertFalse(experiment.train_run(config, 'caformer_s18_tiny64', 42, root/'run', deadline=0))
                self.assertTrue(experiment.train_run(None, None, None, root/'run', resume=True))
            summary = json.loads((root/'run'/'summary.json').read_text())
            self.assertEqual(summary['completed_epochs'], 2)
            self.assertEqual(summary['architecture']['attention_blocks'], 12)
            self.assertEqual(summary['evaluation_precision'], 'float32')


if __name__ == '__main__':
    unittest.main()
