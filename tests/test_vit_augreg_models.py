import copy
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from models.vit_augreg import (PROFILES, FUSION_VARIANTS, DropPath,
    FP32SelfAttention, ViTAugReg, AugRegFusion, build_vit, description)


def small_vit(seed=42):
    return build_vit('vit_tiny_p4_aug', seed, dict(image_size=8, patch_size=2,
        embed_dim=16, depth=2, num_heads=4, mlp_ratio=2., dropout=.1,
        drop_path=.1))


class TinyHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm2d(2048)
        self.relu = nn.ReLU()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dense = nn.Linear(2048, 200)

    def forward(self, features):
        return self.dense(self.gap(self.relu(self.bn(features))).flatten(1))


class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(3, 4, 1)
        self.stages = nn.Sequential(nn.Conv2d(4, 4, 1),
            nn.AvgPool2d(2), nn.Conv2d(4, 4, 1), nn.Conv2d(4, 2048, 1))
        self.final_reduction = nn.Identity()
        self.head = TinyHead()

    def forward(self, images):
        return self.head(self.final_reduction(self.stages(self.stem(images))))


class ViTAugRegModelsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_profiles_are_from_scratch_twelve_block_candidates(self):
        self.assertEqual(set(PROFILES), {'vit_tiny_p4_aug', 'vit_small_p8_aug', 'vit_small_p8_augreg'})
        for name, config in PROFILES.items():
            self.assertEqual(config['depth'], 12)
            self.assertFalse(description(name)['external_pretraining'])
            self.assertEqual(description(name)['patch_tokens'], 256 if 'tiny' in name else 64)
        self.assertEqual(PROFILES['vit_small_p8_augreg']['drop_path'], .1)

    def test_build_is_seeded_without_consuming_global_rng(self):
        before = torch.get_rng_state().clone()
        first = small_vit()
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        second = small_vit()
        for name, value in first.state_dict().items():
            self.assertTrue(torch.equal(value, second.state_dict()[name]), name)
        self.assertEqual(first.position_embedding.shape, (1, 17, 16))
        self.assertEqual(first.blocks[-1].drop_path1.probability, .1)

    def test_sdpa_matches_dense_values_and_gradients_and_forces_fp32(self):
        attention = FP32SelfAttention(16, 4).eval()
        x = torch.randn(2, 7, 16, requires_grad=True)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            actual = attention(x, collect=True)
        q, k, v = [part.reshape(2, 7, 4, 4).transpose(1, 2)
                   for part in attention.qkv(x).chunk(3, -1)]
        weights = (q @ k.transpose(-1, -2)/2).softmax(-1)
        expected = attention.output((weights @ v).transpose(1, 2).reshape(2, 7, 16))
        self.assertEqual(actual.dtype, torch.float32)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
        left = torch.autograd.grad(actual.square().sum(), x, retain_graph=True)[0]
        right = torch.autograd.grad(expected.square().sum(), x)[0]
        torch.testing.assert_close(left, right, atol=1e-6, rtol=1e-5)
        self.assertTrue((attention.last_statistics['attention_entropy'] > 0).all())

    def test_full_vit_tokens_logits_and_all_blocks_receive_gradients(self):
        model = small_vit().train()
        x = torch.randn(4, 3, 8, 8)
        logits = model(x)
        F.cross_entropy(logits, torch.tensor([1, 2, 3, 4])).backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        model.eval()
        with torch.inference_mode():
            tokens = model.forward_tokens(x)
            self.assertEqual(tokens.shape, (4, 17, 16))
            torch.testing.assert_close(model(x), model.classifier(tokens[:, 0]))
            self.assertEqual(model.forward_decomposed(x)['logits'].shape, (4, 200))
        self.assertEqual(model.attention_flops_per_image, 4*2*17**2*16)

    def test_drop_path_is_per_example_and_disabled_in_eval(self):
        drop = DropPath(.5).train()
        x = torch.ones(100, 4, 5)
        y = drop(x)
        self.assertTrue(((y == 0) | (y == 2)).all())
        self.assertTrue((y == y[:, :1, :1]).all())
        self.assertGreater(float(y.sum()), 0.)
        drop.eval()
        self.assertTrue(torch.equal(x, drop(x)))

    def test_concat_preserves_parent_cnn_only_and_normalized_changes_only_features(self):
        parent, vit = TinyCNN().eval(), small_vit().eval()
        concat = AugRegFusion(copy.deepcopy(parent), copy.deepcopy(vit), 'concat').eval()
        normalized = AugRegFusion(copy.deepcopy(parent), copy.deepcopy(vit), 'normalized').eval()
        normalized.load_state_dict(concat.state_dict(), strict=True)
        x = torch.randn(3, 3, 8, 8)
        with torch.inference_mode():
            with concat.analysis('cnn_only'):
                torch.testing.assert_close(concat(x), parent(x), atol=1e-6, rtol=1e-6)
            c, v, p = concat.extract_features(x)
            expected = normalized.fusion_classifier(torch.cat((normalized.cnn_norm(c), normalized.vit_norm(v)), -1))
            torch.testing.assert_close(normalized(x), expected)
            self.assertFalse(torch.allclose(normalized(x), concat(x)))
        self.assertEqual(sum(p.numel() for p in concat.parameters()),
                         sum(p.numel() for p in normalized.parameters()))
        self.assertFalse(normalized.fusion_description()['normalized_initial_prediction_preserved'])

    def test_frozen_backbones_buffers_are_unchanged_and_fusion_gradients_flow(self):
        x = torch.randn(4, 3, 8, 8)
        for variant in FUSION_VARIANTS:
            with self.subTest(variant=variant):
                model = AugRegFusion(TinyCNN(), small_vit(), variant,
                                     frozen=True, width=16, heads=4).train()
                before = {name: value.clone() for name, value in model.encoder.state_dict().items()}
                self.assertFalse(model.encoder.training)
                self.assertFalse(model.vit.training)
                F.cross_entropy(model(x), torch.tensor([1, 2, 3, 4])).backward()
                for name, parameter in model.named_parameters():
                    if parameter.requires_grad:
                        self.assertIsNotNone(parameter.grad, name)
                        self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                        self.assertGreater(float(parameter.grad.abs().sum()), 0., name)
                    else:
                        self.assertIsNone(parameter.grad, name)
                for name, value in model.encoder.state_dict().items():
                    self.assertTrue(torch.equal(value, before[name]), name)

    def test_joint_concat_updates_both_independent_encoders(self):
        model = AugRegFusion(TinyCNN(), small_vit(), 'concat', frozen=False).train()
        F.cross_entropy(model(torch.randn(4, 3, 8, 8)), torch.tensor([1, 2, 3, 4])).backward()
        self.assertTrue(model.encoder.training)
        self.assertTrue(model.vit.training)
        self.assertGreater(float(model.encoder.stem.weight.grad.abs().sum()), 0.)
        self.assertGreater(float(model.vit.patch_embedding.weight.grad.abs().sum()), 0.)

    def test_bounded_residual_uses_exact_parent_anchor(self):
        parent = TinyCNN().eval()
        x = torch.randn(3, 3, 8, 8)
        for variant in ('residual', 'residual_linear', 'cross_patch', 'mlp_patch'):
            model = AugRegFusion(copy.deepcopy(parent), small_vit(), variant,
                                 width=16, heads=4, residual_bound=.7).eval()
            with torch.inference_mode():
                original = parent(x)
                with model.analysis('cnn_only'):
                    torch.testing.assert_close(model(x), original)
                learned = model(x)
                self.assertLess(float((learned-original).abs().max()), .7)
                self.assertAlmostEqual(float(model.correction_scale), .1, places=6)
                self.assertTrue(torch.isfinite(learned).all())
                model.scale_logit.add_(1.)
                self.assertEqual(model.fusion_description()['initial_correction_scale'], .1)
                if variant == 'mlp_patch':
                    self.assertEqual(model.fusion_description()['mlp_hidden_width'], model.mlp_hidden_width)
                    self.assertEqual(model.fusion_description()['cross_reference_parameters'],
                                     model.cross_reference_parameters)

    def test_cross_attention_and_pooled_mlp_budget_and_token_controls(self):
        cross = AugRegFusion(TinyCNN(), small_vit(), 'cross_patch', width=16, heads=4).eval()
        control = AugRegFusion(TinyCNN(), small_vit(), 'mlp_patch', width=16, heads=4).eval()
        nc = sum(p.numel() for p in cross.parameters() if p.requires_grad)
        nm = sum(p.numel() for p in control.parameters() if p.requires_grad)
        self.assertLess(abs(nc-nm)/nc, .02)  # narrow CPU fixture rounds more coarsely
        self.assertTrue(all(not hasattr(control, name) for name in ('query', 'key', 'value', 'output')))
        self.assertNotIn('uniform', control.analysis_modes)
        c, v, p = cross.extract_features(torch.randn(3, 3, 8, 8))
        with torch.inference_mode():
            normal = cross.forward_from_features(c, v, p)
            uniform = cross.forward_from_features(c, v, p, 'uniform')
            permuted = cross.forward_from_features(c, v, p.flip(1))
            self.assertFalse(torch.equal(normal, uniform))
            # Encoded token order alone is not a valid evidence intervention.
            torch.testing.assert_close(normal, permuted, atol=1e-6, rtol=1e-5)

    def test_feature_reuse_mean_and_mismatch_modes_are_explicit(self):
        model = AugRegFusion(TinyCNN(), small_vit(), 'cross_patch', width=16, heads=4).eval()
        x = torch.randn(4, 3, 8, 8)
        with torch.inference_mode():
            c, v, p = model.extract_features(x)
            torch.testing.assert_close(model(x), model.forward_from_features(c, v, p))
            mean = model.forward_from_features(c, v, p, 'vit_train_mean',
                mean_feature=v.mean(0), mean_patches=p.mean(0))
            explicit = model.forward_from_features(c, v.mean(0).expand_as(v),
                                                   p.mean(0).expand_as(p))
            torch.testing.assert_close(mean, explicit)
            swapped = model.forward_from_features(c, v, p, 'vit_mismatch')
            torch.testing.assert_close(swapped, model.forward_from_features(c, v.roll(1, 0), p.roll(1, 0)))
            self.assertFalse(torch.equal(mean, model(x)))
            with self.assertRaises(ValueError):
                model.forward_from_features(c, v, p, 'vit_train_mean')
            with self.assertRaises(ValueError):
                model.forward_from_features(c[:1], v[:1], p[:1], 'vit_mismatch')
            with self.assertRaises(ValueError):
                with model.analysis('patch_shuffle'):
                    pass
        self.assertEqual(model.analysis_mode, 'learned')

    def test_bad_dimensions_and_probability_settings_are_rejected(self):
        with self.assertRaises(ValueError):
            build_vit('missing')
        with self.assertRaises(ValueError):
            build_vit('vit_tiny_p4_aug', config=dict(embed_dim=17))
        with self.assertRaises(ValueError):
            small_vit()(torch.randn(2, 3, 9, 8))
        with self.assertRaises(ValueError):
            DropPath(1.)
        with self.assertRaises(ValueError):
            AugRegFusion(TinyCNN(), small_vit(), 'cross_patch', width=17, heads=4)


if __name__ == '__main__':
    unittest.main()
