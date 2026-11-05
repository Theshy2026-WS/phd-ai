import copy
import gc
import unittest

import torch
from torch import nn

from models.attention_correction import state_digest
from models.spatial_preact import build_spatial_model
from models.vit import ViT
from models.vit_followup import (VARIANTS, VIT_CONFIGURATION, ChunkedGlobalAttention,
    CrossTransformerBlock, FP32MultiheadAttention, build_followup_model, description)


class ViTFollowupModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def tearDown(self):
        gc.collect()

    def test_full_vit_has_patch_cls_position_and_five_attention_mlp_blocks(self):
        before = torch.get_rng_state().clone()
        model = build_followup_model('vit').eval()
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertEqual(model.num_patches, 256)
        self.assertEqual(model.position_embedding.shape, (1, 257, 224))
        self.assertEqual(len(model.blocks), 5)
        self.assertEqual(model.blocks[0].mlp[0].out_features, 896)
        self.assertIsInstance(model.blocks[0].attention, FP32MultiheadAttention)
        x = torch.randn(2, 3, 64, 64)
        with torch.inference_mode():
            values = model.forward_decomposed(x)
            features = model.forward_features(x)
            self.assertEqual(values['logits'].shape, (2, 200))
            self.assertEqual(features.shape, (2, 224))
            torch.testing.assert_close(values['logits'], model.classifier(features))
            self.assertTrue(torch.isfinite(values['logits']).all())
            self.assertTrue((values['attention_entropy'] > 0).all())
        legacy = ViT(**VIT_CONFIGURATION).eval()
        legacy.load_state_dict(model.state_dict(), strict=True)
        with torch.inference_mode():
            torch.testing.assert_close(model(x), legacy(x), atol=1e-5, rtol=1e-5)

    def test_fp32_attention_matches_pytorch_self_attention_and_uniform_mean(self):
        original = nn.MultiheadAttention(16, 4, batch_first=True, dropout=0.)
        explicit = FP32MultiheadAttention(16, 4, batch_first=True, dropout=0.)
        explicit.load_state_dict(original.state_dict())
        x = torch.randn(2, 7, 16)
        expected, _ = original(x, x, x, need_weights=True)
        actual, _ = explicit(x, x, x, need_weights=True)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
        explicit.analysis_mode = 'uniform'
        actual, weights = explicit(x, x, x, need_weights=True)
        self.assertTrue(torch.equal(weights, torch.full_like(weights, 1/7)))
        v = nn.functional.linear(x, explicit.in_proj_weight[32:], explicit.in_proj_bias[32:])
        torch.testing.assert_close(actual,
            explicit.out_proj(v.mean(1, keepdim=True).expand(-1, 7, -1)))

    def test_dual_uses_trained_vit_and_cnn_without_shared_or_unused_heads(self):
        parent = build_spatial_model('cnn8', 42).eval()
        vit = build_followup_model('vit', seed=41).eval()
        # A real parent digest must be the same after feature-head removal,
        # excluding precisely the removed original linear classifier.
        model = build_followup_model('dual_fusion', parent.state_dict(), vit.state_dict()).eval()
        self.assertIsInstance(model.encoder.head.dense, nn.Identity)
        self.assertIsInstance(model.vit.classifier, nn.Identity)
        self.assertFalse({id(p) for p in model.encoder.parameters()} & {id(p) for p in model.vit.parameters()})
        self.assertTrue(all(p.requires_grad for p in model.parameters()))
        for key, value in model.vit.state_dict().items():
            self.assertTrue(torch.equal(value, vit.state_dict()[key]), key)
        x = torch.randn(2, 3, 64, 64)
        with torch.inference_mode():
            with model.analysis('cnn_only'):
                torch.testing.assert_close(model(x), parent(x), atol=1e-6, rtol=1e-6)
            values = model.forward_decomposed(x)
            self.assertEqual(values['cnn_features'].shape, (2, 2048))
            self.assertEqual(values['vit_features'].shape, (2, 224))
            torch.testing.assert_close(values['vit_features'], vit.forward_features(x))
            with model.analysis('vit_only'):
                only_vit = model(x)
            self.assertFalse(torch.equal(only_vit, values['logits']))
        model.train()
        loss = nn.functional.cross_entropy(model(x), torch.tensor([1, 199]))
        loss.backward()
        for parameter in (model.encoder.stem.weight, model.vit.patch_embedding.weight,
                          model.vit.blocks[0].mlp[0].weight, model.fusion_classifier.weight):
            self.assertIsNotNone(parameter.grad)
            self.assertGreater(float(parameter.grad.abs().sum()), 0.)
        missing = [name for name, p in model.named_parameters() if p.grad is None]
        self.assertEqual(missing, [])

    def test_chunked_global_attention_matches_dense_values_and_gradients(self):
        chunked = ChunkedGlobalAttention(6, 16, 4, query_chunk_size=7).train()
        dense = copy.deepcopy(chunked)
        dense.query_chunk_size = 1000
        x1 = torch.randn(2, 6, 5, 7, requires_grad=True)
        x2 = x1.detach().clone().requires_grad_()
        y1, stats1 = chunked(x1, collect=True)
        y2, stats2 = dense(x2, collect=True)
        torch.testing.assert_close(y1, y2, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(stats1['attention_entropy'], stats2['attention_entropy'])
        torch.testing.assert_close(stats1['attention_peak'], stats2['attention_peak'])
        weights = torch.randn_like(y1)
        (y1*weights).sum().backward()
        (y2*weights).sum().backward()
        torch.testing.assert_close(x1.grad, x2.grad, atol=2e-6, rtol=2e-5)
        for (name1, p1), (name2, p2) in zip(chunked.named_parameters(), dense.named_parameters()):
            self.assertEqual(name1, name2)
            torch.testing.assert_close(p1.grad, p2.grad, atol=1e-5, rtol=2e-4)

    def test_true64_shallow_qkv_token_count_and_exact_parent_removal(self):
        parent = build_spatial_model('cnn8', 42).eval()
        model = build_followup_model('shallow64', parent.state_dict(), width=16,
            heads=4, query_chunk_size=128).eval()
        x = torch.randn(1, 3, 64, 64)
        shapes = []
        hooks = [m.register_forward_pre_hook(lambda module, values: shapes.append(values[0].shape))
                 for m in (model.attention.query, model.attention.key, model.attention.value)]
        try:
            with torch.inference_mode():
                actual = model(x)
        finally:
            for hook in hooks:
                hook.remove()
        self.assertEqual(actual.shape, (1, 200))
        self.assertEqual([tuple(s) for s in shapes], [(1, 4096, 64)]*3)
        self.assertEqual(model.attention.last_token_count, 4096)
        self.assertEqual(model.encoder.stem.stride, (2, 2))
        with torch.inference_mode():
            with model.analysis('removed'):
                self.assertTrue(torch.equal(model(x), parent(x)))
            with model.analysis('uniform'):
                values = model.forward_decomposed(x)
                torch.testing.assert_close(values['attention_entropy'], torch.ones(1))
                torch.testing.assert_close(values['attention_peak'], torch.tensor([1/4096]))
        self.assertGreater(model.attention_flops_per_image, 4*4096*4096*16)

    def test_cross_transformer_mlp_has_two_complete_blocks_and_receives_gradient(self):
        model = build_followup_model('cross16_mlp').train()
        self.assertEqual(len(model.blocks), 2)
        self.assertTrue(all(isinstance(block, CrossTransformerBlock) for block in model.blocks))
        self.assertEqual(model.blocks[0].mlp[0].out_features, 512)
        x = torch.randn(2, 3, 64, 64)
        output = model(x)
        nn.functional.cross_entropy(output, torch.tensor([17, 59])).backward()
        for block in model.blocks:
            for parameter in (block.query.weight, block.key.weight, block.value.weight,
                              block.mlp[0].weight, block.mlp[2].weight):
                self.assertGreater(float(parameter.grad.abs().sum()), 0.)
        model.eval()
        digest = state_digest(model)
        with torch.inference_mode():
            normal = model(x)
            with model.analysis('no_mlp'):
                ablated = model(x)
        self.assertFalse(torch.equal(normal, ablated))
        self.assertEqual(digest, state_digest(model))
        self.assertEqual(model.analysis_mode, 'learned')

    def test_true64_training_backpropagates_to_all_qkv_and_shared_cnn_stem(self):
        model = build_followup_model('shallow64', width=16,
            heads=4, query_chunk_size=128).train()
        x = torch.randn(1, 3, 64, 64)
        loss = nn.functional.cross_entropy(model(x), torch.tensor([19]))
        loss.backward()
        for parameter in (model.encoder.stem.weight, model.attention.query.weight,
                          model.attention.key.weight, model.attention.value.weight,
                          model.output.weight, model.gamma):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.abs().sum()), 0.)

    def test_invalid_modes_dimensions_and_state_loads_are_rejected(self):
        for variant in VARIANTS:
            model = build_followup_model(variant, width=16)
            with self.assertRaises(ValueError):
                with model.analysis('unknown'):
                    pass
            with self.assertRaisesRegex(RuntimeError, 'fixture'):
                with model.analysis('learned'):
                    raise RuntimeError('fixture')
            self.assertEqual(model.analysis_mode, 'learned')
            self.assertEqual(description(variant)['name'], variant)
            del model
        with self.assertRaises(ValueError):
            build_followup_model('missing')
        with self.assertRaises(ValueError):
            build_followup_model('shallow64', width=127, heads=4)
        with self.assertRaises(RuntimeError):
            build_followup_model('dual_fusion', vit_state={'bad': torch.zeros(1)})


if __name__ == '__main__':
    unittest.main()
