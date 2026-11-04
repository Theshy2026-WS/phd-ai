import gc
import io
import unittest

import torch
from torch import nn

from models.attention_correction import state_digest
from models.attention_fusion import (ATTENTION_VARIANTS, VARIANTS, attention_matmul_flops,
                                     build_attention_model, description)
from models.spatial_preact import build_spatial_model


class AttentionFusionModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_all_nine_real_models_keep_encoder_and_phase00_head(self):
        parent = build_spatial_model('cnn8', 42).eval()
        expected_digest = state_digest(parent)
        x = torch.randn(1, 3, 64, 64)
        with torch.inference_mode():
            parent_output = parent(x)
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                before_rng = torch.get_rng_state().clone()
                model = build_attention_model(variant, parent.state_dict()).eval()
                self.assertTrue(torch.equal(before_rng, torch.get_rng_state()))
                self.assertEqual(state_digest(model.encoder), expected_digest)
                self.assertTrue(all(parameter.requires_grad for parameter in model.parameters()))
                with torch.inference_mode():
                    values = model.forward_decomposed(x)
                    self.assertEqual(values['logits'].shape, (1, 200))
                    self.assertTrue(torch.isfinite(values['logits']).all())
                    self.assertEqual(values['features_before'].shape, values['features_after'].shape)
                    self.assertEqual(values['delta'].shape, values['features_before'].shape)
                    if variant in ('conv16', 'self16'):
                        self.assertEqual(values['features_before'].shape[1:], (512, 16, 16))
                    else:
                        self.assertEqual(values['features_before'].shape[1:], (2048, 8, 8))
                    if variant == 'control':
                        self.assertTrue(torch.equal(values['logits'], parent_output))
                        self.assertTrue(torch.equal(model(x), parent_output))
                    else:
                        self.assertAlmostEqual(float(model.gamma), .1)
                    with model.analysis('removed'):
                        self.assertTrue(torch.equal(model(x), parent_output))
                        decomposed = model.forward_decomposed(x)
                        self.assertTrue(torch.equal(decomposed['logits'], parent_output))
                        self.assertTrue(torch.equal(decomposed['features_before'], decomposed['features_after']))
                self.assertEqual(state_digest(model.encoder), expected_digest)
                del model
            gc.collect()

    def test_all_insertions_receive_gradients_and_encoder_remains_trainable(self):
        x = torch.randn(2, 3, 64, 64)
        targets = torch.tensor([1, 199])
        for variant in VARIANTS[1:]:
            with self.subTest(variant=variant):
                model = build_attention_model(variant).train()
                optimizer = torch.optim.SGD(model.parameters(), lr=.001)
                initial_gamma = model.gamma.detach().clone()
                stem = model.encoder.stem.weight.detach().clone()
                for _ in range(2):
                    optimizer.zero_grad(set_to_none=True)
                    loss = nn.functional.cross_entropy(model(x), targets)
                    self.assertTrue(torch.isfinite(loss))
                    loss.backward()
                    self.assertIsNotNone(model.gamma.grad)
                    self.assertGreater(float(model.encoder.stem.weight.grad.abs().sum()), 0.)
                    for name, parameter in model.named_parameters():
                        if not name.startswith('encoder.'):
                            self.assertIsNotNone(parameter.grad, name)
                            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                            # An additive key bias is a common-logit shift,
                            # which cancels algebraically in softmax. Its
                            # gradient can be exactly zero or rounding noise.
                            if not name.endswith('key.bias'):
                                self.assertGreater(float(parameter.grad.abs().sum()), 0., name)
                    optimizer.step()
                self.assertFalse(torch.equal(stem, model.encoder.stem.weight))
                self.assertFalse(torch.equal(initial_gamma, model.gamma))
                del model, optimizer
            gc.collect()

    def test_gated_cross_starts_equivalent_and_content_gate_can_learn(self):
        plain = build_attention_model('cross16').eval()
        gated = build_attention_model('gated_cross16').eval()
        for name, value in plain.state_dict().items():
            self.assertTrue(torch.equal(value, gated.state_dict()[name]), name)
        x = torch.randn(2, 3, 64, 64)
        with torch.inference_mode():
            expected = plain(x)
            result = gated.forward_decomposed(x)
        self.assertTrue(torch.equal(expected, result['logits']))
        self.assertTrue(torch.equal(result['gate'], torch.ones_like(result['gate'])))
        gated.train()
        nn.functional.cross_entropy(gated(x), torch.tensor([3, 19])).backward()
        self.assertGreater(float(gated.gate.weight.grad.abs().sum()), 0.)

    def test_uniform_routing_and_removed_branch_are_read_only_inference_interventions(self):
        model = build_attention_model('cross16').eval()
        x = torch.randn(2, 3, 64, 64)
        model.capture_attention = True
        before = state_digest(model)
        with torch.inference_mode():
            learned = model.forward_decomposed(x)
            with model.analysis('uniform'):
                uniform = model.forward_decomposed(x)
            with model.analysis('removed'):
                removed = model.forward_decomposed(x)
        self.assertFalse(torch.equal(learned['logits'], uniform['logits']))
        self.assertFalse(torch.equal(learned['logits'], removed['logits']))
        self.assertEqual(uniform['retrieval_maps']['deep_reads_source'].shape, (2, 16, 16))
        self.assertTrue(torch.equal(uniform['retrieval_maps']['deep_reads_source'],
                                    torch.full((2, 16, 16), 1/256)))
        torch.testing.assert_close(uniform['attention_entropy'], torch.ones(2))
        self.assertEqual(removed['retrieval_maps'], {})
        self.assertEqual(model.analysis_mode, 'learned')
        self.assertEqual(before, state_digest(model))
        with self.assertRaisesRegex(RuntimeError, 'fixture'):
            with model.analysis('uniform'):
                raise RuntimeError('fixture')
        self.assertEqual(model.analysis_mode, 'learned')

    def test_capture_maps_follow_actual_source_grids_and_default_does_not_retain_maps(self):
        x = torch.randn(1, 3, 64, 64)
        for variant, expected in (('self8', [(8, 8)]), ('self16', [(16, 16)]),
                                  ('cross32', [(32, 32)]), ('bidir16', [(16, 16), (8, 8)])):
            with self.subTest(variant=variant):
                model = build_attention_model(variant).eval()
                with torch.inference_mode():
                    result = model.forward_decomposed(x)
                    self.assertEqual(result['retrieval_maps'], {})
                    model.capture_attention = True
                    result = model.forward_decomposed(x)
                self.assertEqual([tuple(v.shape[-2:]) for v in result['retrieval_maps'].values()], expected)
                self.assertTrue(all(0 <= float(value) <= 1 for value in result['attention_peak']))
                self.assertTrue(all(0 <= float(value) <= 1.00001 for value in result['attention_entropy']))
                del model
            gc.collect()

    def test_state_roundtrip_and_attention_flops_are_explicit(self):
        model = build_attention_model('cross16').eval()
        stream = io.BytesIO()
        torch.save(model.state_dict(), stream)
        stream.seek(0)
        restored = build_attention_model('cross16').eval()
        restored.load_state_dict(torch.load(stream, weights_only=True), strict=True)
        x = torch.randn(1, 3, 64, 64)
        with torch.inference_mode():
            self.assertTrue(torch.equal(model(x), restored(x)))
        self.assertEqual(attention_matmul_flops('cross16'), 4*64*256*128)
        self.assertEqual(attention_matmul_flops('bidir16'), 2*attention_matmul_flops('cross16'))
        self.assertEqual(attention_matmul_flops('conv_cross'), 0)
        self.assertEqual(description('control')['attention_blocks'], 0)
        self.assertEqual(description('bidir16')['attention_blocks'], 2)
        self.assertEqual(set(ATTENTION_VARIANTS), set(VARIANTS)-{'control', 'conv16', 'conv_cross'})
        with self.assertRaises(ValueError):
            build_attention_model('cross16', width=127, heads=4)
        with self.assertRaises(ValueError):
            build_attention_model('unknown')


if __name__ == '__main__':
    unittest.main()
