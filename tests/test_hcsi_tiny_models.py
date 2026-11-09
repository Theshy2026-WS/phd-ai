import copy
import math
import unittest

import numpy as np
import torch
from torch import nn

from hcsi_tiny_models import HCSICrossGate, HCSITinyFusion, CNNMLPControl
from models.vit_augreg import ViTAugReg
from models.vit_followup import _cnn_features, _cnn_pool


torch.set_num_threads(1)


class SmallHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dense = nn.Linear(16, 7)


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(3, 4, 3, stride=2, padding=1)
        self.stages = nn.ModuleList([nn.Sequential(nn.Conv2d(a, b, 3, stride=s, padding=1),
            nn.BatchNorm2d(b), nn.ReLU()) for a, b, s in ((4, 4, 1), (4, 8, 2), (8, 12, 2), (12, 16, 1))])
        self.head = SmallHead()
        self.final_reduction = nn.Identity()


def encoders():
    return SmallCNN(), ViTAugReg(dict(input_channels=3, num_classes=7, image_size=16,
        patch_size=4, embed_dim=8, depth=4, num_heads=2, mlp_ratio=2.,
        dropout=.1, attention_dropout=0., drop_path=.1, norm_eps=1e-6))


def build(variant='hcsi', seed=42):
    torch.manual_seed(seed)
    encoder, vit = encoders()
    return HCSITinyFusion(encoder, vit, variant=variant, descriptor_dim=4,
        cnn_channels=(4, 8, 12, 16), hidden=(12, 8))


def array(value):
    return value.detach().cpu().double().numpy()


def reference_gate_numpy(module, cnn, patches):
    """Independent NumPy evaluation of the author's stage equations."""
    x, t = array(cnn), array(patches)
    b, _, h, w = x.shape
    side = math.isqrt(t.shape[1])
    def normalize(value):
        return value/(np.linalg.norm(value, axis=-1, keepdims=True)+1e-6)
    def coordinates(height, width):
        xx, yy = np.meshgrid(np.linspace(-1, 1, width), np.linspace(-1, 1, height))
        return np.stack((xx, yy), axis=-1).reshape(1, height*width, 2)
    def softmax(value):
        exp = np.exp(value-value.max(axis=-1, keepdims=True))
        return exp/exp.sum(axis=-1, keepdims=True)
    def layernorm(value):
        return (value-value.mean(-1, keepdims=True))/np.sqrt(value.var(-1, keepdims=True)+1e-5)
    dc = normalize(np.einsum('oc,bchw->bhwo', array(module.cnn_projection.weight)[:, :, 0, 0], x)
        .reshape(b, h*w, -1))
    dt = normalize(t @ array(module.vit_projection.weight).T)
    pc = normalize(coordinates(h, w) @ array(module.cnn_position.weight).T)
    pt = normalize(coordinates(side, side) @ array(module.vit_position.weight).T)
    qc, qt = normalize(dc+module.beta_pos*pc), normalize(dt+module.beta_pos*pt)
    sim = (qc @ qt.transpose(0, 2, 1))/(module.temperature*math.sqrt(module.descriptor_dim))
    ct, tc = softmax(sim), softmax(sim.transpose(0, 2, 1))
    gc = 1/(1+np.exp(-(qc*layernorm(ct @ qt)).sum(-1)))
    gt = 1/(1+np.exp(-(qt*layernorm(tc @ qc)).sum(-1)))
    return x*gc.reshape(b, 1, h, w), t*gt[:, :, None], ct, tc


class HCSITinyModelTests(unittest.TestCase):
    def test_stage_matches_independent_author_equations_for_unequal_grids(self):
        torch.manual_seed(71)
        module = HCSICrossGate(3, 5, descriptor_dim=4)
        cnn, tokens = torch.randn(2, 3, 2, 3), torch.randn(2, 4, 5)
        actual_cnn, actual_tokens, details = module(cnn, tokens, collect=True)
        expected_cnn, expected_tokens, ct, tc = reference_gate_numpy(module, cnn, tokens)
        np.testing.assert_allclose(array(actual_cnn), expected_cnn, atol=8e-7, rtol=3e-6)
        np.testing.assert_allclose(array(actual_tokens), expected_tokens, atol=8e-7, rtol=3e-6)
        np.testing.assert_allclose(array(details['retrieval_maps']['cnn_to_vit']), ct, atol=2e-7)
        np.testing.assert_allclose(array(details['retrieval_maps']['vit_to_cnn']), tc, atol=2e-7)

    def test_verified_author_defaults_are_not_silently_stabilized(self):
        gate = HCSICrossGate(3, 5)
        self.assertEqual(gate.beta_pos, .2)
        self.assertEqual(gate.temperature, 1.)
        self.assertEqual(gate.descriptor_dim, 64)
        for projection in (gate.cnn_projection, gate.vit_projection, gate.cnn_position, gate.vit_position):
            self.assertIsNone(projection.bias)
        self.assertFalse(gate.context_norm.elementwise_affine)
        cnn, tokens = torch.randn(2, 3, 2, 3), torch.randn(2, 4, 5)
        out_c, out_t, detail = gate(cnn, tokens, collect=True)
        gc, gt = detail['retrieval_maps']['cnn_gate'], detail['retrieval_maps']['vit_gate']
        self.assertTrue(((gc > 0) & (gc < 1)).all())
        self.assertTrue(((gt > 0) & (gt < 1)).all())
        self.assertTrue(torch.equal(out_c, cnn*gc[:, None]))
        self.assertTrue(torch.equal(out_t, tokens*gt.flatten(1)[:, :, None]))

    def test_uniform_is_a_routing_intervention_with_row_normalization(self):
        gate = HCSICrossGate(3, 5, descriptor_dim=4)
        _, _, details = gate(torch.randn(2, 3, 2, 3), torch.randn(2, 4, 5), mode='uniform', collect=True)
        maps = details['retrieval_maps']
        self.assertTrue(torch.equal(maps['cnn_to_vit'], torch.full((2, 6, 4), .25)))
        self.assertTrue(torch.equal(maps['vit_to_cnn'], torch.full((2, 4, 6), 1/6)))
        self.assertTrue(torch.allclose(details['cnn_attention_entropy'], torch.ones(2)))
        self.assertTrue(torch.allclose(details['vit_attention_entropy'], torch.ones(2)))

    def test_invalid_layout_or_temperature_is_rejected(self):
        with self.assertRaises(ValueError): HCSICrossGate(3, 5, temperature=0)
        with self.assertRaises(ValueError): HCSICrossGate(3, 5, beta_pos=float('nan'))
        gate = HCSICrossGate(3, 5)
        with self.assertRaises(ValueError): gate(torch.randn(2, 3, 2, 3), torch.randn(2, 5, 5))
        with self.assertRaises(ValueError): gate(torch.randn(2, 3, 2, 3), torch.randn(3, 4, 5))

    def test_late_mean_pool_matches_original_encoders_without_cls_substitution(self):
        model = build('late_mlp').eval()
        images = torch.randn(3, 3, 16, 16)
        with torch.no_grad():
            _, deep = _cnn_features(model.encoder, images)
            cnn = _cnn_pool(model.encoder, deep)
            vit = model.vit.forward_tokens(images)[:, 1:].mean(1)
            expected = model.classifier(torch.cat((cnn, vit), -1))
            actual = model(images)
        self.assertTrue(torch.equal(actual, expected))

    def test_matched_fusion_heads_share_initial_weights_and_no_dormant_gate_parameters(self):
        late, hcsi = build('late_mlp'), build('hcsi')
        for name, value in late.classifier.state_dict().items():
            self.assertTrue(torch.equal(value, hcsi.classifier.state_dict()[name]))
        self.assertEqual(len(late.interactions), 0)
        self.assertEqual(len(hcsi.interactions), 4)
        self.assertGreater(sum(p.numel() for p in hcsi.parameters()), sum(p.numel() for p in late.parameters()))

    def test_disabling_interactions_recovers_matched_late_model_exactly(self):
        late, hcsi = build('late_mlp').eval(), build('hcsi').eval()
        images = torch.randn(3, 3, 16, 16)
        with torch.no_grad(), hcsi.analysis('no_interaction'):
            self.assertTrue(torch.equal(late(images), hcsi(images)))
        self.assertEqual(hcsi.analysis_mode, 'learned')

    def test_final_patch_gate_affects_vit_readout_and_receives_gradients(self):
        model = build().eval()
        images = torch.randn(3, 3, 16, 16)
        final = model.interactions[-1]
        def remove_last_vit_gate(module, args, result):
            cnn, _, details = result
            return cnn, args[1], details
        baseline = model.extract_features(images)[1]
        hook = final.register_forward_hook(remove_last_vit_gate)
        without_last_gate = model.extract_features(images)[1]
        hook.remove()
        self.assertFalse(torch.allclose(baseline, without_last_gate, atol=1e-6, rtol=1e-5))
        baseline.square().sum().backward()
        self.assertIsNotNone(final.vit_projection.weight.grad)
        self.assertGreater(float(final.vit_projection.weight.grad.abs().max()), 0.)

    def test_all_participating_encoder_and_new_projection_parameters_train(self):
        model = build().train()
        logits = model(torch.randn(4, 3, 16, 16))
        nn.functional.cross_entropy(logits, torch.tensor([0, 1, 2, 3])).backward()
        self.assertTrue(all(parameter.requires_grad for parameter in model.parameters()))
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(float(parameter.grad.abs().max()), 0., name)
        self.assertTrue(model.encoder.training)
        self.assertTrue(model.vit.training)
        self.assertTrue(model.classifier[3].training)

    def test_readonly_diagnostics_leave_parameters_and_buffers_unchanged(self):
        model = build().eval()
        before = copy.deepcopy(model.state_dict())
        with torch.no_grad():
            result = model.forward_decomposed(torch.randn(3, 3, 16, 16))
        self.assertEqual(result['logits'].shape, (3, 7))
        self.assertEqual(result['features_after'].shape, (3, 24))
        for stage in range(1, 5):
            self.assertEqual(result[f'stage{stage}_cnn_gate_mean'].shape, (3,))
            self.assertIn(f'stage{stage}', result['retrieval_maps'])
        for name, value in before.items(): self.assertTrue(torch.equal(value, model.state_dict()[name]), name)

    def test_hcsi_state_roundtrip_retains_inference(self):
        model, other = build().eval(), build(seed=713).eval()
        other.load_state_dict(model.state_dict(), strict=True)
        images = torch.randn(3, 3, 16, 16)
        with torch.no_grad(): self.assertTrue(torch.equal(model(images), other(images)))

    def test_cnn_mlp_control_updates_source_cnn_without_vit_parameters(self):
        encoder, _ = encoders()
        model = CNNMLPControl(encoder, hidden=(12, 8)).train()
        logits = model(torch.randn(4, 3, 16, 16))
        nn.functional.cross_entropy(logits, torch.tensor([0, 1, 2, 3])).backward()
        self.assertEqual(logits.shape, (4, 7))
        self.assertFalse(any('vit' in name for name, _ in model.named_parameters()))
        self.assertTrue(all(p.requires_grad and p.grad is not None for p in model.parameters()))
        model.eval()
        with torch.no_grad(): result = model.forward_decomposed(torch.randn(3, 3, 16, 16))
        self.assertEqual(result['features_after'].shape, (3, 16))
        self.assertEqual(model.attention_flops_per_image, 0)


if __name__ == '__main__': unittest.main()
