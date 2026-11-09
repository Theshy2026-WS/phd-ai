import copy
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from models.vit_augreg import AugRegFusion, build_vit
from pairwise_resolver_model import (PairwiseResolver, PairwiseResolverHead,
                                     RESOLVER_VARIANTS, stable_candidates)


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 4, 1), nn.BatchNorm2d(4), nn.ReLU())
        self.stages = nn.ModuleList([nn.Identity(), nn.Identity(),
                                    nn.Conv2d(4, 2048, 1, stride=4), nn.Identity()])
        self.final_reduction = nn.Identity()
        self.head = nn.Module()
        self.head.bn = nn.BatchNorm2d(2048)
        self.head.relu = nn.ReLU()
        self.head.gap = nn.AdaptiveAvgPool2d(1)
        self.head.dense = nn.Linear(2048, 7)


def tiny_anchor():
    vit = build_vit('vit_small_p8_augreg', seed=42,
        config=dict(image_size=8, patch_size=4, embed_dim=8, depth=6,
                    num_heads=2, num_classes=7, mlp_ratio=2.))
    return AugRegFusion(TinyEncoder(), vit, variant='concat', frozen=False)


class PairwiseResolverModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def data(self):
        torch.manual_seed(17)
        features, tokens = torch.randn(9, 12), torch.randn(9, 6, 8)
        weight, bias = torch.randn(11, 12), torch.randn(11)
        return features, tokens, F.linear(features, weight, bias), weight

    def head(self, variant='pairwise'):
        return PairwiseResolverHead(12, 8, 6, variant, topk=5, width=8, heads=2)

    def test_stable_candidate_ties_match_class_index(self):
        logits = torch.tensor([[1., 2., 2., 2., 0., 2., 2.]])
        self.assertEqual(stable_candidates(logits, 5).tolist(), [[1, 2, 3, 5, 6]])
        self.assertEqual(stable_candidates(torch.zeros(1, 7), 5).tolist(), [[0, 1, 2, 3, 4]])

    def test_pair_reversal_is_antisymmetric_for_every_variant(self):
        features, tokens, _, _ = self.data()
        evidence = torch.randn(features.shape[0], 10, 12)
        for variant in RESOLVER_VARIANTS:
            with self.subTest(variant=variant):
                head = self.head(variant)
                forward, _ = head.pair_scores(evidence, tokens)
                reverse, _ = head.pair_scores(-evidence, tokens)
                torch.testing.assert_close(forward, -reverse, rtol=0, atol=2e-9)

    def test_signed_attention_cancels_constant_value_component(self):
        head = self.head().double()
        evidence = torch.randn(3, 10, 12, dtype=torch.double)
        tokens = torch.randn(3, 6, 8, dtype=torch.double)
        _, maps = head.signed_attention(evidence, tokens, collect=True)
        signed = maps['difference']
        constant = torch.randn(3, 2, 1, 4, dtype=torch.double).expand(-1, -1, 6, -1)
        torch.testing.assert_close(signed @ constant, torch.zeros(3, 2, 10, 4, dtype=torch.double),
                                   rtol=0, atol=2e-15)

    def test_delta_zero_sum_bound_support_and_candidate_coverage(self):
        features, tokens, logits, weight = self.data()
        for variant in RESOLVER_VARIANTS:
            with self.subTest(variant=variant):
                head = self.head(variant)
                head.set_train_patch_mean(tokens.mean(0))
                # Exercise a material correction, rather than near-zero init only.
                with torch.no_grad():
                    head.raw_eta.fill_(4.)
                    if variant == 'mlp': head.mlp[-1].weight.mul_(50000.)
                    else: head.output.weight.mul_(50000.)
                details = head(features, tokens, logits, weight, collect=True)
                delta, candidates = details['delta'], details['candidates']
                self.assertLess(delta.sum(-1).abs().max().item(), 1e-6)
                self.assertLessEqual(delta.abs().max().item(), head.eta.item()+1e-6)
                mask = torch.zeros_like(delta, dtype=torch.bool).scatter(1, candidates, True)
                self.assertTrue(torch.all(delta[~mask] == 0))
                winner = details['logits'].argmax(-1)
                self.assertTrue((winner[:, None] == candidates).any(-1).all())
                margin = logits.topk(2).values.diff(dim=1).abs().flatten()
                protected = margin > 2*head.eta
                self.assertTrue(torch.equal(winner[protected], logits.argmax(-1)[protected]))

    def test_true_topk_mean_proof_and_large_random_updates(self):
        torch.manual_seed(99)
        logits = torch.randn(2048, 200)
        candidates = stable_candidates(logits, 5)
        local = torch.zeros(2048, 5)
        left, right = torch.triu_indices(5, 5, 1)
        updates = torch.randn(2048, 10)*100
        local.scatter_add_(1, left[None].expand_as(updates), updates)
        local.scatter_add_(1, right[None].expand_as(updates), -updates)
        altered = logits + torch.zeros_like(logits).scatter(1, candidates, local)
        self.assertTrue((altered.argmax(-1)[:, None] == candidates).any(-1).all())
        tied = torch.zeros(20, 200)
        candidates = stable_candidates(tied, 5)
        altered = tied+torch.zeros_like(tied).scatter(1, candidates, local[:20])
        self.assertTrue((altered.argmax(-1)[:, None] == candidates).any(-1).all())

    def test_margin_protection_for_adversarial_signed_pair_updates(self):
        eta, topk = 3., 5
        logits = torch.tensor([[30., 20., 19., 18., 17., 16.]])
        candidates = stable_candidates(logits, topk)
        local = torch.tensor([[-eta, eta, 0., 0., 0.]])
        final = logits+torch.zeros_like(logits).scatter(1, candidates, local)
        self.assertGreater((logits[:, 0]-logits[:, 1]).item(), 2*eta)
        self.assertEqual(final.argmax(-1).item(), 0)

    def test_soft_label_pair_gradient_matches_formula(self):
        torch.manual_seed(12)
        logits = torch.randn(2, 7, dtype=torch.double)
        target = torch.rand(2, 7, dtype=torch.double)
        target /= target.sum(-1, keepdim=True)
        scores = torch.randn(2, 10, dtype=torch.double, requires_grad=True)
        eta, topk = .7, 5
        candidates = stable_candidates(logits, topk)
        left, right = torch.triu_indices(topk, topk, 1)
        updates = eta/(topk-1)*scores.tanh()
        local = torch.zeros(2, topk, dtype=torch.double)
        local.scatter_add_(1, left[None].expand_as(updates), updates)
        local.scatter_add_(1, right[None].expand_as(updates), -updates)
        final = logits+torch.zeros_like(logits).scatter(1, candidates, local)
        loss = -(target*final.log_softmax(-1)).sum()  # Sum, not batch mean.
        derivative, = torch.autograd.grad(loss, scores)
        difference = (final.softmax(-1)-target).gather(1, candidates)
        expected = eta/(topk-1)*(difference[:, left]-difference[:, right])*(1-scores.tanh()**2)
        torch.testing.assert_close(derivative, expected, rtol=1e-12, atol=1e-14)

    def test_every_trainable_parameter_gets_finite_nonzero_gradient(self):
        features, tokens, logits, weight = self.data()
        target = torch.arange(features.shape[0]) % logits.shape[-1]
        for variant in RESOLVER_VARIANTS:
            with self.subTest(variant=variant):
                head = self.head(variant)
                head.set_train_patch_mean(tokens.mean(0))
                F.cross_entropy(head(features, tokens, logits, weight), target).backward()
                for name, parameter in head.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                    self.assertGreater(parameter.grad.abs().sum().item(), 0, name)

    def test_mean_tokens_replaces_only_patch_information(self):
        features, tokens, logits, weight = self.data()
        head = self.head('mean_tokens')
        with self.assertRaisesRegex(RuntimeError, 'training-only'):
            head(features, tokens, logits, weight)
        mean = tokens.mean(0)
        head.set_train_patch_mean(mean)
        first = head(features, tokens, logits, weight, collect=True)
        second = head(features, tokens*100+50, logits, weight, collect=True)
        torch.testing.assert_close(first['logits'], second['logits'], rtol=0, atol=0)
        torch.testing.assert_close(first['used_patch_tokens'], mean[None].expand_as(tokens))
        torch.testing.assert_close(first['features'], features)
        with self.assertRaisesRegex(ValueError, 'finite'):
            head.set_train_patch_mean(torch.full_like(mean, float('nan')))

    def test_full_anchor_forward_matches_cached_and_original_paths(self):
        torch.manual_seed(7)
        anchor = tiny_anchor().eval()
        images = torch.randn(2, 3, 8, 8)
        with torch.no_grad():
            original = anchor(images)
        model = PairwiseResolver(anchor, width=8, heads=2, intermediate_block=6)
        cached = model.extract_anchor(images)
        torch.testing.assert_close(cached['logits'], original, rtol=0, atol=0)
        self.assertEqual(tuple(cached['patch_tokens'].shape), (2, 4, 8))
        self.assertFalse(cached['features'].requires_grad)
        from_cache = model.forward_cached(cached['features'], cached['patch_tokens'], cached['logits'])
        torch.testing.assert_close(model(images), from_cache, rtol=0, atol=0)

    def test_frozen_anchor_bn_dropout_and_weights_survive_training(self):
        torch.manual_seed(7)
        model = PairwiseResolver(tiny_anchor(), width=8, heads=2)
        before = copy.deepcopy(model.anchor.state_dict())
        model.train()
        self.assertFalse(any(module.training for module in model.anchor.modules()))
        self.assertTrue(model.resolver.training)
        self.assertFalse(any(p.requires_grad for p in model.anchor.parameters()))
        images = torch.randn(2, 3, 8, 8)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad])
        F.cross_entropy(model(images), torch.tensor([1, 4])).backward()
        optimizer.step()
        for key, value in model.anchor.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)
        self.assertTrue(all(p.grad is None for p in model.anchor.parameters()))

    def test_checkpoint_roundtrip_includes_mean_and_eta(self):
        torch.manual_seed(7)
        model = PairwiseResolver(tiny_anchor(), variant='mean_tokens', width=8, heads=2)
        model.set_train_patch_mean(torch.randn(4, 8))
        other = PairwiseResolver(tiny_anchor(), variant='mean_tokens', width=8, heads=2)
        other.load_state_dict(copy.deepcopy(model.state_dict()), strict=True)
        images = torch.randn(2, 3, 8, 8)
        torch.testing.assert_close(model(images), other(images), rtol=0, atol=0)
        self.assertTrue(other.resolver.train_patch_mean_ready)

    def test_analysis_modes_restore_and_uniform_removes_signed_update(self):
        model = PairwiseResolver(tiny_anchor(), width=8, heads=2)
        images = torch.randn(2, 3, 8, 8)
        model.set_train_patch_mean(torch.randn(4, 8))
        base = model.extract_anchor(images)['logits']
        for mode in ('anchor', 'uniform'):
            with model.analysis(mode):
                torch.testing.assert_close(model(images), base, rtol=0, atol=0)
            self.assertEqual(model.analysis_mode, 'learned')
        with self.assertRaisesRegex(ValueError, 'Unknown'):
            with model.analysis('bad'):
                pass

    def test_real_default_parameter_budget_is_matched(self):
        attention = PairwiseResolverHead()
        mlp = PairwiseResolverHead(variant='mlp')
        attention_count = sum(p.numel() for p in attention.parameters())
        mlp_count = sum(p.numel() for p in mlp.parameters())
        self.assertEqual(attention_count, 409729)
        self.assertEqual(mlp_count, 408611)
        self.assertLess(abs(attention_count-mlp_count)/attention_count, .005)


if __name__ == '__main__':
    unittest.main()
