import io
import unittest

import torch

from models.literature_resnet import build_model
from models.spatial_preact import (EvenIndexDecimation, SPATIAL_VARIANTS,
                                   build_spatial_model, spatial_description)


class SpatialPreActTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_reference_identity_and_paired_initial_parameters(self):
        before = torch.get_rng_state().clone()
        reference = build_model('resnet50_preact', 42).eval()
        cnn4 = build_spatial_model('cnn4', 42).eval()
        cnn8 = build_spatial_model('cnn8', 42).eval()
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        expected = reference.state_dict()
        for candidate in (cnn4, cnn8):
            self.assertEqual(list(candidate.state_dict()), list(expected))
            self.assertEqual(sum(p.numel() for p in candidate.parameters()), 23902472)
            for name, value in candidate.state_dict().items():
                torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
            self.assertEqual([len(stage) for stage in candidate.stages], [3, 4, 6, 3])
        x = torch.randn(2, 3, 64, 64)
        with torch.no_grad():
            torch.testing.assert_close(cnn4(x), reference(x), rtol=0, atol=0)

    def test_native_stage_maps_and_consistent_transition_strides(self):
        for variant in SPATIAL_VARIANTS:
            with self.subTest(variant=variant):
                model = build_spatial_model(variant).eval()
                stages, before_head = [], []
                hooks = [stage.register_forward_hook(
                    lambda m, inputs, output: stages.append(list(output.shape[-2:])))
                         for stage in model.stages]
                hooks.append(model.head.register_forward_pre_hook(
                    lambda m, inputs: before_head.append(list(inputs[0].shape[-2:]))))
                with torch.no_grad():
                    result = model(torch.randn(1, 3, 64, 64))
                for hook in hooks:
                    hook.remove()
                expected = 8 if variant == 'cnn8' else 4
                self.assertEqual(stages, [[32, 32], [16, 16], [8, 8], [expected, expected]])
                self.assertEqual(before_head, [[4, 4]])
                self.assertEqual(result.shape, (1, 200))
                transition = model.stages[3][0]
                stride = (1, 1) if variant == 'cnn8' else (2, 2)
                self.assertEqual(transition.conv2[2].stride, stride)
                self.assertEqual(transition.shortcut[0].stride, stride)
                self.assertEqual(spatial_description(variant)['stage_spatial_sizes'],
                                 [shape[0] for shape in stages])

    def test_gradients_optimizer_and_serialized_reload(self):
        for variant in SPATIAL_VARIANTS:
            with self.subTest(variant=variant):
                model = build_spatial_model(variant).train()
                optimizer = torch.optim.SGD(model.parameters(), lr=.001)
                x = torch.randn(2, 3, 64, 64)
                target = torch.tensor([0, 199])
                with torch.autocast('cpu', dtype=torch.bfloat16):
                    prediction = model(x)
                loss = torch.nn.functional.cross_entropy(prediction.float(), target)
                loss.backward()
                for name, parameter in model.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                original = model.stem.weight.detach().clone()
                optimizer.step()
                self.assertFalse(torch.equal(original, model.stem.weight))
                buffer = io.BytesIO()
                torch.save(model.state_dict(), buffer)
                buffer.seek(0)
                restored = build_spatial_model(variant, 43).eval()
                restored.load_state_dict(torch.load(buffer, map_location='cpu', weights_only=True), strict=True)
                model.eval()
                with torch.no_grad():
                    torch.testing.assert_close(restored(x), model(x), rtol=0, atol=0)

    def test_decimation_retains_even_samples_without_averaging(self):
        x = torch.arange(16., requires_grad=True).reshape(1, 1, 4, 4)
        reduction = EvenIndexDecimation()
        result = reduction(x)
        torch.testing.assert_close(result, torch.tensor([[[[0., 2.], [8., 10.]]]]), rtol=0, atol=0)
        self.assertEqual(list(reduction.parameters()), [])
        self.assertEqual(reduction.state_dict(), {})

    def test_unknown_variant_is_rejected(self):
        for function in (build_spatial_model, spatial_description):
            with self.assertRaisesRegex(ValueError, 'Unknown spatial variant'):
                function('unsupported')


if __name__ == '__main__':
    unittest.main()
