import unittest
import torch
from models.literature_resnet import build_model, architecture_description, BoTAttention
from models.literature_coatnet import CoAtAttentionFP32
from models.literature_twostage import EXPERIMENT_VARIANTS
from utils.literature_profiling import estimate_flops
from utils.literature_fusion_compat import BASELINE_0049D1A, is_pre_twostage_source


class TwoStageTests(unittest.TestCase):
    def test_shapes_gradients_reload_and_profile(self):
        torch.set_num_threads(2)
        for variant in EXPERIMENT_VARIANTS:
            with self.subTest(variant=variant):
                model = build_model(variant)
                shapes, att_shapes = [], []
                hooks = [m.register_forward_hook(lambda m, i, o: shapes.append(o.shape[-1]))
                         for m in model.stages]
                hooks += [m.register_forward_pre_hook(lambda m, i: att_shapes.append(i[0].shape[-1]))
                          for m in model.modules() if isinstance(m, (BoTAttention, CoAtAttentionFP32))]
                with torch.autocast('cpu', dtype=torch.bfloat16):
                    y = model(torch.randn(2, 3, 64, 64))
                y.float().square().mean().backward()
                for h in hooks:
                    h.remove()
                self.assertEqual(y.shape, (2, 200))
                meta = architecture_description(variant)
                self.assertEqual(shapes, meta['stage_spatial_sizes'])
                self.assertEqual(att_shapes, meta['attention_spatial_sizes'])
                for name, p in model.named_parameters():
                    self.assertIsNotNone(p.grad, name)
                    self.assertTrue(torch.isfinite(p.grad).all(), name)
                clone = build_model(variant, 43).eval()
                clone.load_state_dict(model.state_dict(), strict=True)
                model.eval()
                x = torch.randn(1, 3, 64, 64)
                with torch.no_grad():
                    torch.testing.assert_close(model(x), clone(x), rtol=0, atol=0)
                self.assertGreater(estimate_flops(model, 'cpu'), 0)

    def test_completed_only_compatibility(self):
        self.assertTrue(is_pre_twostage_source(BASELINE_0049D1A))
        changed = dict(BASELINE_0049D1A)
        changed['models/literature_conformer.py'] = 'changed'
        self.assertFalse(is_pre_twostage_source(changed))


if __name__ == '__main__':
    unittest.main()
