import argparse
import unittest

import torch
from torch import nn
from models.literature_alternet import PreBottleneck
from models.literature_resnet import build_model, architecture_description, BoTAttention
from utils.literature_profiling import estimate_flops
from utils.literature_fusion_compat import BASELINE_392A385, is_pre_prebot_source
from overnight_twostage import training_command


class PreBoTTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_only_cnn_blocks_changed(self):
        base, model = build_model('botnet50'), build_model('botnet50_preact_cnn')
        for name in ('conv1', 'bn1', 'layer4', 'fc'):
            a, b = getattr(base, name).state_dict(), getattr(model, name).state_dict()
            self.assertEqual(a.keys(), b.keys())
            for key in a:
                torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
        for name, count in zip(('layer1', 'layer2', 'layer3'), (3, 4, 6)):
            self.assertEqual(len(getattr(model, name)), count)
            for old, new in zip(getattr(base, name), getattr(model, name)):
                self.assertIsInstance(new, PreBottleneck)
                for a, b in ((old.conv1, new.conv1), (old.conv2, new.conv2[2]),
                             (old.conv3, new.conv3[2])):
                    torch.testing.assert_close(a.weight, b.weight, rtol=0, atol=0)
                    self.assertEqual(a.stride, b.stride)
                if old.downsample is not None:
                    torch.testing.assert_close(old.downsample[0].weight, new.shortcut[0].weight,
                                               rtol=0, atol=0)

    def test_training_shapes_gradients_and_checkpoint(self):
        model = build_model('botnet50_preact_cnn')
        shapes, att_shapes = [], []
        hooks = [getattr(model, f'layer{i}').register_forward_hook(
            lambda m, inp, out: shapes.append(out.shape[-1])) for i in range(1, 5)]
        hooks += [m.register_forward_pre_hook(lambda m, inp: att_shapes.append(inp[0].shape[-1]))
                  for m in model.modules() if isinstance(m, BoTAttention)]
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            y = model(torch.randn(2, 3, 64, 64))
            loss = nn.functional.cross_entropy(y, torch.tensor([1, 199]))
        loss.backward()
        for name, p in model.named_parameters():
            self.assertIsNotNone(p.grad, name)
            self.assertTrue(torch.isfinite(p.grad).all(), name)
        optimizer.step()
        for h in hooks:
            h.remove()
        self.assertEqual(y.shape, (2, 200))
        self.assertEqual(shapes, [32, 16, 8, 4])
        self.assertEqual(att_shapes, [8, 4, 4])
        clone = build_model('botnet50_preact_cnn', 43).eval()
        clone.load_state_dict(model.state_dict(), strict=True)
        model.eval()
        with torch.no_grad():
            x = torch.randn(2, 3, 64, 64)
            torch.testing.assert_close(model(x), clone(x), rtol=0, atol=0)
        self.assertEqual(estimate_flops(model, 'cpu'), estimate_flops(build_model('botnet50'), 'cpu'))
        self.assertEqual(architecture_description('botnet50_preact_cnn')['control'], 'botnet50')

    def test_old_comparisons_and_overnight_selection(self):
        self.assertTrue(is_pre_prebot_source(BASELINE_392A385))
        self.assertFalse(is_pre_prebot_source(dict(BASELINE_392A385, extra='unknown')))
        args = argparse.Namespace(resume=None, epochs=100, variants=['botnet50_preact_cnn', 'botnet50'])
        cmd = training_command(args)
        self.assertIn('botnet50_preact_cnn', cmd)
        self.assertIn('botnet50', cmd)
        self.assertNotIn('conformer_tiny', cmd)
        self.assertEqual(cmd[-4:], ['--seeds', '42', '--epochs', '100'])


if __name__ == '__main__':
    unittest.main()
