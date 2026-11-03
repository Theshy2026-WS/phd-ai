import csv
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, Subset

from utils.baseline_diagnostics import evaluate_clean


class KnownDataset(Dataset):
    split = 'selection'

    def __init__(self, labels=None, identifiers=None, root=None):
        self.labels = [0, 2, 5, 1, 0] if labels is None else labels
        self.identifiers = identifiers or [f'image_{i}.png' for i in range(len(self.labels))]
        self.calls = 0
        if root is not None:
            self.folder = SimpleNamespace(root=str(root))
            for i, name in enumerate(self.identifiers):
                Image.new('RGB', (64, 64), (i*40, 50, 200)).save(root/name)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        self.calls += 1
        return torch.full((3, 4, 4), float(index)), self.labels[index], self.identifiers[index]


class KnownModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.))
        self.register_buffer('scores', torch.tensor([[5., 4, 3, 2, 1, 0],
                                                   [0., 5, 4, 3, 2, 1],
                                                   [5., 4, 3, 2, 1, 0],
                                                   [0., 5, 4, 3, 2, 1],
                                                   [5., 4, 3, 2, 1, 0]]))

    def forward(self, x):
        return self.scores[x[:, 0, 0, 0].long()]*self.scale


class DiagnosticMetricsTests(unittest.TestCase):
    names = ['a', 'b', 'c', 'd', 'e', 'f']

    def evaluate(self, model, dataset, output, **kwargs):
        return evaluate_clean(model, dataset, device='cpu', batch_size=2, num_workers=0,
                              output_dir=output, class_names=self.names, **kwargs)

    def test_known_scores_exports_and_model_immutability(self):
        torch.set_num_threads(2)
        model = KnownModel().train()
        before = {name: value.clone() for name, value in model.state_dict().items()}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dataset = KnownDataset(root=root)
            result = self.evaluate(model, dataset, root/'result', example_count=3)
            self.assertEqual(result['n'], 5)
            self.assertEqual(result['accuracy'], .6)
            self.assertEqual(result['top5_accuracy'], .8)
            self.assertEqual(result['macro_accuracy'], .5)
            self.assertEqual(result['top5_wrong_count'], 1)
            self.assertEqual(result['outside_top5_count'], 1)
            expected = torch.nn.functional.cross_entropy(model.scores,
                                                         torch.tensor(dataset.labels)).item()
            self.assertAlmostEqual(result['nll'], expected, places=6)
            self.assertEqual(result['class_counts'], dict(a=2, b=1, c=1, d=0, e=0, f=1))
            self.assertEqual(dataset.calls, 5)  # Galleries open the raw source image.
            with np.load(root/'result/predictions.npz', allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved['targets'], dataset.labels)
                self.assertEqual(saved['probabilities'].dtype, np.float32)
                np.testing.assert_allclose(saved['probabilities'].sum(1), 1., atol=1e-6)
                np.testing.assert_array_equal(saved['image_ids'], dataset.identifiers)
            with (root/'result/predictions.csv').open(newline='', encoding='utf-8') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([int(row['true_rank']) for row in rows], [1, 2, 6, 1, 1])
            self.assertEqual(json.loads(rows[1]['top5_labels']), [1, 2, 3, 4, 5])
            with (root/'result/per_class.csv').open(newline='', encoding='utf-8') as stream:
                classes = list(csv.DictReader(stream))
            self.assertEqual(classes[0]['correct'], '2')
            self.assertEqual(classes[2]['recall'], '0.0')
            self.assertEqual(classes[3]['nll'], '')
            with (root/'result/confusion_pairs.csv').open(newline='', encoding='utf-8') as stream:
                pairs = list(csv.DictReader(stream))
            self.assertEqual([(int(row['true_label']), int(row['predicted_label']), int(row['count']))
                              for row in pairs], [(2, 1, 1), (5, 0, 1)])
            self.assertEqual(json.loads((root/'result/metrics.json').read_text()), result)
            for name, expected_count in [('true_top5_wrong', 1), ('true_outside_top5', 1),
                                          ('confident_wrong', 2), ('correct_controls', 3)]:
                with Image.open(root/f'result/galleries/{name}.png') as gallery:
                    self.assertEqual(gallery.width, 1360)
                self.assertEqual(result['galleries'][name]['n'], expected_count)
        self.assertFalse(model.training)
        self.assertIsNone(model.scale.grad)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)

    def test_stable_ties_and_fallback_gallery(self):
        model = KnownModel()
        model.scores.zero_()
        dataset = KnownDataset(labels=[0, 5])
        with tempfile.TemporaryDirectory() as temp:
            result = self.evaluate(model, dataset, Path(temp)/'result', example_count=2)
            self.assertEqual(result['accuracy'], .5)
            self.assertEqual(result['top5_accuracy'], .5)
            self.assertAlmostEqual(result['nll'], np.log(6), places=6)
            self.assertEqual(result['galleries']['true_top5_wrong']['n'], 0)
            self.assertTrue((Path(temp)/'result/galleries/true_top5_wrong.png').exists())

    def test_invalid_inputs_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cases = [(KnownDataset(labels=[]), 'empty', 'Empty evaluation'),
                     (KnownDataset(identifiers=['same']*5), 'duplicate', 'Duplicate'),
                     (KnownDataset(labels=[6]), 'label', 'Class index'),
                     (KnownDataset(identifiers=['../escape']*5), 'path', 'dataset-relative')]
            for dataset, name, message in cases:
                with self.subTest(name=name), self.assertRaisesRegex(ValueError, message):
                    self.evaluate(KnownModel(), dataset, root/name, example_count=0)
                self.assertFalse((root/name/'metrics.json').exists())
            output = root/'exists'
            output.mkdir()
            (output/'sentinel').write_text('keep')
            with self.assertRaisesRegex(ValueError, 'overwrite'):
                self.evaluate(KnownModel(), KnownDataset(), output)
            self.assertEqual((output/'sentinel').read_text(), 'keep')
            dirty = KnownDataset()
            dirty.split = 'train'
            with self.assertRaisesRegex(ValueError, 'augmented'):
                self.evaluate(KnownModel(), Subset(dirty, [0]), root/'augmented')

    def test_nonfinite_logits_and_fp32_policy(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model = KnownModel()
            model.scores[0, 0] = float('nan')
            with self.assertRaisesRegex(ValueError, 'non-finite'):
                self.evaluate(model, KnownDataset(), root/'nonfinite', example_count=0)
            self.assertFalse((root/'nonfinite/metrics.json').exists())
            with self.assertRaisesRegex(ValueError, 'FP32 model'):
                self.evaluate(KnownModel().double(), KnownDataset(), root/'double')


if __name__ == '__main__':
    unittest.main()
