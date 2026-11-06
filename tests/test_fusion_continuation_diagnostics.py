import copy
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

import fusion_continuation_diagnostics as diagnostics
from models.attention_correction import state_digest


NAMES = ['zero', 'one', 'two']


class Images(Dataset):
    def __init__(self, split='train_clean', n=6, label_shift=0):
        self.split, self.n, self.label_shift, self.reads = split, n, label_shift, 0
        self.epoch = 0

    def __len__(self): return self.n

    def __getitem__(self, index):
        self.reads += 1
        x = torch.arange(12, dtype=torch.float32).reshape(3, 2, 2)/12+index/7
        return x, (index+self.label_shift) % 3, f'{self.split}/{index}.png'


class Feature(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.project = nn.Linear(3, width)
        self.norm = nn.BatchNorm1d(width)

    def forward(self, x): return self.norm(self.project(x.mean((-2, -1))))


class ToyViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature = Feature(2)
        self.classifier = nn.Linear(2, 3)

    def forward_features(self, x): return self.feature(x)
    def forward(self, x): return self.classifier(self.forward_features(x))


class ToyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature = Feature(4)
        self.classifier = nn.Linear(4, 3)

    def forward(self, x): return self.classifier(self.feature(x))


class ToyDual(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Feature(4)
        self.vit = ToyViT()
        self.vit.classifier = nn.Identity()
        self.fusion_classifier = nn.Linear(6, 3)

    def forward(self, x):
        return self.fusion_classifier(torch.cat((self.encoder(x), self.vit.forward_features(x)), 1))


def config():
    return dict(dataset=dict(batch_size=2, num_workers=0), training=dict(seed=42))


def result(correct, label='x'):
    truth = np.zeros(len(correct), dtype=np.int64)
    p = np.asarray([[.8, .1, .1] if c else [.1, .8, .1] for c in correct], dtype=np.float32)
    return diagnostics.probability_result(p, truth, [f'{label}/{i}' for i in range(len(p))], NAMES)


class ProbabilityEvidenceTests(unittest.TestCase):
    def test_fixed_mixtures_are_probability_combinations_and_pair_counts_are_exact(self):
        y = np.array([0, 1, 2, 0])
        cnn = diagnostics.probability_result(np.array([[.6,.3,.1],[.5,.4,.1],[.1,.2,.7],[.4,.5,.1]]),
            y, list('abcd'), NAMES)
        vit = diagnostics.probability_result(np.array([[.1,.8,.1],[.1,.8,.1],[.2,.7,.1],[.9,.05,.05]]),
            y, list('abcd'), NAMES)
        rows, mixtures = diagnostics.fixed_mixtures(cnn, vit)
        self.assertEqual(len(rows), 8)
        np.testing.assert_allclose(mixtures[.2]['probabilities'], .8*cnn['probabilities']+.2*vit['probabilities'])
        self.assertEqual(mixtures[.2]['metrics']['accuracy'], 1.)
        paired = diagnostics.paired_result(mixtures[.2], cnn)
        self.assertEqual((paired['corrected'], paired['harmed']), (2, 0))
        self.assertEqual(mixtures[.5]['metrics']['accuracy'], .5)
        self.assertEqual(mixtures[0.]['metrics']['accuracy'], cnn['metrics']['accuracy'])

    def test_threeway_counts_retain_all_eight_cells_without_claiming_oracle_is_a_model(self):
        cells = [(a,b,c) for a in (False,True) for b in (False,True) for c in (False,True)]
        cnn, vit, candidate = [result([cell[i] for cell in cells]) for i in range(3)]
        values = diagnostics.threeway_counts(cnn, vit, candidate)
        self.assertEqual([r['n'] for r in values['rows']], [1]*8)
        self.assertEqual(values['vit_only_available'], 2)
        self.assertEqual(values['vit_only_retained'], 1)
        self.assertEqual(values['newly_correct_when_both_wrong'], 1)
        self.assertEqual(values['harmed_when_both_correct'], 1)
        self.assertIn('not deployable', values['oracle_note'])

    def test_probability_file_rejects_wrong_ids_classes_and_stored_predictions(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'predictions.npz'
            base = result([True,False])
            np.savez_compressed(path, image_ids=base['image_ids'], targets=base['targets'],
                probabilities=base['probabilities'], predicted_labels=base['predictions'], class_names=NAMES)
            loaded = diagnostics.load_probabilities(path, NAMES, base['image_ids'], base['targets'])
            self.assertEqual(loaded['metrics']['accuracy'], .5)
            with self.assertRaisesRegex(ValueError, 'image order'):
                diagnostics.load_probabilities(path, NAMES, list(reversed(base['image_ids'])))
            with self.assertRaisesRegex(ValueError, 'class mapping'):
                diagnostics.load_probabilities(path, list(reversed(NAMES)))
            np.savez_compressed(path, image_ids=base['image_ids'], targets=base['targets'],
                probabilities=base['probabilities'], predicted_labels=[2,2], class_names=NAMES)
            with self.assertRaisesRegex(ValueError, 'argmax'):
                diagnostics.load_probabilities(path)

    def test_invalid_probability_normalization_and_path_escape_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'normalization'):
            diagnostics.probability_result([[.8,.8,.1]], [0], ['a'], NAMES)
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, 'relative'):
                diagnostics._inside(temp, '../foreign.npz')


class FeatureInterventionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def test_training_mean_is_label_blind_and_rejects_selection_fit(self):
        torch.manual_seed(7)
        model = ToyDual().eval()
        before = state_digest(model)
        mean, info = diagnostics.mean_vit_feature(model, Images(), config(), 42, 'cpu')
        another, _ = diagnostics.mean_vit_feature(model, Images(label_shift=1), config(), 42, 'cpu')
        torch.testing.assert_close(mean, another, rtol=0, atol=0)
        self.assertEqual(info['n'], 6)
        self.assertFalse(info['uses_labels'])
        self.assertFalse(info['uses_selection_images'])
        self.assertEqual(before, state_digest(model))
        with self.assertRaisesRegex(ValueError, 'clean training probe'):
            diagnostics.mean_vit_feature(model, Images('selection'), config(), 42, 'cpu')

    def test_all_modes_share_features_and_match_direct_forward_and_manual_mean(self):
        torch.manual_seed(19)
        model = ToyDual().eval()
        mean, _ = diagnostics.mean_vit_feature(model, Images(), config(), 42, 'cpu')
        selected = Images('selection', 4)
        before = state_digest(model)
        values, contributions = diagnostics.evaluate_modes(model, selected, config(), 42, 'cpu', NAMES,
            mean_feature=mean)
        self.assertEqual(set(values), set(diagnostics.DUAL_MODES))
        images = torch.stack([selected[i][0] for i in range(len(selected))])
        with torch.inference_mode():
            direct = model(images).softmax(1).numpy()
            manual = model.fusion_classifier(torch.cat((model.encoder(images), mean.expand(len(images), -1)), 1)).softmax(1).numpy()
        np.testing.assert_allclose(values['learned']['probabilities'], direct, rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(values['vit_train_mean']['probabilities'], manual, rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(contributions['cnn_true_margin']+contributions['vit_true_margin']+
            contributions['bias_true_margin'], contributions['true_margin'], rtol=1e-5, atol=1e-6)
        self.assertEqual(before, state_digest(model))
        grouped = diagnostics.contribution_groups(contributions, values['learned'], values['cnn_only'])
        self.assertEqual(grouped[0]['n'], 4)

    def test_cnn_control_has_no_mean_or_fake_branch_interventions(self):
        model = ToyCNN().eval()
        values, contributions = diagnostics.evaluate_modes(model, Images('selection', 4), config(), 42, 'cpu', NAMES)
        self.assertEqual(set(values), {'learned'})
        self.assertEqual(contributions, {})

    def test_complete_postdiagnostic_report_survives_compaction_and_reuses_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            training, output, old = root/'training', root/'analysis', root/'old_source'
            training.mkdir(); old.mkdir()
            parent = root/'original_cnn'; parent.mkdir()
            source_file = old/'source_marker.json'; source_file.write_text('{}')
            torch.manual_seed(3)
            cnn, vit, dual = ToyCNN(), ToyViT(), ToyDual()
            sources = dict(source_suite=str(old), source_training_dir=str(old/'training'),
                parent_run=str(parent), parent_model=cnn.state_dict(), vit_model=vit.state_dict(),
                dual_best=dict(epoch=30, model=copy.deepcopy(dual.state_dict())),
                dual_last=dict(epoch=30, model=copy.deepcopy(dual.state_dict())),
                config=config(), seed=42, split_hash='same-split', provenance={'old': 'immutable'},
                source_files={str(source_file):diagnostics.file_hash(source_file)})
            plan = dict(jobs=['dual_continue','cnn_head_control'], source_hashes={'new':'same'},
                source_provenance=sources['provenance'], seed=42, split_hash='same-split')
            (training/'suite.json').write_text(json.dumps(plan))
            for job in plan['jobs']:
                run = training/job; run.mkdir()
                variant = 'dual_fusion' if job == 'dual_continue' else 'cnn_head_control'
                metadata = dict(job=job, variant=variant, seed=42, split_hash='same-split',
                    source_provenance=sources['provenance'], source_hashes=plan['source_hashes'],
                    recipe_hash=diagnostics.recipe_hash(config()))
                summary = dict(metadata, status='complete', best_epoch=0, completed_epochs=2,
                    inherited_joint_epochs=30 if job == 'dual_continue' else 0,
                    best_trained_checkpoint='best_trained.pt', best_trained_selection={'epoch':1})
                (run/'summary.json').write_text(json.dumps(summary))
                state_model = dual if variant == 'dual_fusion' else cnn
                for name, epoch in [('best.pt',0),('last.pt',2),('best_trained.pt',1)]:
                    torch.save(dict(model=state_model.state_dict(), epoch=epoch,
                        metadata=metadata, config=config()), run/name)
            probe, selection = Images('train_clean', 6), Images('selection', 4)
            manifest = dict(train_clean=[f'train_clean/{i}.png' for i in range(6)],
                selection=[f'selection/{i}.png' for i in range(4)])
            source_module = types.ModuleType('fusion_continuation_sources')
            source_module.load_sources = lambda *args, **kwargs: sources
            training_module = types.ModuleType('fusion_continuation_experiment')
            training_module.source_hashes = lambda: {'new':'same'}
            training_module.build_job_model = lambda job, *args, **kwargs: ToyDual() if job.startswith('dual') else ToyCNN()
            with patch.dict(sys.modules, {'fusion_continuation_sources':source_module,
                    'fusion_continuation_experiment':training_module}), \
                    patch.object(diagnostics, 'clean_datasets', return_value=(probe,selection,None,'same-split',manifest)), \
                    patch.object(diagnostics, 'class_labels', return_value=NAMES), \
                    patch.object(diagnostics, 'build_spatial_model', side_effect=lambda *args: ToyCNN()), \
                    patch.object(diagnostics, 'build_followup_model', side_effect=lambda *args, **kwargs: vit), \
                    patch.object(diagnostics.torch.cuda, 'is_available', return_value=False):
                first = diagnostics.run_continuation_diagnostics(old, training, output, examples=0)
                reads = (probe.reads, selection.reads)
                second = diagnostics.run_continuation_diagnostics(old, training, output, examples=0)
            self.assertEqual(first, second)
            self.assertEqual((probe.reads,selection.reads), reads)
            self.assertEqual(first['status'], 'complete')
            self.assertEqual(len(list(output.glob('attempt_*'))), 1)
            attempt = output/first['analysis_directory']
            import csv
            rows = list(csv.DictReader((attempt/'comparison.csv').open()))
            self.assertEqual(len(rows), 8+24+6)
            self.assertEqual(set(row['mode'] for row in rows if row['job']=='cnn_head_control'), {'learned'})
            self.assertEqual(len(list(csv.DictReader((attempt/'static_probability_fusion.csv').open()))), 16)
            trios = json.loads((attempt/'threeway_summary.json').read_text())
            self.assertEqual(len(trios), 7)
            self.assertTrue((attempt/'contribution_summary.csv').is_file())
            self.assertFalse(first['audit_partition_evaluated'])
            self.assertFalse(first['official_validation_evaluated'])
            self.assertIn('dual_continue/best/selection/vs_static_0.8cnn_0.2vit', first['paired'])


if __name__ == '__main__':
    unittest.main()
