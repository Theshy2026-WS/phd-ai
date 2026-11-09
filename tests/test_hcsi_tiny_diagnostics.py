import copy
from contextlib import contextmanager
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from torch import nn

import hcsi_tiny_diagnostics as diagnostics


class TinyDataset:
    split = 'selection'
    folder = SimpleNamespace(classes=['a', 'b', 'c', 'd'])
    def __len__(self): return 4


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(3.))
    def forward(self, images): return images*self.scale
    def forward_decomposed(self, images):
        return dict(logits=self(images), stage1_cnn_gate_mean=images.new_full((len(images),), .5),
            stage1_vit_attention_entropy=images.new_full((len(images),), .75))
    @contextmanager
    def analysis(self, mode):
        yield self


def result(prediction):
    truth = np.array([0, 1, 2, 3])
    probabilities = np.full((4, 4), .01, np.float32)
    probabilities[np.arange(4), prediction] = .97
    return diagnostics.probability_result(probabilities, truth, ['a', 'b', 'c', 'd'], ['a', 'b', 'c', 'd'])


class HCSIDiagnosticsTests(unittest.TestCase):
    def test_clean_full_denominator_statistics_and_unchanged_state(self):
        model = ToyModel(); before = copy.deepcopy(model.state_dict())
        batches = [(torch.eye(4)[:2], torch.arange(2), ['a', 'b']),
                   (torch.eye(4)[2:], torch.arange(2, 4), ['c', 'd'])]
        with patch.object(diagnostics, 'loader', return_value=batches):
            values, stats = diagnostics.evaluate_predictions(model, TinyDataset(), {'dataset':{}}, 42, 'cpu',
                                                              collect_statistics=True)
        self.assertEqual(values['metrics']['n'], 4)
        self.assertEqual(values['metrics']['accuracy'], 1.)
        self.assertGreater(values['metrics']['nll'], 0.)
        self.assertEqual(stats['stage1_cnn_gate_mean']['mean'], .5)
        self.assertEqual(stats['stage1_vit_attention_entropy']['mean'], .75)
        for key, value in before.items(): self.assertTrue(torch.equal(value, model.state_dict()[key]))

    def test_reserved_split_rejected_without_iteration(self):
        dataset = TinyDataset(); dataset.split = 'audit'
        with patch.object(diagnostics, 'loader') as loader:
            with self.assertRaises(ValueError):
                diagnostics.evaluate_predictions(ToyModel(), dataset, {'dataset':{}}, 42, 'cpu')
        loader.assert_not_called()

    def test_deadline_never_returns_partial_score(self):
        with patch.object(diagnostics, 'loader', return_value=[(torch.eye(4), torch.arange(4), ['a','b','c','d'])]):
            with self.assertRaises(diagnostics.AnalysisDeadline):
                diagnostics.evaluate_predictions(ToyModel(), TinyDataset(), {'dataset':{}}, 42, 'cpu',
                                                  deadline_unix=time.time()-1)

    def test_nonfinite_logits_and_duplicate_ids_rejected(self):
        model = ToyModel()
        for images, ids in ((torch.full((4,4), float('nan')), ['a','b','c','d']),
                            (torch.eye(4), ['a','a','c','d'])):
            with patch.object(diagnostics, 'loader', return_value=[(images, torch.arange(4), ids)]):
                with self.assertRaises(ValueError):
                    diagnostics.evaluate_predictions(model, TinyDataset(), {'dataset':{}}, 42, 'cpu')

    def test_eval_detects_buffer_mutation(self):
        class Mutates(ToyModel):
            def __init__(self):
                super().__init__(); self.register_buffer('counter', torch.tensor(0))
            def forward(self, images):
                self.counter.add_(1); return super().forward(images)
        with patch.object(diagnostics, 'loader', return_value=[(torch.eye(4), torch.arange(4), ['a','b','c','d'])]):
            with self.assertRaisesRegex(RuntimeError, 'changed'):
                diagnostics.evaluate_predictions(Mutates(), TinyDataset(), {'dataset':{}}, 42, 'cpu')

    def test_mechanism_requires_both_matched_control_and_continued_cnn(self):
        values = dict(source_cnn=result([1,1,2,3]), cnn_continue=result([1,1,2,3]),
            cnn_mlp=result([1,1,2,3]), late_mlp=result([1,1,2,3]),
            hcsi=result([0,1,2,3]), existing_joint=result([1,1,2,3]),
            late_mlp_random_vit=result([1,1,2,3]))
        summaries = {job:dict(status='complete',completed_epochs=30,best_epoch=2) for job in diagnostics.JOBS}
        comparison = diagnostics.interpret_comparisons(values, 30, summaries)
        self.assertTrue(comparison['hcsi_mechanism_supported_on_selection'])
        self.assertEqual(comparison['comparisons']['hcsi_vs_late_mlp']['corrected'], 1)
        values['late_mlp'] = values['hcsi']
        self.assertFalse(diagnostics.interpret_comparisons(values,30,summaries)['hcsi_mechanism_supported_on_selection'])
        values['late_mlp'] = values['source_cnn']; values['cnn_continue'] = values['hcsi']
        self.assertFalse(diagnostics.interpret_comparisons(values,30,summaries)['hcsi_mechanism_supported_on_selection'])

    def test_incomplete_training_and_epoch_zero_cannot_advance(self):
        values = {job:result([1,1,2,3]) for job in diagnostics.JOBS}
        values['hcsi'] = result([0,1,2,3])
        summaries = {job:dict(status='complete',completed_epochs=30,best_epoch=2) for job in diagnostics.JOBS}
        summaries['hcsi']['best_epoch'] = 0
        self.assertFalse(diagnostics.interpret_comparisons(values,30,summaries)['hcsi_mechanism_supported_on_selection'])
        summaries['hcsi']['best_epoch'] = 2; summaries['cnn_mlp']['completed_epochs'] = 29
        self.assertFalse(diagnostics.interpret_comparisons(values,30,summaries)['hcsi_mechanism_supported_on_selection'])

    def test_independent_vit_pretraining_has_its_own_capacity_control(self):
        values = {job:result([1,1,2,3]) for job in diagnostics.JOBS}
        values['late_mlp'] = result([0,1,2,3])
        summaries = {job:dict(status='complete',completed_epochs=30,best_epoch=2) for job in diagnostics.JOBS}
        report = diagnostics.interpret_comparisons(values,30,summaries)
        self.assertTrue(report['independent_vit_pretraining_supported_on_selection'])
        self.assertFalse(report['hcsi_mechanism_supported_on_selection'])
        values['late_mlp_random_vit'] = values['late_mlp']
        self.assertFalse(diagnostics.interpret_comparisons(values,30,summaries)['independent_vit_pretraining_supported_on_selection'])

    def test_complete_pipeline_loads_five_jobs_and_reuses_existing_joint_without_training(self):
        class Sized:
            def __init__(self, n): self.n = n
            def __len__(self): return self.n
        plan = dict(config={'dataset':{}}, seed=42, split_hash='split', epochs=30)
        summary = dict(status='complete', completed_epochs=30, best_epoch=1)
        state = dict(epoch=1, source='raw')
        module = SimpleNamespace(load_sources=MagicMock(return_value={}),
            load_reference_model=MagicMock(return_value=(ToyModel(), state)),
            load_model=MagicMock(side_effect=lambda *a, **k:(ToyModel(), state)))
        values = result([0,1,2,3])
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); training=root/'training'; training.mkdir()
            for job in diagnostics.JOBS:
                run=training/job;run.mkdir()
                (run/'summary.json').write_text(json.dumps(summary))
                for name in ('best.pt','last.pt'):(run/name).write_bytes(b'checkpoint')
            with patch.dict('sys.modules', {'hcsi_tiny_experiment':module}), \
                    patch.object(diagnostics, 'prepare_datasets', return_value=(
                        {'selection':Sized(10000),'train_probe':Sized(4000)},
                        {'train':['train/a'],'selection':['selection/a']},'split')), \
                    patch.object(diagnostics, '_build_source_models', return_value={
                        'source_cnn':(ToyModel(),state),'source_vit':(ToyModel(),state)}), \
                    patch.object(diagnostics, 'evaluate_predictions', return_value=(values,{})) as evaluation, \
                    patch.object(diagnostics, '_write_result'):
                report = diagnostics.run_hcsi_diagnostics(plan,training,root/'analysis')
            self.assertEqual(report['status'], 'complete')
            self.assertEqual(module.load_model.call_count, len(diagnostics.JOBS)*3)
            self.assertEqual({c.args[2] for c in module.load_model.call_args_list},set(diagnostics.JOBS))
            self.assertEqual(evaluation.call_count, 2*(3+len(diagnostics.JOBS)*3)+2)
            module.load_reference_model.assert_called_once()
            self.assertFalse(report['audit_partition_evaluated'])
            self.assertFalse(report['official_validation_evaluated'])
            self.assertIn('hcsi/best.pt', report['jobs'])
            self.assertIn('existing_joint/source_best',report['jobs'])
            self.assertTrue((root/'analysis/report.md').is_file())

    def test_json_report_retry_and_symlink_guard(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'summary.json'
            diagnostics.save_json(path, {'status':'partial'})
            self.assertEqual(json.loads(path.read_text())['status'], 'partial')
            with patch.object(Path, 'is_symlink', return_value=True):
                with self.assertRaises(ValueError): diagnostics.save_json(path, {})


if __name__ == '__main__': unittest.main()
