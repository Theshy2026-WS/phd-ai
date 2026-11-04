from contextlib import contextmanager
import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

import attention_fusion_diagnostics as runner
from datasets.utility_tiny import UtilityDataset
from test_baseline_diagnostics import config as original_config, fixture_datasets


class ToyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(3, 4, 1), nn.BatchNorm2d(4), nn.ReLU())
        self.fusion = nn.Conv2d(4, 4, 1)
        self.head = nn.Linear(4, 3)
        self.gamma = nn.Parameter(torch.tensor(.1))
        self.analysis_mode = 'learned'
        self.capture_attention = False

    @contextmanager
    def analysis(self, mode):
        previous = self.analysis_mode
        self.analysis_mode = mode
        try: yield self
        finally: self.analysis_mode = previous

    def forward_decomposed(self, images):
        before = self.encoder(images)
        if self.analysis_mode == 'removed':
            delta = torch.zeros_like(before)
        elif self.analysis_mode == 'uniform':
            delta = self.fusion(before).mean((-2, -1), keepdim=True).expand_as(before)
        else:
            delta = self.fusion(before)
        after = before+self.gamma*delta
        return dict(logits=self.head(after.mean((-2, -1))), features_before=before,
                    features_after=after, gamma=self.gamma,
                    attention_entropy=torch.ones(len(images))*.5, attention_peak=.25,
                    gate=None, retrieval_maps={})


class AttentionDiagnosticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def supplied(self, root):
        datasets, manifest, split = fixture_datasets(root)
        base = datasets['train']
        probe = UtilityDataset(base.folder, base.indices, 'train_probe', 42)
        return probe, datasets['selection'], manifest, split

    def test_all_modes_evaluate_same_ids_without_parameter_or_buffer_changes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            probe, selection, _, _ = self.supplied(root/'images')
            config = original_config(); config['dataset'].update(batch_size=2, num_workers=0)
            model = ToyAttention().eval()
            digest = runner.state_digest(model)
            results = {}
            for mode in ('learned', 'uniform', 'removed'):
                results[mode] = runner.evaluate_intervention(model, selection, config, mode=mode,
                    device='cpu', output=root/mode, class_names=['a','b','c'])
                self.assertEqual(results[mode]['metrics']['n'], len(selection))
                self.assertEqual(model.analysis_mode, 'learned')
            self.assertEqual(results['removed']['metrics']['relative_feature_change'], 0.)
            self.assertGreater(results['learned']['metrics']['relative_feature_change'], 0.)
            self.assertEqual(results['learned']['image_ids'], results['uniform']['image_ids'])
            self.assertEqual(digest, runner.state_digest(model))
            pair = runner.compare_decisions(results['learned'], results['uniform'])
            self.assertEqual(pair['n'], len(selection))

    def test_pairing_direction_and_id_guard(self):
        left = dict(image_ids=['a','b','c','d'], targets=np.array([0,0,0,0]),
                    predictions=np.array([0,0,1,1]), nll=np.array([.1,.2,2.,2.]))
        right = dict(image_ids=['a','b','c','d'], targets=np.array([0,0,0,0]),
                     predictions=np.array([1,1,0,1]), nll=np.array([2.,2.,.1,2.]))
        result = runner.compare_decisions(left, right)
        self.assertEqual((result['corrected_by_learned'], result['harmed_by_learned']), (2,1))
        self.assertEqual(result['delta_learned_top1_pp'], 25.)
        right['image_ids'] = right['image_ids'][::-1]
        with self.assertRaisesRegex(ValueError, 'identical ordered'):
            runner.compare_decisions(left, right)

    def test_complete_api_analyzes_best_last_and_trained_peak_when_best_is_zero(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            parent = root/'parent'; parent.mkdir(); (parent/'best.pt').write_bytes(b'immutable')
            training = root/'training'; training.mkdir()
            job = 'warm_cross16'; run = training/job; run.mkdir()
            model = ToyAttention().eval()
            config = original_config(); config['dataset'].update(batch_size=2, num_workers=0)
            probe, selection, manifest, split = self.supplied(root/'images')
            parent_hash = runner.file_hash(parent/'best.pt')
            metadata = dict(job=job, variant='cross16', model_variant='cross16', scope='warm',
                seed=42, parent_checkpoint_sha256=parent_hash, source_hashes={'fixture':'hash'},
                recipe_hash=runner.recipe_hash(config))
            summary = dict(**metadata, best_epoch=0, completed_epochs=2,
                best_trained_checkpoint='best_trained.pt', best_trained_selection=dict(epoch=1, source='raw'))
            (run/'summary.json').write_text(json.dumps(summary))
            for name, epoch in (('best.pt',0),('last.pt',2),('best_trained.pt',1)):
                torch.save(dict(model=model.state_dict(), metadata=metadata, config=config, epoch=epoch), run/name)
            (training/'suite.json').write_text(json.dumps(dict(jobs=[job], parent_checkpoint_sha256=parent_hash,
                split_hash=split, width=128, heads=4)))
            loaded = (parent, dict(seed=42, split_hash=split), dict(config=config, model={}), parent_hash)
            output = root/'analysis'
            with patch.object(runner, 'load_spatial_parent', return_value=loaded), \
                    patch.object(runner, 'clean_datasets', return_value=(probe,selection,[],split,manifest)), \
                    patch('models.attention_fusion.build_attention_model', side_effect=lambda *a,**kw:copy.deepcopy(model)), \
                    patch.object(runner.torch.cuda, 'is_available', return_value=False):
                result = runner.run_attention_diagnostics(parent, training, output, config=config, examples=0)
                self.assertEqual(result['status'], 'complete')
                again = runner.run_attention_diagnostics(parent, training, output, config=config, examples=0)
                self.assertEqual(result, again)
            with (output/result['analysis_directory']/'comparison.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 18)
            self.assertEqual({row['checkpoint'] for row in rows}, {'best','last_raw','best_trained'})
            self.assertEqual((parent/'best.pt').read_bytes(), b'immutable')
            self.assertFalse(result['official_validation_evaluated'])


if __name__ == '__main__': unittest.main()
