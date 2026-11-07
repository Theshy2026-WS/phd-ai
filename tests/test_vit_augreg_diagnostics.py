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

import vit_augreg_diagnostics as diagnostics
from models.attention_correction import state_digest


NAMES = ['zero', 'one', 'two']
TEST_ROOT = Path(__file__).resolve().parents[1]/'runs'/'vit_augreg_diagnostic_tests_tmp'


def temporary_directory():
    TEST_ROOT.mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(dir=TEST_ROOT)


class Images(Dataset):
    def __init__(self, split='train_clean', n=6, label_shift=0):
        self.split, self.n, self.label_shift = split, n, label_shift
        self.epoch = 0

    def __len__(self): return self.n

    def __getitem__(self, index):
        x = torch.arange(12, dtype=torch.float32).reshape(3,2,2)/12+index/7
        return x, (index+self.label_shift) % 3, f'{self.split}/{index}.png'


class Features(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.project = nn.Linear(3, width)
        self.bn = nn.BatchNorm1d(width)

    def forward(self, x): return self.bn(self.project(x.mean((-2,-1))))


class ToyViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = Features(2)
        self.classifier = nn.Linear(2, 3)

    def forward_tokens(self, x):
        cls = self.features(x)
        return torch.stack((cls, cls*.8+.2, cls*1.2-.1), 1)

    def forward_features(self, x): return self.forward_tokens(x)[:,0]
    def forward(self, x): return self.classifier(self.forward_features(x))


class ToyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = Features(4)
        self.classifier = nn.Linear(4,3)

    def forward(self, x): return self.classifier(self.features(x))


class ToyDual(nn.Module):
    analysis_modes = ('learned','cnn_only','vit_only','vit_train_mean','vit_mismatch','uniform')

    def __init__(self):
        super().__init__()
        self.encoder, self.vit = Features(4), ToyViT()
        self.vit.classifier = nn.Identity()
        self.fusion_classifier = nn.Linear(6,3)
        self.extractions = 0

    def extract_features(self, images):
        self.extractions += 1
        tokens = self.vit.forward_tokens(images)
        return self.encoder(images), tokens[:,0], tokens[:,1:]

    def forward_from_features(self, cnn, cls, patches, mode='learned', mean_feature=None, mean_patches=None):
        if mode == 'cnn_only': cls = torch.zeros_like(cls)
        elif mode == 'vit_only': cnn = torch.zeros_like(cnn)
        elif mode == 'vit_train_mean': cls = mean_feature.to(cls).expand_as(cls)
        elif mode == 'uniform': cls = patches.mean(1)
        return self.fusion_classifier(torch.cat((cnn,cls),1))

    def forward(self, images): return self.forward_from_features(*self.extract_features(images))


class ToyAnchored(ToyDual):
    analysis_modes = ('learned','cnn_only','vit_train_mean','vit_mismatch')
    residual_bound = 1.

    def __init__(self, violate_bound=False):
        super().__init__()
        self.register_buffer('correction_scale',torch.tensor(.1))
        self.violate_bound=violate_bound

    def forward_from_features(self,cnn,cls,patches,mode='learned',mean_feature=None,mean_patches=None):
        base=torch.zeros((len(cnn),3),device=cnn.device)
        base[:,0]=2.
        base[:,2]=-2.
        if mode=='cnn_only':return base
        if mode=='vit_train_mean':cls=mean_feature.to(cls).expand_as(cls)
        delta=self.correction_scale*torch.cat((cls,cls[:,:1]),1).tanh()
        if self.violate_bound:delta=delta+1.
        return base+delta


def config(): return dict(dataset=dict(batch_size=2,num_workers=0), training=dict(seed=42))


class ViTAugRegDiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def test_probe_means_are_label_blind_and_selection_cannot_fit_them(self):
        torch.manual_seed(7)
        model = ToyDual().eval()
        before = state_digest(model)
        cls, patches, info = diagnostics.mean_vit_features(model, Images(),config(),42,'cpu')
        another, other_patches, _ = diagnostics.mean_vit_features(model, Images(label_shift=1),config(),42,'cpu')
        torch.testing.assert_close(cls,another,rtol=0,atol=0)
        torch.testing.assert_close(patches,other_patches,rtol=0,atol=0)
        self.assertEqual(tuple(patches.shape),(2,2))
        self.assertFalse(info['uses_labels'])
        self.assertFalse(info['uses_selection_images'])
        self.assertEqual(state_digest(model),before)
        with self.assertRaisesRegex(ValueError,'clean training probe'):
            diagnostics.mean_vit_features(model,Images('selection'),config(),42,'cpu')

    def test_modes_reuse_features_and_margins_add_up_with_final_singleton(self):
        torch.manual_seed(19)
        model = ToyDual().eval()
        cls, patches, _ = diagnostics.mean_vit_features(model,Images(),config(),42,'cpu')
        selected = Images('selection',5)
        before = state_digest(model)
        values, contribution, policy = diagnostics.evaluate_modes(model,selected,config(),42,'cpu',NAMES,
            mean_feature=cls,mean_patches=patches)
        self.assertEqual(model.extractions,3)
        self.assertEqual(set(values),set(model.analysis_modes))
        self.assertEqual(policy['mismatched_images'],5)
        self.assertTrue(policy['all_mismatches_have_different_ids'])
        self.assertEqual(state_digest(model),before)
        np.testing.assert_allclose(contribution['base_true_margin']+contribution['correction_true_margin'],
            contribution['full_true_margin'],rtol=1e-5,atol=1e-6)
        images = torch.stack([selected[i][0] for i in range(5)])
        with torch.inference_mode():
            direct = model(images).softmax(1).numpy()
            cnn, _, patch = model.extract_features(images)
            manual = model.forward_from_features(cnn,cls.expand(5,-1),patch).softmax(1).numpy()
        np.testing.assert_allclose(values['learned']['probabilities'],direct,rtol=1e-6,atol=1e-7)
        np.testing.assert_allclose(values['vit_train_mean']['probabilities'],manual,rtol=1e-6,atol=1e-7)
        self.assertFalse(np.allclose(values['learned']['probabilities'],values['vit_mismatch']['probabilities']))

    def test_missing_means_and_undefined_single_image_mismatch_are_rejected(self):
        model = ToyDual().eval()
        with self.assertRaisesRegex(ValueError,'training-only'):
            diagnostics.evaluate_modes(model,Images('selection',3),config(),42,'cpu',NAMES)
        cls,patches,_ = diagnostics.mean_vit_features(model,Images(),config(),42,'cpu')
        with self.assertRaisesRegex(ValueError,'two distinct images'):
            diagnostics.evaluate_modes(model,Images('selection',1),config(),42,'cpu',NAMES,
                mean_feature=cls,mean_patches=patches)

    def test_cnn_control_has_no_fake_vit_modes(self):
        values, contribution, policy = diagnostics.evaluate_modes(ToyCNN().eval(),Images('selection',4),config(),42,'cpu',NAMES)
        self.assertEqual(set(values),{'learned'})
        self.assertEqual(contribution,{})
        self.assertEqual(policy['mismatched_images'],0)

    def test_bounded_logit_correction_preserves_predictions_in_safe_anchor_zone(self):
        model=ToyAnchored().eval()
        cls,patches,_=diagnostics.mean_vit_features(model,Images(),config(),42,'cpu')
        values,_,policy=diagnostics.evaluate_modes(model,Images('selection',5),config(),42,'cpu',NAMES,
            mean_feature=cls,mean_patches=patches)
        bounds=policy['bounded_correction']
        self.assertAlmostEqual(bounds['learned_correction_scale'],.1,places=6)
        self.assertLessEqual(bounds['correction_max_abs_logit'],.1+1e-6)
        self.assertEqual(bounds['safe_anchor_zone_count'],5)
        self.assertEqual(bounds['safe_anchor_zone_prediction_changes'],0)
        self.assertEqual(bounds['safe_anchor_zone_fraction'],1.)
        self.assertTrue(bounds['bound_verified'])
        np.testing.assert_array_equal(values['learned']['predictions'],values['cnn_only']['predictions'])

    def test_diagnostic_rejects_violation_of_declared_correction_bound(self):
        model=ToyAnchored(violate_bound=True).eval()
        cls,patches,_=diagnostics.mean_vit_features(model,Images(),config(),42,'cpu')
        with self.assertRaisesRegex(ValueError,'exceeds its declared bound'):
            diagnostics.evaluate_modes(model,Images('selection',3),config(),42,'cpu',NAMES,
                mean_feature=cls,mean_patches=patches)

    def test_selected_checkpoint_must_stay_inside_training_and_match_its_digest(self):
        with temporary_directory() as temp:
            training = Path(temp)/'training'; training.mkdir()
            external = Path(temp)/'outside.pt';external.write_bytes(b'weights')
            record=dict(checkpoint=str(external),checkpoint_sha256=diagnostics.file_hash(external),complete=True)
            (training/'selected_vit.json').write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError,'identity/location'):
                diagnostics._selected_record(training)
            checkpoint=training/'best.pt';checkpoint.write_bytes(b'weights')
            record.update(checkpoint=str(checkpoint),checkpoint_sha256='wrong')
            (training/'selected_vit.json').write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError,'identity/location'):
                diagnostics._selected_record(training)

    def test_complete_analysis_writes_aggregates_and_reuses_unchanged_identity(self):
        with temporary_directory() as temp:
            root=Path(temp);training=root/'training';training.mkdir();old=root/'old';old.mkdir()
            parent=root/'parent';parent.mkdir();output=root/'analysis'
            marker=old/'marker.json';marker.write_text('{}')
            cnn,vit,dual=ToyCNN().eval(),ToyViT().eval(),ToyDual().eval()
            sources=dict(source_suite=str(old),parent_run=str(parent),parent_model=cnn.state_dict(),
                vit_model=vit.state_dict(),dual_best=dict(model=dual.state_dict(),epoch=30),
                source_files={str(marker):diagnostics.file_hash(marker)})
            plan=dict(source_suite=str(old),config=config(),source_hashes={'new':'sha'},
                source_provenance={'old':'identity'},seed=42,split_hash='same',profiles=['tiny'],
                vit_epochs=3,fusion_epochs=2)
            (training/'suite.json').write_text(json.dumps(plan))
            runs={'pretrain_tiny':vit,'cnn_frozen_head':cnn,'new_frozen_concat':dual}
            for job,model in runs.items():
                folder=training/job;folder.mkdir()
                for file,epoch in [('best.pt',1),('last.pt',2)]:
                    torch.save(dict(model=model.state_dict(),epoch=epoch),folder/file)
                (folder/'summary.json').write_text(json.dumps(dict(status='complete',best_epoch=1,completed_epochs=2)))
            selected=dict(protocol='selected',profile='tiny',job='pretrain_tiny',
                checkpoint=str(training/'pretrain_tiny'/'best.pt'),
                checkpoint_sha256=diagnostics.file_hash(training/'pretrain_tiny'/'best.pt'),complete=True)
            (training/'selected_vit.json').write_text(json.dumps(selected))
            probe,selection=Images(n=6),Images('selection',5)
            manifest=dict(train_clean=[f'train_clean/{i}.png' for i in range(6)],
                selection=[f'selection/{i}.png' for i in range(5)])
            def baselines(*args):
                results={}
                for split,dataset in [('train_probe',probe),('selection',selection)]:
                    results[split]={}
                    for key,model in [('cnn',cnn),('vit',vit)]:
                        values,_,_=diagnostics.evaluate_modes(model,dataset,config(),42,'cpu',NAMES)
                        results[split][key]=values['learned']
                return results,{}
            def analysis_model(job,run_dir,*args,checkpoint='best.pt',**kwargs):
                state=torch.load(Path(run_dir)/checkpoint,weights_only=False)
                model=copy.deepcopy(runs[job]);model.load_state_dict(state['model']);return model,state
            source_module=types.ModuleType('fusion_continuation_sources')
            source_module.load_sources=lambda *args,**kwargs:sources
            train_module=types.ModuleType('vit_augreg_experiment')
            train_module.load_analysis_model=analysis_model
            with patch.dict(sys.modules,{'fusion_continuation_sources':source_module,'vit_augreg_experiment':train_module}), \
                    patch.object(diagnostics,'clean_datasets',return_value=(probe,selection,None,'same',manifest)), \
                    patch.object(diagnostics,'class_labels',return_value=NAMES), \
                    patch.object(diagnostics,'_baselines',side_effect=baselines) as base, \
                    patch.object(diagnostics,'_prediction_source',return_value=None), \
                    patch.object(diagnostics,'build_followup_model',side_effect=lambda *a,**k:copy.deepcopy(dual)), \
                    patch('torch.cuda.is_available',return_value=False):
                result=diagnostics.run_augreg_diagnostics(plan,training,output,examples=0)
                self.assertEqual(result['status'],'complete')
                second=diagnostics.run_augreg_diagnostics(plan,training,output,examples=0)
                self.assertEqual(result,second)
                self.assertEqual(base.call_count,1)
            attempt=output/result['analysis_directory']
            self.assertTrue((attempt/'comparison.csv').is_file())
            self.assertTrue((attempt/'probability_fusion.csv').is_file())
            self.assertEqual(len((attempt/'probability_fusion.csv').read_text().splitlines()),33)
            self.assertEqual(len(result['threeway']),8)
            self.assertTrue((attempt/'new_frozen_concat'/'best'/'selection'/'vit_mismatch'/'predictions.npz').is_file())
            self.assertFalse(result['audit_partition_evaluated'])
            self.assertFalse(result['official_validation_evaluated'])


if __name__=='__main__': unittest.main()
