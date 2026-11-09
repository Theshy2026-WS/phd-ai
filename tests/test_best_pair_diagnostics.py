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

import best_pair_diagnostics as diagnostics
from models.attention_correction import state_digest
from test_vit_augreg_diagnostics import Images, ToyCNN, ToyDual, ToyViT, NAMES, config


TEST_ROOT = Path(__file__).resolve().parents[1]/'runs'/'best_pair_diagnostic_tests_tmp'


def temporary_directory():
    TEST_ROOT.mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(dir=TEST_ROOT)


class Concat(ToyDual):
    variant = 'concat'


class BestPairDiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def test_global_derangement_breaks_class_sorted_batch_confound(self):
        ids = [f'class_{label:03d}/{i:03d}.png' for label in range(200) for i in range(50)]
        targets = np.repeat(np.arange(200), 50)
        donors, policy = diagnostics.mismatch_policy(ids, targets, 730045)
        self.assertEqual(len(set(donors.tolist())), 10000)
        self.assertEqual(policy['same_image_pairs'], 0)
        self.assertLess(policy['same_class_fraction'], .02)
        self.assertAlmostEqual(policy['expected_same_class_fraction_under_random_distinct_donor'], 49/9999)
        self.assertFalse(policy['pairing_uses_labels'])
        self.assertFalse(policy['pairing_uses_batch_boundaries'])

    def test_global_pairs_are_deterministic_by_id_and_independent_of_labels(self):
        ids = [f'image{i}' for i in range(37)]
        targets = np.arange(37) % 3
        indexes, _ = diagnostics.mismatch_policy(ids, targets, 42)
        mapping = {target:ids[index] for target,index in zip(ids,indexes)}
        reversed_ids = ids[::-1]
        another = diagnostics.global_derangement(reversed_ids,42)
        self.assertEqual(mapping,{target:reversed_ids[index] for target,index in zip(reversed_ids,another)})
        changed, _ = diagnostics.mismatch_policy(ids, targets[::-1],42)
        np.testing.assert_array_equal(indexes,changed)
        self.assertFalse(np.array_equal(indexes,diagnostics.global_derangement(ids,43)))
        for bad in [[],['one'],['one','one']]:
            with self.assertRaises(ValueError): diagnostics.global_derangement(bad,42)

    def test_cached_modes_equal_direct_model_and_global_pairing_survives_singleton(self):
        torch.manual_seed(19)
        model = Concat().eval()
        cfg = config(); probe = Images(n=6); selection = Images('selection',5)
        before = state_digest(model)
        training = diagnostics.collect_concat_features(model,probe,cfg,42,'cpu')
        mean, provenance = diagnostics.training_mean(training)
        selected = diagnostics.collect_concat_features(model,selection,cfg,42,'cpu')
        extractions = model.extractions
        values,margins,policy = diagnostics.evaluate_cached_modes(model,selected,cfg,42,'cpu',NAMES,mean_feature=mean)
        self.assertEqual(model.extractions,extractions)
        images = torch.stack([selection[i][0] for i in range(len(selection))])
        with torch.inference_mode(): direct = model(images).softmax(1).numpy()
        np.testing.assert_allclose(values['learned']['probabilities'],direct,rtol=1e-6,atol=1e-7)
        np.testing.assert_allclose(margins['cnn_head_true_margin']+margins['vit_true_margin'],
            margins['full_true_margin'],rtol=1e-5,atol=1e-6)
        self.assertTrue(all(a != b for a,b in zip(selected['image_ids'],policy['donor_image_ids'])))
        self.assertEqual(policy['n'],5)
        self.assertEqual(state_digest(model),before)
        self.assertFalse(provenance['uses_selection_images'])
        with self.assertRaisesRegex(ValueError,'clean training probe'): diagnostics.training_mean(selected)
        changed = dict(cfg,dataset=dict(cfg['dataset'],batch_size=3))
        other,_,other_policy = diagnostics.evaluate_cached_modes(model,selected,changed,42,'cpu',NAMES,mean_feature=mean)
        self.assertEqual(other_policy['donor_image_ids'],policy['donor_image_ids'])
        np.testing.assert_allclose(other['global_mismatch']['probabilities'],
            values['global_mismatch']['probabilities'],rtol=1e-6,atol=1e-7)

    def test_training_mean_is_label_blind_and_nonconcat_cache_is_rejected(self):
        model = Concat().eval(); cfg = config()
        a = diagnostics.collect_concat_features(model,Images(),cfg,42,'cpu')
        b = diagnostics.collect_concat_features(model,Images(label_shift=1),cfg,42,'cpu')
        torch.testing.assert_close(diagnostics.training_mean(a)[0],diagnostics.training_mean(b)[0],rtol=0,atol=0)
        model.variant = 'cross_patch'
        with self.assertRaisesRegex(ValueError,'concat model only'):
            diagnostics.collect_concat_features(model,Images(),cfg,42,'cpu')

    def test_independent_reference_evaluation_is_read_only(self):
        model = ToyCNN().eval(); before=state_digest(model)
        result = diagnostics.evaluate_plain(model,Images('selection',5),config(),42,'cpu',NAMES)
        self.assertEqual(result['metrics']['n'],5)
        self.assertEqual(state_digest(model),before)

    def _report_fixture(self, previous=False):
        with temporary_directory() as temp:
            root=Path(temp); training=root/'training'; training.mkdir()
            cnn_source=root/'cnn_source'; cnn_source.mkdir(); vit_source=root/'vit_source'; vit_source.mkdir()
            marker=cnn_source/'marker.json'; marker.write_text('{}')
            cnn,vit,dual=ToyCNN().eval(),ToyViT().eval(),Concat().eval()
            sources=dict(parent_model=cnn.state_dict(),selected=dict(profile='toy',state=dict(model=vit.state_dict())),
                provenance={'same':'sources'},source_files={str(marker):diagnostics.file_hash(marker)})
            previous_model = copy.deepcopy(dual)
            with torch.no_grad(): previous_model.fusion_classifier.bias[0] += .5
            previous_state = dict(model=previous_model.state_dict(),epoch=2)
            if previous:
                sources['previous_joint'] = dict(state=previous_state,checkpoint_sha256='previous')
            plan=dict(cnn_suite=str(cnn_source),vit_suite=str(vit_source),config=config(),seed=42,
                split_hash='same',source_provenance=sources['provenance'],source_files=sources['source_files'],
                source_hashes={'current':'hash'})
            (training/'suite.json').write_text(json.dumps(plan))
            run=training/'new_joint_concat'; run.mkdir()
            for file,epoch in [('best.pt',2),('last.pt',30)]:
                torch.save(dict(model=dual.state_dict(),epoch=epoch),run/file)
            (run/'summary.json').write_text(json.dumps(dict(status='complete',best_epoch=2)))
            probe,selection=Images(n=6),Images('selection',5)
            manifest=dict(train_clean=[f'train_clean/{i}.png' for i in range(6)],
                selection=[f'selection/{i}.png' for i in range(5)])
            source_module=types.ModuleType('best_pair_sources')
            source_module.load_best_sources=lambda *a,**kw:sources
            training_module=types.ModuleType('best_pair_joint')
            def get_model(plan,training,checkpoint='best.pt',sources=None):
                state=torch.load(run/checkpoint,weights_only=False)
                model=copy.deepcopy(dual); model.load_state_dict(state['model']); return model,state
            training_module.load_best_pair_model=get_model
            with patch.dict(sys.modules,{'best_pair_sources':source_module,'best_pair_joint':training_module}), \
                    patch.object(diagnostics,'clean_datasets',return_value=(probe,selection,None,'same',manifest)), \
                    patch.object(diagnostics,'class_labels',return_value=NAMES), \
                    patch.object(diagnostics,'build_spatial_model',side_effect=lambda *a:copy.deepcopy(cnn)), \
                    patch.object(diagnostics,'build_vit',side_effect=lambda *a:copy.deepcopy(vit)), \
                    patch.object(diagnostics,'build_previous_joint_model',side_effect=lambda *a:
                        (copy.deepcopy(previous_model),previous_state) if previous else (None,None)), \
                    patch('torch.cuda.is_available',return_value=False):
                result=diagnostics.run_best_pair_diagnostics(plan,training,root/'analysis',examples=0)
                self.assertEqual(result['status'],'complete')
                second=diagnostics.run_best_pair_diagnostics(plan,training,root/'analysis',examples=0)
                self.assertEqual(result,second)
            attempt=root/'analysis'/result['analysis_directory']
            self.assertTrue((attempt/'best/selection/global_mismatch/predictions.npz').is_file())
            self.assertTrue((attempt/'best/selection/global_mismatch_pairs.csv').is_file())
            self.assertEqual(len((attempt/'probability_fusion.csv').read_text().splitlines()),17)
            self.assertEqual(len(result['threeway']),2)
            policy=json.loads((attempt/'best/selection/intervention_policy.json').read_text())
            self.assertEqual(policy['same_image_pairs'],0)
            self.assertFalse(policy['pairing_uses_batch_boundaries'])
            self.assertFalse(result['audit_partition_evaluated'])
            self.assertFalse(result['official_validation_evaluated'])
            self.assertEqual(result['previous_joint_reference_available'],previous)
            if previous:
                self.assertEqual(result['paired']['best/selection/vs_previous_joint']['n'],5)
                self.assertEqual(result['paired']['last_raw/selection/vs_previous_joint']['n'],5)
                self.assertTrue((attempt/'independent_references/previous_joint/selection/predictions.npz').is_file())

    def test_complete_report_and_resume_use_exact_source_and_checkpoint_identity(self):
        self._report_fixture()

    def test_previous_joint_reference_is_reevaluated_and_paired_against_new_results(self):
        self._report_fixture(previous=True)

    def test_previous_reference_loads_complete_old_cnn_and_vit_state(self):
        torch.manual_seed(7); previous=Concat().eval()
        torch.manual_seed(19); current=Concat().eval()
        expected=copy.deepcopy(previous.state_dict())
        self.assertFalse(torch.equal(current.encoder.project.weight,previous.encoder.project.weight))
        sources=dict(selected={'profile':'toy'},previous_joint={'state':dict(model=expected,epoch=2)})
        with patch('vit_augreg_experiment.build_job_model',return_value=current):
            actual,state=diagnostics.build_previous_joint_model(sources,42)
        self.assertEqual(state['epoch'],2)
        for name,value in actual.state_dict().items():
            torch.testing.assert_close(value,expected[name],rtol=0,atol=0)


if __name__=='__main__': unittest.main()
