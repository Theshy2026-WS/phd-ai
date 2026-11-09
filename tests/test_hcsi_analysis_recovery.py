import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import best_pair_joint
import hcsi_analysis_recovery as recovery
import hcsi_tiny_experiment


class RecoveryScopeTests(unittest.TestCase):
    def fixture(self, root):
        suite = root/'reference'
        checkpoint = suite/'training/new_joint_concat/best.pt'
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b'unchanged-joint-weights')
        reference = dict(suite=str(suite), checkpoint=str(checkpoint),
            checkpoint_sha256=recovery.file_hash(checkpoint), epoch=7, source='ema',
            selection=dict(accuracy=.6545, nll=1.793))
        independent = dict(provenance={'source':'original'}, source_files={'cnn':'c','vit':'v'},
            selected=dict(profile='vit_small_p8_augreg', checkpoint='vit', state={}),
            parent_completed_epochs=213)
        canonical = dict(source_provenance=independent['provenance'],
            source_files=independent['source_files'],
            selected_vit={k:v for k,v in independent['selected'].items() if k!='state'},
            parent_completed_epochs=213)
        enlarged = dict(independent, source_files=dict(independent['source_files'], reference='r'),
            reference=dict(record=reference))
        state = dict(epoch=7, source='ema', selection=reference['selection'])
        return dict(reference=reference), independent, canonical, enlarged, state

    def test_actual_identity_guard_reproduces_bug_and_canonical_scope_passes(self):
        with tempfile.TemporaryDirectory() as temp:
            plan, original, canonical, enlarged, state = self.fixture(Path(temp))
            snapshot = copy.deepcopy(enlarged)
            best_pair_joint._validate_source_identity(canonical, original)
            with self.assertRaisesRegex(ValueError, 'sources, selected checkpoint'):
                best_pair_joint._validate_source_identity(canonical, enlarged)
            calls=[]
            def canonical_loader(actual_plan, directory, checkpoint='best.pt', sources=None):
                calls.append(sources)
                best_pair_joint._validate_source_identity(actual_plan, original if sources is None else sources)
                return 'model', state
            with patch.object(best_pair_joint, 'validate_resume', return_value=canonical), \
                    patch.object(best_pair_joint, 'load_best_pair_model', side_effect=canonical_loader):
                self.assertEqual(recovery.load_reference_model(plan, enlarged), ('model', state))
            self.assertEqual(calls, [None]); self.assertEqual(enlarged, snapshot)

    def test_real_changed_source_is_still_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            plan, original, canonical, enlarged, state = self.fixture(Path(temp))
            changed=copy.deepcopy(original); changed['source_files']['cnn']='changed'
            def loader(actual_plan, directory):
                best_pair_joint._validate_source_identity(actual_plan, changed)
            with patch.object(best_pair_joint, 'validate_resume', return_value=canonical), \
                    patch.object(best_pair_joint, 'load_best_pair_model', side_effect=loader):
                with self.assertRaises(ValueError): recovery.load_reference_model(plan, enlarged)

    def test_changed_checkpoint_is_rejected_before_loading(self):
        with tempfile.TemporaryDirectory() as temp:
            plan, _, _, enlarged, _ = self.fixture(Path(temp))
            Path(plan['reference']['checkpoint']).write_bytes(b'tampered')
            with patch.object(best_pair_joint, 'validate_resume') as validate:
                with self.assertRaisesRegex(ValueError, 'bytes changed'):
                    recovery.load_reference_model(plan, enlarged)
            validate.assert_not_called()

    def test_reference_path_and_selection_cannot_be_substituted(self):
        with tempfile.TemporaryDirectory() as temp:
            plan, _, canonical, enlarged, state = self.fixture(Path(temp))
            bad=copy.deepcopy(plan);bad['reference']['checkpoint']=str(Path(temp)/'elsewhere.pt')
            with self.assertRaisesRegex(ValueError, 'path differs'): recovery.load_reference_model(bad)
            with patch.object(best_pair_joint, 'validate_resume', return_value=canonical), \
                    patch.object(best_pair_joint, 'load_best_pair_model', return_value=('model',dict(state,epoch=8))):
                with self.assertRaisesRegex(ValueError, 'selected checkpoint identity'):
                    recovery.load_reference_model(plan, enlarged)

    def test_scoped_adapter_restores_original_even_after_failure(self):
        old=hcsi_tiny_experiment.load_reference_model
        with self.assertRaisesRegex(RuntimeError, 'synthetic'):
            with recovery.reference_loader_scope():
                self.assertIs(hcsi_tiny_experiment.load_reference_model,recovery.load_reference_model)
                raise RuntimeError('synthetic')
        self.assertIs(hcsi_tiny_experiment.load_reference_model,old)

    def test_diagnostics_adapter_never_calls_training_and_restores_loader(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);plan,_,_,sources,_=self.fixture(root)
            training=root/'training';training.mkdir();output=root/'analysis'
            old=hcsi_tiny_experiment.load_reference_model
            def analyze(*args,**kwargs):
                self.assertIs(hcsi_tiny_experiment.load_reference_model,recovery.load_reference_model)
                return dict(status='complete')
            with patch.object(hcsi_tiny_experiment,'load_sources',return_value=sources), \
                    patch('hcsi_tiny_diagnostics.run_hcsi_diagnostics',side_effect=analyze), \
                    patch.object(hcsi_tiny_experiment,'train_job') as train:
                self.assertEqual(recovery.run_hcsi_recovery_diagnostics(plan,training,output)['status'],'complete')
            train.assert_not_called();self.assertIs(hcsi_tiny_experiment.load_reference_model,old)
            self.assertTrue((output/'analysis_recovery_implementation.json').is_file())

    def test_output_cannot_overlap_training(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            with self.assertRaisesRegex(ValueError,'separate'):
                recovery.run_hcsi_recovery_diagnostics({},root,root/'analysis')


if __name__ == '__main__': unittest.main()
