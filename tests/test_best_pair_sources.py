import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

import best_pair_sources as sources
from test_baseline_diagnostics import config as original_config


PROFILE = 'vit_small_p8_augreg'


class BestPairSourceTests(unittest.TestCase):
    def fixture(self, root):
        root = Path(root)
        cnn_source, vit_source = root/'cnn-suite', root/'vit-suite'
        cnn_run = cnn_source/'training'/sources.CNN_JOB
        vit_run = vit_source/'training'/('pretrain_'+PROFILE)
        for path in (cnn_run, vit_run): path.mkdir(parents=True)
        config = original_config()
        config['training']['epochs'] = 100
        environment = {'device':'fixture', 'dtype':'torch.float16'}
        code = {'models/vit.py': sources.file_hash(sources.ROOT/'models/vit.py')}
        ancestor = root/'ancestor.json'; ancestor.write_text('{}')
        common = dict(config=config, seed=42, split_hash='split', smoke=False,
            source_hashes=code, execution_environment=environment,
            source_files={str(ancestor):sources.file_hash(ancestor)})
        cnn_plan = dict(common, protocol=sources.CNN_SUITE_PROTOCOL, continue_epochs=100,
            control_epochs=30, source_suite='original-source')
        cnn_recorded = dict(cnn_plan, protocol=sources.CNN_TRAINING_PROTOCOL)
        vit_plan = dict(common, protocol=sources.VIT_SUITE_PROTOCOL, profiles=[PROFILE], vit_epochs=300)
        def save(path, value): path.write_text(json.dumps(value), encoding='utf-8')
        for source, plan, recorded, training_protocol in (
                (cnn_source, cnn_plan, cnn_recorded, sources.CNN_TRAINING_PROTOCOL),
                (vit_source, vit_plan, vit_plan, sources.VIT_TRAINING_PROTOCOL)):
            save(source/'suite.json', plan)
            save(source/'completion.json', dict(protocol=plan['protocol'], status='complete'))
            save(source/'training/suite.json', recorded)
            save(source/'training/completion.json', dict(protocol=training_protocol, status='complete', failures={}, skipped_jobs=[]))
        def checkpoint(run, plan, protocol, job, variant, epochs, best_epoch, score, model, **extra):
            job_config = copy.deepcopy(config); job_config['training']['epochs'] = epochs
            metadata = dict(protocol=protocol, job=job, variant=variant,
                seed=42, split_hash='split', environment=environment, source_hashes=code,
                recipe_hash=sources.recipe_hash(job_config), parent_checkpoint_sha256='original-cnn', **extra)
            selected = dict(accuracy=score, nll=1., top5_accuracy=.9)
            state = dict(metadata=metadata, model=model, config=job_config, epoch=best_epoch,
                         source='ema' if job == sources.CNN_JOB else 'raw', selection=selected)
            summary = dict(metadata, status='complete', completed_epochs=epochs,
                best_epoch=best_epoch, best_source=state['source'], best_selection=selected,
                total_train_seconds=100.)
            if job == sources.CNN_JOB: summary['best_global_adaptation_epoch'] = 30+best_epoch
            torch.save(state, run/'best.pt')
            save(run/'metadata.json', metadata); save(run/'summary.json', summary)
            return state, summary
        cnn, cnn_summary = checkpoint(cnn_run, cnn_recorded, sources.CNN_TRAINING_PROTOCOL,
            sources.CNN_JOB, 'cnn_head_control', 100, 83, .648,
            {'encoder.stem.weight':torch.ones(2,2), 'encoder.head.dense.weight':torch.full((3,2),.7),
             'encoder.head.dense.bias':torch.tensor([.1,.2,.3])},
            augmentation_epoch_offset=130, inherited_adaptation_epochs=30)
        vit, vit_summary = checkpoint(vit_run, vit_plan, sources.VIT_TRAINING_PROTOCOL,
            'pretrain_'+PROFILE, PROFILE, 300, 273, .5018,
            {'classifier.weight':torch.ones(3,2), 'class_token':torch.zeros(1,1,2)},
            scope='pretrain', external_pretraining=False, profile=PROFILE)
        candidate = dict(profile=PROFILE, job='pretrain_'+PROFILE, declaration_index=0,
            checkpoint=str((vit_run/'best.pt').resolve()), checkpoint_sha256=sources.file_hash(vit_run/'best.pt'),
            best_epoch=273, best_source='raw', best_selection=vit['selection'])
        selected = dict(candidate, protocol=sources.VIT_TRAINING_PROTOCOL, complete=True, candidates=[candidate])
        save(vit_source/'training/selected_vit.json', selected)
        for source, identity in (
                (cnn_source, dict(checkpoints={sources.CNN_JOB:{'best.pt':sources.file_hash(cnn_run/'best.pt')}})),
                (vit_source, dict(selected_vit=selected))):
            (source/'feature_analysis').mkdir()
            save(source/'feature_analysis/summary.json', dict(status='complete', identity=identity))
        return cnn_source, vit_source, cnn_run, vit_run, cnn, vit

    def load(self, cnn, vit, **kwargs):
        with patch.object(sources, '_validate_models') as validate:
            result = sources.load_best_sources(cnn, vit, **kwargs)
        return result, validate

    def add_previous_joint(self, vit_source, vit_state):
        run = vit_source/'training/new_joint_concat'; run.mkdir()
        for path in (vit_source/'suite.json', vit_source/'training/suite.json'):
            plan = json.loads(path.read_text())
            plan['source_provenance'] = {'parent_checkpoint_sha256':'original-cnn'}
            plan['fusion_epochs'] = 30
            path.write_text(json.dumps(plan))
        state = copy.deepcopy(vit_state)
        state['config']['training']['epochs'] = 30
        state['metadata'].update(job='new_joint_concat', variant='concat', scope='fusion',
            vit_checkpoint_sha256=sources.file_hash(vit_source/'training'/('pretrain_'+PROFILE)/'best.pt'),
            recipe_hash=sources.recipe_hash(state['config']))
        state.update(epoch=2, source='ema', selection=dict(accuracy=.6525, nll=1.705, top5_accuracy=.8379))
        summary = dict(state['metadata'], status='complete', completed_epochs=30,
            best_epoch=2, best_source='ema', best_selection=state['selection'])
        torch.save(state, run/'best.pt')
        (run/'metadata.json').write_text(json.dumps(state['metadata']))
        (run/'summary.json').write_text(json.dumps(summary))
        diagnostic_path = vit_source/'feature_analysis/summary.json'
        diagnostic = json.loads(diagnostic_path.read_text())
        diagnostic['identity']['checkpoints'] = {'new_joint_concat':{'best.pt':sources.file_hash(run/'best.pt')}}
        diagnostic_path.write_text(json.dumps(diagnostic))
        return run, state, summary

    def test_exact_selected_weights_and_chronology_are_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            cnn, vit, cnn_run, vit_run, cnn_state, _ = self.fixture(temporary)
            before = {str(p):sources.file_hash(p) for p in Path(temporary).rglob('*') if p.is_file()}
            result, validate = self.load(cnn, vit)
            self.assertEqual(result['parent_completed_epochs'], 213)
            self.assertEqual(result['provenance']['source_cnn_completed_training_epochs'], 230)
            self.assertEqual(result['provenance']['source_cnn_best_adaptation_epoch'], 113)
            self.assertEqual(result['provenance']['parent_checkpoint_sha256'], sources.file_hash(cnn_run/'best.pt'))
            self.assertEqual(result['selected']['checkpoint_sha256'], sources.file_hash(vit_run/'best.pt'))
            self.assertEqual(result['selected']['state']['epoch'], 273)
            self.assertFalse(any(key.startswith('encoder.') for key in result['parent_model']))
            torch.testing.assert_close(result['parent_model']['head.dense.weight'],
                cnn_state['model']['encoder.head.dense.weight'], rtol=0, atol=0)
            torch.testing.assert_close(result['parent_model']['head.dense.bias'],
                cnn_state['model']['encoder.head.dense.bias'], rtol=0, atol=0)
            self.assertIn(str((cnn_run/'best.pt').resolve()), result['source_files'])
            self.assertIn(str((vit_run/'best.pt').resolve()), result['source_files'])
            validate.assert_called_once()
            self.assertEqual(before, {str(p):sources.file_hash(p) for p in Path(temporary).rglob('*') if p.is_file()})

    def test_incomplete_or_smoke_source_fails_before_loading_weights(self):
        with tempfile.TemporaryDirectory() as temporary:
            cnn, vit, *_ = self.fixture(temporary)
            p = vit/'suite.json'; plan = json.loads(p.read_text()); plan['smoke'] = True
            p.write_text(json.dumps(plan))
            with patch.object(sources, '_checkpoint') as checkpoint:
                with self.assertRaisesRegex(ValueError, 'complete non-smoke'):
                    self.load(cnn, vit)
                checkpoint.assert_not_called()

    def test_historical_checkpoint_mutation_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            cnn, vit, *_ = self.fixture(temporary)
            (Path(temporary)/'ancestor.json').write_text('{"changed":true}')
            with self.assertRaisesRegex(ValueError, 'Historical source'):
                self.load(cnn, vit)

    def test_best_cnn_bytes_checked_against_prior_diagnostics(self):
        with tempfile.TemporaryDirectory() as temporary:
            cnn, vit, run, _, state, _ = self.fixture(temporary)
            state['model']['encoder.stem.weight'].add_(.1)
            torch.save(state, run/'best.pt')
            with self.assertRaisesRegex(ValueError, 'changed since its completed feature analysis'):
                self.load(cnn, vit)

    def test_fused_cnn_state_or_wrong_epoch_cannot_be_used(self):
        with tempfile.TemporaryDirectory() as temporary:
            cnn, vit, run, _, state, _ = self.fixture(temporary)
            state['model']['vit.weight'] = torch.ones(2)
            torch.save(state, run/'best.pt')
            with self.assertRaisesRegex(ValueError, 'not a fused model'):
                self.load(cnn, vit)
            state['model'].pop('vit.weight'); state['epoch'] = 100
            torch.save(state, run/'best.pt')
            with self.assertRaisesRegex(ValueError, 'chronology'):
                self.load(cnn, vit)

    def test_selected_vit_must_be_independent_and_match_declared_search(self):
        with tempfile.TemporaryDirectory() as temporary:
            cnn, vit, _, run, _, _ = self.fixture(temporary)
            p = vit/'training/selected_vit.json'; selected = json.loads(p.read_text())
            selected['checkpoint_sha256'] = 'wrong'; p.write_text(json.dumps(selected))
            with self.assertRaisesRegex(ValueError, 'Selected ViT record differs'):
                self.load(cnn, vit)
            selected['checkpoint_sha256'] = sources.file_hash(run/'best.pt')
            selected['candidates'] = []; p.write_text(json.dumps(selected))
            with self.assertRaisesRegex(ValueError, 'lacks all scheduled candidates'):
                self.load(cnn, vit)

    def test_only_dataset_relocation_is_allowed(self):
        with tempfile.TemporaryDirectory() as temporary:
            cnn, vit, *_ = self.fixture(temporary)
            original = json.loads((vit/'suite.json').read_text())['config']
            requested = copy.deepcopy(original); requested['dataset']['data_dir'] = 'other-data'
            loaded, _ = self.load(cnn, vit, config=requested)
            self.assertEqual(loaded['config']['dataset']['data_dir'], 'other-data')
            requested['training']['weight_decay'] += .1
            with self.assertRaisesRegex(ValueError, 'relocation'):
                self.load(cnn, vit, config=requested)

    def test_missing_or_nonfinite_weights_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            cnn, vit, run, _, state, _ = self.fixture(temporary)
            state['model']['encoder.stem.weight'].fill_(float('nan'))
            torch.save(state, run/'best.pt')
            with self.assertRaisesRegex(ValueError, 'non-finite'):
                self.load(cnn, vit)

    def test_code_hashes_accept_lf_and_detect_semantic_change(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'implementation.py'; path.write_bytes(b'a = 1\r\n')
            digest = sources.hashlib.sha256(b'a = 1\n').hexdigest()
            with patch.object(sources, 'ROOT', Path(temporary)):
                sources._hash_code({'implementation.py':digest})
                path.write_bytes(b'a = 2\r\n')
                with self.assertRaisesRegex(ValueError, 'implementation changed'):
                    sources._hash_code({'implementation.py':digest})

    def test_different_source_environment_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            cnn, vit, *_ = self.fixture(temporary)
            for path in (vit/'suite.json', vit/'training/suite.json'):
                plan = json.loads(path.read_text())
                plan['execution_environment'] = {'device':'different GPU'}
                path.write_text(json.dumps(plan))
            with self.assertRaisesRegex(ValueError, 'different execution_environment'):
                self.load(cnn, vit)

    def test_previous_joint_is_validated_and_included_as_read_only_reference(self):
        with tempfile.TemporaryDirectory() as temporary:
            cnn, vit, _, _, _, vit_state = self.fixture(temporary)
            run, state, _ = self.add_previous_joint(vit, vit_state)
            before = sources.file_hash(run/'best.pt')
            result, _ = self.load(cnn, vit)
            previous = result['previous_joint']
            self.assertEqual(previous['state']['selection']['accuracy'], .6525)
            self.assertEqual(previous['state']['epoch'], 2)
            self.assertEqual(previous['summary']['completed_epochs'], 30)
            self.assertEqual(result['provenance']['previous_joint_info']['checkpoint_sha256'], before)
            self.assertIn(str((run/'best.pt').resolve()), result['source_files'])
            self.assertEqual(before, sources.file_hash(run/'best.pt'))
            state['metadata']['vit_checkpoint_sha256'] = 'different-vit'
            torch.save(state, run/'best.pt')
            summary = json.loads((run/'summary.json').read_text()); summary['vit_checkpoint_sha256'] = 'different-vit'
            (run/'summary.json').write_text(json.dumps(summary))
            (run/'metadata.json').write_text(json.dumps(state['metadata']))
            with self.assertRaisesRegex(ValueError, 'different CNN/ViT dependency'):
                self.load(cnn, vit)


if __name__ == '__main__': unittest.main()
