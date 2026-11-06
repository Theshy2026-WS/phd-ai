import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

import fusion_continuation_sources as sources
from test_baseline_diagnostics import config as original_config


class ContinuationSourceTests(unittest.TestCase):
    def fixture(self, root):
        root = Path(root)
        source, parent = root/'source', root/'parent'
        training = source/'training'
        for path in (training/'vit_pretrain', training/'dual_fusion', parent):
            path.mkdir(parents=True)
        config = original_config()
        config['training']['epochs'] = 100
        parent_hash = sources.file_hash(parent/'best.pt') if (parent/'best.pt').exists() else None
        (parent/'best.pt').write_bytes(b'original-cnn')
        parent_hash = sources.file_hash(parent/'best.pt')
        code = {name: sources.file_hash(sources.ROOT/name) for name in sources.MODEL_SOURCES}
        plan = dict(protocol=sources.SOURCE_PROTOCOL, parent_run=str(parent),
            parent_checkpoint_sha256=parent_hash, seed=42, split_hash='split',
            smoke=False, vit_epochs=100, fusion_epochs=30, execution_environment={'device':'fixture'})
        training_plan = dict(plan, protocol=sources.TRAINING_PROTOCOL, source_hashes=code)
        def save(path, value):
            path.write_text(json.dumps(value), encoding='utf-8')
        save(parent/'summary.json', {'source':'parent'})
        save(source/'suite.json', plan)
        save(source/'completion.json', dict(protocol=sources.SOURCE_PROTOCOL,status='complete'))
        save(training/'suite.json', training_plan)
        save(training/'completion.json', dict(protocol=sources.TRAINING_PROTOCOL,status='complete',failures={},skipped_jobs=[]))
        def make(job, epoch, accuracy, vit_hash=None):
            job_config = copy.deepcopy(config)
            job_config['training']['epochs'] = 100 if job == 'vit_pretrain' else 30
            metadata = dict(protocol=sources.TRAINING_PROTOCOL, job=job,
                variant='vit' if job == 'vit_pretrain' else 'dual_fusion',
                seed=42, split_hash='split', parent_checkpoint_sha256=parent_hash,
                source_hashes=code, recipe_hash=sources.recipe_hash(job_config),
                environment={'device':'fixture'}, initial_state_sha256='initial',
                vit_checkpoint_sha256=vit_hash)
            selected = dict(accuracy=accuracy,nll=1.,top5_accuracy=.9)
            state = dict(metadata=metadata, model={'weight':torch.ones(2,2)},
                epoch=epoch, selection=selected, source='raw',config=job_config)
            summary = dict(metadata, status='complete', completed_epochs=job_config['training']['epochs'],
                best_epoch=epoch,best_selection=selected)
            return state, summary
        vit, vit_summary = make('vit_pretrain',97,.4597)
        torch.save(vit,training/'vit_pretrain/best.pt')
        save(training/'vit_pretrain/summary.json',vit_summary)
        best, dual_summary = make('dual_fusion',30,.6469,sources.file_hash(training/'vit_pretrain/best.pt'))
        last = copy.deepcopy(best)
        last.update(ema_model={'weight':torch.full((2,2),.9)},
            optimizer={'state':{0:{'step':torch.tensor(100.),'exp_avg':torch.ones(2,2)}},
                       'param_groups':[{'role':role} for role in ('encoder','vit','fusion')]},
            scheduler={'last_epoch':30},scaler={},rng={'torch':torch.get_rng_state()},
            history=[dict(epoch=i,augmentation_epoch=100+i,selection_accuracy=.6469) for i in range(31)])
        torch.save(best,training/'dual_fusion/best.pt')
        torch.save(last,training/'dual_fusion/last.pt')
        save(training/'dual_fusion/summary.json',dual_summary)
        diagnostics = source/'feature_analysis'
        diagnostics.mkdir()
        recorded={job:{name:sources.file_hash(training/job/name) for name in names}
            for job,names in [('vit_pretrain',['best.pt']),('dual_fusion',['best.pt','last.pt'])]}
        save(diagnostics/'summary.json',dict(status='complete',identity=dict(
            parent_checkpoint_sha256=parent_hash,checkpoints=recorded)))
        loaded_parent=(parent,dict(seed=42,split_hash='split',completed_epochs=100),
                       dict(config=config,model={'parent.weight':torch.ones(1)}),parent_hash)
        return source, training, loaded_parent

    def load(self, source, parent, **kwargs):
        with patch.object(sources,'load_spatial_parent',return_value=parent):
            return sources.load_sources(source,**kwargs)

    def test_full_state_and_immutable_file_provenance(self):
        with tempfile.TemporaryDirectory() as temp:
            source,training,parent=self.fixture(temp)
            before={str(p):sources.file_hash(p) for p in source.rglob('*') if p.is_file()}
            loaded=self.load(source,parent)
            self.assertEqual(loaded['dual_completed_epochs'],30)
            self.assertEqual(loaded['parent_completed_epochs'],100)
            self.assertEqual(loaded['dual_last']['scheduler']['last_epoch'],30)
            self.assertEqual(loaded['dual_last']['optimizer']['state'][0]['step'],100.)
            self.assertEqual(loaded['provenance']['source_joint_last_accuracy'],.6469)
            self.assertEqual(loaded['provenance']['source_execution_environment'],{'device':'fixture'})
            self.assertEqual(before,{str(p):sources.file_hash(p) for p in source.rglob('*') if p.is_file()})
            self.assertIn(str((training/'dual_fusion/last.pt').resolve()),loaded['source_files'])

    def test_smoke_and_incomplete_source_rejected_before_checkpoint_load(self):
        with tempfile.TemporaryDirectory() as temp:
            source,_,parent=self.fixture(temp)
            path=source/'suite.json'
            plan=json.loads(path.read_text())
            plan['smoke']=True
            path.write_text(json.dumps(plan))
            with patch.object(sources,'_checkpoint') as checkpoint:
                with self.assertRaisesRegex(ValueError,'non-smoke'):
                    self.load(source,parent)
                checkpoint.assert_not_called()

    def test_missing_training_state_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            source,training,parent=self.fixture(temp)
            path=training/'dual_fusion/last.pt'
            state=torch.load(path,weights_only=False)
            state.pop('ema_model')
            torch.save(state,path)
            with self.assertRaisesRegex(ValueError,'continuation state'):
                self.load(source,parent)

    def test_changed_checkpoint_bytes_rejected_against_existing_diagnostics(self):
        with tempfile.TemporaryDirectory() as temp:
            source,training,parent=self.fixture(temp)
            path=training/'dual_fusion/best.pt'
            state=torch.load(path,weights_only=False)
            state['model']['weight'].add_(.01)
            torch.save(state,path)
            with self.assertRaisesRegex(ValueError,'changed since feature analysis'):
                self.load(source,parent)

    def test_new_epoch_schedule_cannot_be_disguised_as_original_source(self):
        with tempfile.TemporaryDirectory() as temp:
            source,training,parent=self.fixture(temp)
            path=training/'dual_fusion/last.pt'
            state=torch.load(path,weights_only=False)
            state['history'][-1]['augmentation_epoch']=30
            torch.save(state,path)
            with self.assertRaisesRegex(ValueError,'101..130'):
                self.load(source,parent)

    def test_recipe_changes_rejected_but_loader_relocation_allowed(self):
        with tempfile.TemporaryDirectory() as temp:
            source,_,parent=self.fixture(temp)
            requested=copy.deepcopy(parent[2]['config'])
            requested['dataset']['data_dir']='new-data-path'
            loaded=self.load(source,parent,config=requested)
            self.assertEqual(loaded['config']['dataset']['data_dir'],'new-data-path')
            requested['training']['weight_decay']+=.1
            with self.assertRaisesRegex(ValueError,'relocation'):
                self.load(source,parent,config=requested)


if __name__=='__main__':
    unittest.main()
