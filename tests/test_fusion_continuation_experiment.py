import copy
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

import fusion_continuation_experiment as runner
from datasets.utility_tiny import UtilityDataset
from test_baseline_diagnostics import config as original_config, fixture_datasets


class ToyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Linear(3, 4), nn.GELU(), nn.Linear(4, 3))
        self.head = nn.Module()
        self.head.dense = nn.Linear(3, 3)

    def forward(self, images):
        return self.head.dense(self.features(images.mean((-2, -1))))


class ToyModel(nn.Module):
    def __init__(self, dual=True):
        super().__init__()
        self.variant = 'dual_fusion' if dual else 'cnn_head_control'
        self.encoder = ToyEncoder()
        if dual:
            self.vit = ToyEncoder()
            self.fusion_classifier = nn.Linear(6, 3)
        self.attention_flops_per_image = 0

    def forward(self, images):
        cnn = self.encoder(images)
        if not hasattr(self, 'vit'):
            return cnn
        return self.fusion_classifier(torch.cat((cnn, self.vit(images)), 1))


def factory(job, sources, seed=None):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(sources['seed'] if seed is None else seed)
        model = ToyModel(not job.startswith('cnn_'))
    if job == 'dual_continue':
        model.load_state_dict(sources['dual_last']['model'], strict=True)
    return model


def base_config():
    config = original_config()
    config['dataset'].update(batch_size=2, num_workers=0)
    config['training'].update(mixed_precision=False, gradient_clip_norm=5.)
    return config


def inherited_source():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        model = ToyModel()
    config = runner.job_config(base_config(), 'dual_continue', 30)
    optimizer = runner.make_optimizer(model, config)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    loss = sum(p.square().sum() for p in model.parameters())
    loss.backward()
    optimizer.step()
    runner.update_ema(ema, model)
    return dict(model=copy.deepcopy(model.state_dict()), ema_model=copy.deepcopy(ema.state_dict()),
        optimizer=copy.deepcopy(optimizer.state_dict()), scaler={}, scheduler={'last_epoch':30,'_last_lr':[1e-6,1e-5]},
        rng=runner.rng_state('cpu'), epoch=30)


class ContinuationOptimizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_exact_queue_excludes_native64_and_invalid_jobs(self):
        self.assertEqual(runner.DEFAULT_JOBS,
            ('dual_continue','cnn_head_control','cnn_head_continue','dual_frozen_head'))
        for job in ('shallow64','shallow64_attention','cross16_mlp','unknown'):
            with self.assertRaises(ValueError): runner.split_job(job)
        for kwargs in (dict(continue_epochs=101),dict(control_epochs=31),dict(jobs=[]),
                       dict(jobs=['cnn_head_continue']),dict(jobs=['cnn_head_continue','cnn_head_control'])):
            with self.assertRaises(ValueError): runner.run_continuation_study('unused','unused',**kwargs)

    def test_cnn_original_head_has_high_lr_and_all_parameters_once(self):
        model = ToyModel(False)
        config = runner.job_config(base_config(),'cnn_head_control',30)
        optimizer = runner.make_optimizer(model,config)
        classifier_ids = {id(p) for p in model.encoder.head.dense.parameters()}
        seen = set()
        for group in optimizer.param_groups:
            for p in group['params']:
                self.assertNotIn(id(p),seen)
                seen.add(id(p))
                self.assertEqual(group['role'],'fusion' if id(p) in classifier_ids else 'encoder')
                self.assertEqual(group['lr'],5e-4 if id(p) in classifier_ids else 5e-5)
        self.assertEqual(seen,{id(p) for p in model.parameters()})

    def test_inherited_moments_retained_without_aliasing_and_lr_restarted(self):
        source = inherited_source()
        before = copy.deepcopy(source['optimizer'])
        model = ToyModel()
        model.load_state_dict(source['model'])
        config = runner.job_config(base_config(),'dual_continue',100)
        optimizer = runner.make_optimizer(model,config)
        runner.restore_inherited_optimizer(optimizer,source,config)
        self.assertTrue(all(float(s['step'])==1 for s in optimizer.state.values()))
        scheduler = runner.make_scheduler(optimizer,config)
        self.assertAlmostEqual(runner._rates(optimizer)['encoder_lr'],1e-6)
        self.assertAlmostEqual(runner._rates(optimizer)['fusion_lr'],1e-5)
        optimizer.zero_grad()
        sum(p.square().sum() for p in model.parameters()).backward()
        optimizer.step()
        self.assertTrue(all(float(s['step'])==2 for s in optimizer.state.values()))
        for key,state in before['state'].items():
            for field,value in state.items():
                if torch.is_tensor(value):
                    torch.testing.assert_close(source['optimizer']['state'][key][field],value,rtol=0,atol=0)
        for _ in range(2):
            scheduler.step()
        self.assertAlmostEqual(runner._rates(optimizer)['encoder_lr'],5e-5)
        self.assertAlmostEqual(runner._rates(optimizer)['fusion_lr'],5e-4)

    def test_optimizer_role_or_shape_changes_are_rejected(self):
        source = inherited_source()
        config = runner.job_config(base_config(),'dual_continue',100)
        altered = copy.deepcopy(source)
        altered['optimizer']['param_groups'][0]['role']='fusion'
        with self.assertRaisesRegex(ValueError,'role/order'):
            runner.restore_inherited_optimizer(runner.make_optimizer(ToyModel(),config),altered,config)
        altered = copy.deepcopy(source)
        identifier = altered['optimizer']['param_groups'][0]['params'][0]
        altered['optimizer']['state'][identifier]['exp_avg']=torch.zeros(17)
        with self.assertRaisesRegex(ValueError,'tensor shape'):
            runner.restore_inherited_optimizer(runner.make_optimizer(ToyModel(),config),altered,config)

    def test_frozen_real_model_keeps_bn_dropout_weights_and_ema_exact(self):
        sources=dict(parent_model=None,vit_model=None,seed=42)
        model=runner.build_job_model('dual_frozen_head',sources)
        config=runner.job_config(base_config(),'dual_frozen_head',30)
        before=runner._frozen_digests(model)
        ema=copy.deepcopy(model).eval().requires_grad_(False)
        optimizer=runner.make_optimizer(model,config)
        self.assertEqual({g['role'] for g in optimizer.param_groups},{'fusion'})
        self.assertFalse(model.encoder.training)
        self.assertFalse(model.vit.training)
        _,amp,dtype,scaler=runner.runtime(config,42)
        batches=[(torch.randn(2,3,64,64),torch.tensor([0,1]),['one','two'])]
        runner.train_one_epoch(model,ema,batches,config,42,101,'cpu',amp,dtype,optimizer,scaler)
        self.assertEqual(before,runner._frozen_digests(model))
        self.assertEqual(before,runner._frozen_digests(ema))
        self.assertTrue(all(p.grad is None for p in model.encoder.parameters()))
        self.assertTrue(all(p.grad is None for p in model.vit.parameters()))
        self.assertTrue(all(p.grad is not None for p in model.fusion_classifier.parameters()))
        self.assertFalse(model.encoder.training)
        self.assertFalse(model.vit.training)


class ContinuationTransactionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def sources(self):
        return dict(config=base_config(),seed=42,split_hash='fixture-split',parent_run='original-cnn8',
            parent_completed_epochs=100,dual_completed_epochs=30,parent_model={},vit_model={},
            dual_last=inherited_source(),dual_best={},provenance={'identity':'source'},source_files={})

    def plan(self,sources):
        return dict(config=sources['config'],continue_epochs=3,control_epochs=2,seed=42,
            split_hash='fixture-split',parent_run='original-cnn8',parent_checkpoint_sha256='original-hash',
            source_provenance=sources['provenance'],source_files={},source_hashes={'source':'fixture'},examples=0)

    def supplied(self,root):
        supplied,manifest,fingerprint=fixture_datasets(root)
        supplied['train_probe']=UtilityDataset(supplied['train'].folder,[0,4,8],'train_probe',42)
        return supplied,manifest,fingerprint

    def fake_run(self,root,job='dual_continue',interrupted=False,race=False,improve=True,prior_failure=None):
        root.mkdir(parents=True,exist_ok=True)
        sources=self.sources()
        # Keep every comparison branch rooted in precisely the same source RNG.
        random.seed(123);np.random.seed(123);torch.manual_seed(123)
        sources['dual_last']['rng']=runner.rng_state('cpu')
        plan=self.plan(sources)
        supplied=self.supplied(root)
        inherited=sources['dual_last'] if job=='dual_continue' else None
        progress=dict(epoch=0,interrupted=interrupted,race=race,early_failure=prior_failure=='before_last')
        if prior_failure=='only':
            self.assertTrue(runner._record_failure(plan,job,root/'run',{'error_type':'fixture'}))
        epochs=[]
        initial_steps=[]

        def train(model,ema,batches,config,seed,data_epoch,device,amp,dtype,optimizer,scaler):
            local=data_epoch-(130 if job=='dual_continue' else 100)
            if local==2 and progress['interrupted']:
                progress['interrupted']=False
                raise KeyboardInterrupt()
            progress['epoch']=local
            epochs.append(data_epoch)
            initial_steps.append([float(s['step']) for s in optimizer.state.values()])
            optimizer.zero_grad()
            coefficient=random.random()+np.random.rand()
            loss=sum((p*torch.randn_like(p)*coefficient).sum() for p in model.parameters() if p.requires_grad)
            loss.backward();optimizer.step();runner.update_ema(ema,model)
            return dict(train_soft_target_loss=1.,skipped_amp_steps=0)

        def evaluate(model,batches,*args):
            if batches.split=='train_probe':
                random.random();np.random.rand();torch.rand(2)
                return dict(accuracy=1.,top5_accuracy=1.,nll=.01)
            values={0:.7,1:.8,2:.9,3:.6} if improve else {0:.7,1:.6,2:.5,3:.4}
            score=values[progress['epoch']]
            if progress['epoch']==0 and not any(p.requires_grad for p in model.parameters()): score=.72
            return dict(accuracy=score,top5_accuracy=1.,nll=.5)

        real_save=runner.atomic_save
        def save(path,value):
            if Path(path).name=='last.pt' and value['epoch']==0 and progress['early_failure']:
                progress['early_failure']=False
                raise RuntimeError('failure before first phase checkpoint')
            if Path(path).name=='last.pt' and value['epoch']==2 and progress['race']:
                progress['race']=False
                raise KeyboardInterrupt()
            return real_save(path,value)

        with patch.object(runner,'prepare_datasets',return_value=supplied), \
                patch.object(runner,'build_job_model',side_effect=factory), \
                patch.object(runner,'loader',side_effect=lambda ds,*args:ds), \
                patch.object(runner,'train_one_epoch',side_effect=train), \
                patch('generalization_experiment.evaluate',side_effect=evaluate), \
                patch('generalization_experiment.loader',side_effect=lambda ds,*args:ds), \
                patch('spatial_experiment.evaluate',side_effect=evaluate), \
                patch('spatial_experiment.loader',side_effect=lambda ds,*args:ds), \
                patch.object(runner,'_cost',return_value={'forward_flops_estimate':1000}), \
                patch.object(runner,'atomic_save',side_effect=save):
            if prior_failure=='before_last':
                with self.assertRaisesRegex(RuntimeError,'before first'):
                    runner._train_job(plan,job,root/'run',sources,inherited,'source-last' if inherited else None)
                self.assertTrue(runner._record_failure(plan,job,root/'run',{'error_type':'fixture'}))
            if interrupted or race:
                with self.assertRaises(KeyboardInterrupt):
                    runner._train_job(plan,job,root/'run',sources,inherited,'source-last' if inherited else None)
            summary=runner._train_job(plan,job,root/'run',sources,inherited,'source-last' if inherited else None)
        state=torch.load(root/'run/last.pt',weights_only=False)
        self.assertEqual(supplied[0]['audit'].reads,0)
        self.assertEqual(supplied[0]['official_validation'].reads,0)
        return summary,state,epochs,initial_steps,root/'run',plan,sources

    def test_extra_epochs_inherit_moments_ema_and_correct_time_axis(self):
        with tempfile.TemporaryDirectory() as temp:
            summary,state,epochs,steps,run,_,_=self.fake_run(Path(temp))
            self.assertEqual(epochs,[131,132,133])
            self.assertTrue(all(value==1 for value in steps[0]))
            self.assertTrue(all(float(s['step'])==4 for s in state['optimizer']['state'].values()))
            self.assertEqual(summary['total_adaptation_epochs'],33)
            self.assertEqual(summary['best_global_adaptation_epoch'],32)
            self.assertEqual(summary['initial_selection']['accuracy'],.7)
            self.assertEqual(summary['initial_ema_selection']['accuracy'],.72)
            self.assertTrue(summary['optimizer_restored_from_source'])
            self.assertFalse(summary['source_scheduler_reused'])
            self.assertEqual(summary['completed_epochs'],3)
            self.assertTrue((run/'best_trained.pt').is_file())
            self.assertNotEqual(summary['diagnostic_directory'],summary['last_diagnostic_directory'])

    def test_best_initial_ema_is_not_called_new_training(self):
        with tempfile.TemporaryDirectory() as temp:
            summary,_,_,_,run,_,_=self.fake_run(Path(temp),improve=False)
            self.assertEqual(summary['best_epoch'],0)
            self.assertEqual(summary['best_source'],'ema')
            self.assertEqual(summary['best_selection']['accuracy'],.72)
            self.assertEqual(summary['best_trained_selection']['epoch'],1)
            self.assertEqual(torch.load(run/'best_trained.pt',weights_only=False)['epoch'],1)

    def test_interrupt_and_checkpoint_commit_race_resume_exactly(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            _,full,*_=self.fake_run(root/'full')
            _,interrupted,*_=self.fake_run(root/'interrupted',interrupted=True)
            _,race,*_=self.fake_run(root/'race',race=True)
            for key,value in full['model'].items():
                torch.testing.assert_close(value,interrupted['model'][key],rtol=0,atol=0)
                torch.testing.assert_close(value,race['model'][key],rtol=0,atol=0)
                torch.testing.assert_close(full['ema_model'][key],race['ema_model'][key],rtol=0,atol=0)

    def test_head_control_is_fresh_and_uses_original_data_epoch_range(self):
        with tempfile.TemporaryDirectory() as temp:
            summary,_,epochs,steps,_,_,_=self.fake_run(Path(temp),job='cnn_head_control')
            self.assertEqual(epochs,[101,102])
            self.assertEqual(steps[0],[])
            self.assertFalse(summary['optimizer_restored_from_source'])
            self.assertEqual(summary['total_adaptation_epochs'],2)

    def test_cnn_dependency_requires_completed_matching_source(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            summary,_,_,_,run,plan,_=self.fake_run(root/'fixture',job='cnn_head_control')
            run.rename(root/'cnn_head_control')
            state,digest=runner._load_cnn_dependency(plan,root)
            self.assertEqual(state['epoch'],2)
            self.assertEqual(digest,runner.file_hash(root/'cnn_head_control/last.pt'))
            changed=copy.deepcopy(plan);changed['source_provenance']={'different':'source'}
            with self.assertRaisesRegex(ValueError,'provenance'):
                runner._load_cnn_dependency(changed,root)

    def test_main_failure_keeps_independent_frozen_head_job_and_source_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            source=root/'source';source.mkdir()
            sources=self.sources()
            sources.update(source_suite=str(source),source_training_dir=str(source/'training'))
            calls=[]
            def train(plan,job,run,supplied,inherited,inherited_hash):
                calls.append(job)
                if job=='dual_continue':raise RuntimeError('synthetic continuation failure')
                return dict(status='complete',completed_epochs=30)
            with patch.object(runner,'load_sources',return_value=sources), \
                    patch.object(runner,'source_hashes',return_value={'fixture':'source'}), \
                    patch.object(runner,'execution_environment',return_value={}), \
                    patch.object(runner,'_make_reference'),patch.object(runner,'comparison'), \
                    patch.object(runner,'_train_job',side_effect=train), \
                    patch.object(runner.torch.cuda,'is_available',return_value=True), \
                    patch.object(runner.torch.cuda,'empty_cache'):
                with self.assertRaisesRegex(RuntimeError,'continuation jobs failed'):
                    runner.run_continuation_study(source,root/'output',jobs=['dual_continue','dual_frozen_head'])
            self.assertEqual(calls,['dual_continue','dual_frozen_head'])
            result=json.loads((root/'output/completion.json').read_text())
            self.assertEqual(result['jobs']['dual_frozen_head']['status'],'complete')
            self.assertIn('dual_continue',result['failures'])
            self.assertTrue(result['source_files_unchanged'])

    def test_foreign_failure_identity_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);run=root/'run';run.mkdir()
            sources=self.sources();plan=self.plan(sources)
            unrelated={'identity':{'other':'experiment'}}
            (run/'failure.json').write_text(json.dumps(unrelated))
            self.assertFalse(runner._record_failure(plan,'dual_continue',run,{'error_type':'fixture'}))
            self.assertEqual(json.loads((run/'failure.json').read_text()),unrelated)

    def test_owned_failure_before_metadata_or_first_checkpoint_retries(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for prior in ('only','before_last'):
                summary,_,epochs,_,run,_,_=self.fake_run(root/prior,prior_failure=prior)
                self.assertEqual(summary['status'],'complete')
                self.assertEqual(epochs,[131,132,133])
                self.assertFalse((run/'failure.json').exists())


if __name__=='__main__': unittest.main()
