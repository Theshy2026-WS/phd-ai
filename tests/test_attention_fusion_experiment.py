import copy
import json
from pathlib import Path
import random
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

import attention_fusion_experiment as runner
from models.attention_correction import state_digest
from test_baseline_diagnostics import config as original_config, fixture_datasets, toy_model


class ToyFusion(nn.Module):
    def __init__(self, variant='cross16'):
        super().__init__()
        self.encoder = toy_model()
        if variant != 'control':
            self.fusion = nn.Linear(3, 3)
            self.gamma = nn.Parameter(torch.tensor(.1))
        else:
            self.fusion = None
        self.attention_flops_per_image = 64

    def forward(self, images):
        logits = self.encoder(images)
        return logits if self.fusion is None else logits+self.gamma*self.fusion(images.mean((-2,-1)))


def factory(variant, parent_state=None, seed=42):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return ToyFusion(variant)


class OptimizerTests(unittest.TestCase):
    def test_new_parameters_get10x_lr_and_ratio_survives_entire_schedule(self):
        config = runner.job_config(original_config(), 'warm_cross16', 30)
        model = ToyFusion()
        optimizer = runner.make_optimizer(model, config)
        scheduler = runner.make_scheduler(optimizer, config)
        encoder_ids = {id(p) for p in model.encoder.parameters()}
        seen = set()
        for group in optimizer.param_groups:
            for parameter in group['params']:
                self.assertNotIn(id(parameter), seen)
                seen.add(id(parameter))
                self.assertEqual(group['role'], 'encoder' if id(parameter) in encoder_ids else 'fusion')
                self.assertEqual(group['weight_decay'], config['training']['weight_decay'] if parameter.ndim > 1 else 0.)
        self.assertEqual(seen, {id(p) for p in model.parameters()})
        for step in range(31):
            rates = runner._rates(optimizer)
            self.assertAlmostEqual(rates['fusion_lr']/rates['encoder_lr'], 10.)
            if step == 0:
                self.assertAlmostEqual(rates['encoder_lr'], 5e-6)
            optimizer.step(); scheduler.step()
        self.assertAlmostEqual(optimizer.param_groups[0]['lr'], 1e-6)

    def test_fresh_schedule_equal_lr_and_config_keeps_original_aug(self):
        parent = original_config()
        before = copy.deepcopy(parent)
        config = runner.job_config(parent, 'fresh_cross16', 100)
        self.assertEqual(parent, before)
        self.assertEqual(config['augmentation'], parent['augmentation'])
        for key in ('weight_decay','label_smoothing','mixup_alpha','cutmix_alpha','mix_probability'):
            self.assertEqual(config['training'][key],parent['training'][key])
        optimizer=runner.make_optimizer(ToyFusion(),config)
        scheduler=runner.make_scheduler(optimizer,config)
        for _ in range(101):
            rates=runner._rates(optimizer)
            self.assertEqual(rates['encoder_lr'],rates['fusion_lr'])
            optimizer.step();scheduler.step()

    def test_control_has_only_encoder_groups_and_all13_jobs_are_explicit(self):
        config=runner.job_config(original_config(),'warm_control',30)
        optimizer=runner.make_optimizer(ToyFusion('control'),config)
        self.assertEqual({group['role'] for group in optimizer.param_groups},{'encoder'})
        self.assertEqual(len(runner.DEFAULT_JOBS),13)
        self.assertEqual(runner.split_job('warm_gated_cross16'),('warm','gated_cross16'))
        with self.assertRaises(ValueError): runner.split_job('warm_unknown')


class RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def plan(self):
        config=original_config()
        config['dataset'].update(batch_size=2,num_workers=0)
        config['training'].update(mixed_precision=False,gradient_clip_norm=5.)
        return dict(config=config,warm_epochs=3,fresh_epochs=4,seed=42,
            source_hashes=runner.source_hashes(),parent_run='original-cnn8',
            parent_checkpoint_sha256='parent-hash',parent_completed_epochs=100,
            split_hash='fixture-split',examples=0,estimated_epoch_seconds=.001)

    def supplied(self,root):
        supplied,manifest,fingerprint=fixture_datasets(root)
        base=supplied['train']
        from datasets.utility_tiny import UtilityDataset
        supplied['train_probe']=UtilityDataset(base.folder,[0,4,8],'train_probe',42)
        return supplied,manifest,fingerprint

    def run_fake(self,root,job='warm_cross16',interrupted=False,race=False,deadline=None,improve=True):
        root.mkdir(parents=True,exist_ok=True)
        plan=self.plan(); supplied=self.supplied(root)
        state=dict(epoch=0,interrupt=interrupted,race=race)
        data_epochs=[]
        def train(model,ema,batches,config,seed,epoch,device,amp,dtype,optimizer,scaler,**kwargs):
            added=epoch-100 if job.startswith('warm_') else epoch
            if added==2 and state['interrupt']:
                state['interrupt']=False
                raise KeyboardInterrupt()
            state['epoch']=added;data_epochs.append(epoch)
            optimizer.zero_grad()
            coefficient=random.random()+np.random.rand()
            loss=sum((p*torch.randn_like(p)*coefficient).sum() for p in model.parameters())
            loss.backward();optimizer.step();runner.update_ema(ema,model)
            return dict(train_soft_target_loss=1.,skipped_amp_steps=0)
        def evaluate(model,batches,*args):
            if batches.split=='train_probe':
                random.random();np.random.rand();torch.rand(3)
                return dict(accuracy=1.,top5_accuracy=1.,nll=.01)
            scores={0:.7,1:.8,2:.9,3:.6,4:.6} if improve else {0:.7,1:.6,2:.5,3:.4,4:.3}
            return dict(accuracy=scores[state['epoch']],top5_accuracy=1.,nll=.5)
        original_save=runner.atomic_save
        def save(path,value):
            if Path(path).name=='last.pt' and value['epoch']==2 and state['race']:
                state['race']=False
                raise KeyboardInterrupt()
            return original_save(path,value)
        with patch.object(runner,'prepare_datasets',return_value=supplied), \
             patch.object(runner,'build_attention_model',side_effect=factory), \
             patch.object(runner,'loader',side_effect=lambda ds,*args:ds), \
             patch.object(runner,'train_one_epoch',side_effect=train), \
             patch('generalization_experiment.evaluate',side_effect=evaluate), \
             patch('generalization_experiment.loader',side_effect=lambda ds,*args:ds), \
             patch('spatial_experiment.evaluate',side_effect=evaluate), \
             patch('spatial_experiment.loader',side_effect=lambda ds,*args:ds), \
             patch.object(runner,'estimate_flops',return_value=1000), \
             patch.object(runner,'atomic_save',side_effect=save):
            if interrupted or race:
                with self.assertRaises(KeyboardInterrupt): runner._train_job(plan,job,root/'run',{},deadline)
            result=runner._train_job(plan,job,root/'run',{},deadline)
        last=torch.load(root/'run/last.pt',weights_only=False)
        self.assertEqual(supplied[0]['audit'].reads,0)
        self.assertEqual(supplied[0]['official_validation'].reads,0)
        return result,last,data_epochs,root/'run',plan

    def test_warm_fresh_epoch0_best_last_and_cost_metadata(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            a,last,epochs,run,_=self.run_fake(root/'warm')
            self.assertEqual(epochs,[101,102,103])
            self.assertEqual(a['best_epoch'],2)
            self.assertEqual(a['best_trained_selection']['epoch'],2)
            self.assertEqual(a['variant'],'cross16')
            self.assertEqual(a['job'],'warm_cross16')
            self.assertFalse(a['encoder_frozen'])
            self.assertEqual(a['forward_flops_estimate'],1064)
            self.assertNotEqual(a['diagnostic_directory'],a['last_diagnostic_directory'])
            for key in ('diagnostic_directory','last_diagnostic_directory'):
                self.assertTrue((run/a[key]/'selection/predictions.npz').is_file())
            b,_,epochs,_,_=self.run_fake(root/'fresh',job='fresh_cross16')
            self.assertEqual(b['completed_epochs'],4)
            self.assertEqual(epochs,[1,2,3,4])

    def test_deterioration_keeps_epoch0_but_reports_best_trained_and_last(self):
        with tempfile.TemporaryDirectory() as temp:
            summary,_,_,run,_=self.run_fake(Path(temp),improve=False)
            self.assertEqual(summary['best_epoch'],0)
            self.assertEqual(summary['best_trained_selection']['epoch'],1)
            self.assertEqual(summary['best_trained_selection']['accuracy'],.6)
            self.assertEqual(summary['last_epoch_selection']['accuracy'],.4)
            trained=torch.load(run/summary['best_trained_checkpoint'],weights_only=False)
            self.assertEqual(trained['epoch'],1)
            self.assertEqual(trained['selection']['accuracy'],.6)

    def test_resume_and_write_before_last_race_preserve_model_ema_and_lr(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            _,full,_,_,_=self.run_fake(root/'full')
            a,resumed,_,_,_=self.run_fake(root/'resumed',interrupted=True)
            b,race,_,race_run,_=self.run_fake(root/'race',race=True)
            self.assertEqual((a['best_epoch'],b['best_epoch']),(2,2))
            for name in full['model']:
                torch.testing.assert_close(full['model'][name],resumed['model'][name],atol=0,rtol=0)
                torch.testing.assert_close(full['model'][name],race['model'][name],atol=0,rtol=0)
                torch.testing.assert_close(full['ema_model'][name],race['ema_model'][name],atol=0,rtol=0)
            self.assertEqual(full['history'][-1]['fusion_lr'],race['history'][-1]['fusion_lr'])
            learned=torch.load(race_run/'best_trained.pt',weights_only=False)
            overall=torch.load(race_run/'best.pt',weights_only=False)
            self.assertEqual(learned['epoch'],2)
            for name in learned['model']:
                torch.testing.assert_close(learned['model'][name],overall['model'][name],atol=0,rtol=0)

    def test_budget_partial_preserves_untrained_epoch0_and_horizon_validation(self):
        with tempfile.TemporaryDirectory() as temp:
            result,last,epochs,_,_=self.run_fake(Path(temp),deadline=time.monotonic()-1)
            self.assertEqual(result['status'],'budget_exhausted')
            self.assertEqual(result['completed_epochs'],0)
            self.assertIsNone(result['best_trained_selection'])
            self.assertEqual(last['epoch'],0)
            self.assertEqual(epochs,[])
        for kwargs in (dict(warm_epochs=31),dict(fresh_epochs=101),dict(jobs=[]),dict(jobs=['unknown'])):
            with self.assertRaises(ValueError): runner.run_attention_study('unused','unused',**kwargs)

    def test_public_study_continues_after_independent_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);parent=root/'parent';parent.mkdir()
            loaded=(parent,dict(seed=42,split_hash='split',completed_epochs=100),
                    dict(config=original_config(),model={}),'parent-hash')
            calls=[]
            def train(plan,job,run,state,deadline):
                calls.append(job)
                if job=='warm_control': raise RuntimeError('synthetic arm failure')
                return dict(status='complete',completed_epochs=30,encoder_initial_state_sha256='shared')
            with patch.object(runner,'load_spatial_parent',return_value=loaded), \
                 patch.object(runner,'clean_datasets',return_value=(None,None,None,'split',None)), \
                 patch.object(runner,'source_hashes',return_value={}), \
                 patch.object(runner,'execution_environment',return_value={}), \
                 patch.object(runner.torch.cuda,'is_available',return_value=True), \
                 patch.object(runner.torch.cuda,'empty_cache'), \
                 patch.object(runner,'_train_job',side_effect=train), \
                 patch.object(runner,'_make_reference'), \
                 patch.object(runner,'comparison'), \
                 patch.object(runner,'file_hash',return_value='parent-hash'):
                with self.assertRaisesRegex(RuntimeError,'attention jobs failed'):
                    runner.run_attention_study(parent,root/'study',jobs=['warm_control','warm_cross16'])
            self.assertEqual(calls,['warm_control','warm_cross16'])
            summary=json.loads((root/'study/completion.json').read_text())
            self.assertEqual(summary['status'],'failed')


if __name__=='__main__': unittest.main()
