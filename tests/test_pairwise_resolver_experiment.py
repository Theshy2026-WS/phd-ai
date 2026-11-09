import copy
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import torch
from torch import nn

import pairwise_resolver_experiment as study


class FakeResolver(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor=nn.Linear(5,5).requires_grad_(False)
        self.gain=nn.Parameter(torch.tensor(.01))
        self.register_buffer('mean',torch.zeros(3,2))
    @property
    def eta(self):return self.gain.abs()+.1
    def set_train_patch_mean(self,value):self.mean.copy_(value)
    def train(self,mode=True):
        super().train(mode);self.anchor.eval();return self
    def forward_cached(self,features,patch_tokens,anchor_logits=None):
        return anchor_logits+self.gain*features
    def forward(self,images,return_details=False):
        logits=self.gain*images
        return dict(logits=logits,delta=logits,retrieval_maps={}) if return_details else logits


class Sized:
    def __init__(self,n):self.n=n
    def __len__(self):return self.n


class PairwiseExperimentTests(unittest.TestCase):
    def plan_fixture(self):
        config=dict(dataset=dict(batch_size=64,num_workers=0,data_dir='data'),
            augmentation=dict(crop_scale=[.6,1.],randaugment_ops=2,randaugment_magnitude=5),
            training=dict(epochs=100,learning_rate=5e-4,weight_decay=.05,mixed_precision=False,
                seed=42,probe_per_class=20,scheduler={},mixup_alpha=.2,cutmix_alpha=1.,
                mix_probability=.5,label_smoothing=.1))
        return dict(protocol=study.PROTOCOL,config=config,seed=42,split_hash='split',
            epochs=1,jobs=list(study.JOBS),source_hashes={'new':'sha'},
            anchor=dict(checkpoint_sha256='anchor',suite='source',best_selection=dict(accuracy=.6525,nll=1.)),
            resolver=dict(topk=5,width=128,heads=4,intermediate_block=6,initial_eta=.1),
            optimizer=dict(learning_rate=5e-4),source_files={})

    def test_deadline_never_marks_incomplete_work_complete(self):
        with self.assertRaises(study.BudgetExceeded):study.check_deadline(time.time()-1)
        study.check_deadline(None)
        study.check_deadline(time.time()+60)

    def test_auto_anchor_selection_ignores_incomplete_and_compares_complete_scores(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);vit=root/'vit';cnn=root/'cnn'
            previous=dict(checkpoint=str(vit/'best.pt'),checkpoint_sha256='original',
                state=dict(epoch=2,source='ema',selection=dict(accuracy=.6525,nll=1.)),summary={})
            sources=dict(cnn_suite=str(cnn),vit_suite=str(vit),previous_joint=previous,
                selected=dict(profile='vit_small_p8_augreg'),source_files={})
            first=root/'runs/best_pair_joint_tiny/suite_complete';first.mkdir(parents=True)
            (first/'completion.json').write_text(json.dumps(dict(status='complete')))
            incomplete=root/'runs/best_pair_joint_tiny/suite_incomplete';incomplete.mkdir()
            (incomplete/'completion.json').write_text(json.dumps(dict(status='failed')))
            record=dict(suite=str(first),best_selection=dict(accuracy=.66,nll=.9))
            with patch.object(study,'ROOT',root),patch.object(study,'load_best_sources',return_value=sources),\
                    patch.object(study,'_complete_best_pair',return_value=(record,{'state':'new'},[])) as load:
                chosen=study.load_anchor_sources(cnn,vit)
                self.assertEqual(chosen['anchor'],record)
                self.assertEqual(len(chosen['anchor_candidates']),2)
                self.assertEqual(len(chosen['ignored_anchor_suites']),1)
                load.assert_called_once_with(first,sources)
            record['best_selection']['accuracy']=.64
            with patch.object(study,'ROOT',root),patch.object(study,'load_best_sources',return_value=sources),\
                    patch.object(study,'_complete_best_pair',return_value=(record,{'state':'new'},[])):
                chosen=study.load_anchor_sources(cnn,vit)
                self.assertEqual(chosen['anchor']['checkpoint_sha256'],'original')
                explicit=study.load_anchor_sources(cnn,vit,first)
                self.assertEqual(explicit['anchor'],record)

    def test_plan_pins_four_arms_frozen_anchor_and_no_validation_or_shutdown(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);config=self.plan_fixture()['config']
            sources=dict(cnn_suite=str(root/'cnn'),vit_suite=str(root/'vit'),
                anchor=dict(suite=str(root/'vit'),checkpoint=str(root/'vit/best.pt'),
                    checkpoint_sha256='anchor',best_selection=dict(accuracy=.6525,nll=1.)),
                anchor_candidates=[],ignored_anchor_suites=[],source_files={},
                seed=42,split_hash='split',config=config)
            args=study.arguments().parse_args(['--output',str(root/'new')])
            with patch.object(study,'load_anchor_sources',return_value=sources),\
                    patch.object(study,'_validate_data',return_value={}),\
                    patch.object(study,'source_hashes',return_value={'new':'hash'}):
                plan=study.make_plan(args)
                self.assertEqual(plan['jobs'],list(study.JOBS));self.assertEqual(plan['epochs'],30)
                self.assertEqual(plan['total_new_model_epochs'],120)
                self.assertEqual(plan['resolver'],study.RESOLVER_POLICY)
                self.assertFalse(plan['shutdown_after_completion']);self.assertFalse(plan['audit_partition_evaluated'])
                self.assertFalse(plan['official_validation_evaluated'])
                suite=study.create_suite(args,plan)
                with patch.object(study,'_load_pinned_sources',return_value=sources):
                    self.assertEqual(study.validate_resume(suite),plan)
                    changed=copy.deepcopy(plan);changed['optimizer']['learning_rate']=1.
                    (suite/'suite.json').write_text(json.dumps(changed))
                    with self.assertRaisesRegex(ValueError,'recipe'):study.validate_resume(suite)

    def test_data_validation_requires_exact_80k_10k_and_train_only_4k_probe(self):
        config=self.plan_fixture()['config']
        sources=dict(seed=42,split_hash='split',provenance=dict(source_execution_environment={}))
        datasets=dict(train=Sized(80000),selection=Sized(10000),train_probe=Sized(4000))
        manifest=dict(train=['a'],selection=['b'])
        with patch.object(study,'prepare_datasets',return_value=(datasets,manifest,'split')),\
                patch.object(study,'execution_environment',return_value={}):
            self.assertEqual(study._validate_data(config,sources),{})
            manifest['selection']=['a']
            with self.assertRaisesRegex(ValueError,'disjoint'):study._validate_data(config,sources)

    def test_lightweight_checkpoint_never_contains_or_changes_anchor(self):
        model=FakeResolver();before=copy.deepcopy(model.anchor.state_dict())
        state=study.resolver_state(model)
        self.assertFalse(any(name.startswith('anchor.') for name in state))
        with torch.no_grad():model.gain.fill_(4.)
        study.load_resolver_state(model,state)
        self.assertAlmostEqual(float(model.gain.detach()),.01,places=6)
        for key,value in before.items():self.assertTrue(torch.equal(value,model.anchor.state_dict()[key]))
        with self.assertRaises(ValueError):study.load_resolver_state(model,dict(state,bad=torch.tensor(1)))
        state['gain'].fill_(float('nan'))
        with self.assertRaises(ValueError):study.load_resolver_state(model,state)

    def test_clean_evaluation_uses_cached_logits_and_returns_full_denominator(self):
        model=FakeResolver();cache=dict(features=torch.eye(5),patch_tokens=torch.zeros(5,3,2),
            anchor_logits=2*torch.eye(5),labels=torch.arange(5))
        result=study.evaluate_cached(model,cache,'cpu',batch_size=2)
        self.assertEqual(result['n'],5);self.assertEqual(result['accuracy'],1.)
        self.assertEqual(result['top5_accuracy'],1.)
        self.assertGreater(result['nll'],0.)

    def test_four_jobs_have_identical_training_schedule_and_no_backbone_lr(self):
        plan=self.plan_fixture();before=copy.deepcopy(plan)
        configs=[study.job_config(plan,job) for job in study.JOBS]
        self.assertTrue(all(value==configs[0] for value in configs))
        self.assertEqual(configs[0]['training']['learning_rate'],5e-4)
        self.assertEqual(configs[0]['training']['epochs'],1)
        self.assertEqual(plan,before)
        with self.assertRaises(ValueError):study.job_config(plan,'seed_sweep')

    def test_optimizer_contains_only_new_parameters(self):
        model=FakeResolver();optimizer=study.make_optimizer(model,study.job_config(self.plan_fixture(),'pairwise'))
        ids={id(p) for group in optimizer.param_groups for p in group['params']}
        self.assertEqual(ids,{id(model.gain)})

    def test_checkpoint_identity_and_recipe_validation(self):
        metadata={'identity':'one'};config={'training':{'epochs':30}}
        state=dict(metadata=metadata,config=config,epoch=30)
        study._validate_checkpoint(state,metadata,config)
        with self.assertRaises(ValueError):study._validate_checkpoint(dict(state,epoch=31),metadata,config)
        with self.assertRaises(ValueError):study._validate_checkpoint(state,{'identity':'two'},config)

    def test_one_epoch_checkpoint_roundtrip_and_completed_resume_is_idempotent(self):
        torch.manual_seed(42)
        prototype=FakeResolver();plan=self.plan_fixture()
        cache=dict(features=torch.eye(5),patch_tokens=torch.zeros(5,3,2),
            anchor_logits=2*torch.eye(5),labels=torch.arange(5))
        caches=dict(selection=cache,train_probe=cache)
        batches=[(torch.eye(5),torch.arange(5),[str(i) for i in range(5)])]
        fake_scaler=torch.amp.GradScaler('cpu',enabled=False)
        with tempfile.TemporaryDirectory() as temp:
            training=Path(temp)
            with patch.object(study,'build_resolver',side_effect=lambda *a:copy.deepcopy(prototype)),\
                    patch.object(study,'runtime',return_value=('cpu',False,torch.float32,fake_scaler)),\
                    patch.object(study,'prepare_datasets',return_value=({'train':object()}, {},'split')),\
                    patch.object(study,'loader',return_value=batches),\
                    patch.object(study,'mix_batch',side_effect=lambda x,y,*a:(x,F_onehot(y))):
                result=study.train_job(plan,'pairwise',training,{},caches)
                last=torch.load(training/'pairwise/last.pt',weights_only=False)
                self.assertEqual(result['status'],'complete');self.assertEqual(result['completed_epochs'],1)
                self.assertTrue(result['frozen_anchor_unchanged'])
                self.assertEqual(last['epoch'],1)
                self.assertEqual([row['epoch'] for row in last['history']],[0,1])
                self.assertIn('mean_trainable_gradient_l2',last['history'][1])
                self.assertIn('eta',last['history'][1])
                with patch.object(study,'train_one_epoch',side_effect=AssertionError('completed epoch repeated')):
                    again=study.train_job(plan,'pairwise',training,{},caches)
                self.assertEqual(again,result)

    def test_budget_mid_epoch_preserves_completed_epoch_zero_for_resume(self):
        torch.manual_seed(42);prototype=FakeResolver();plan=self.plan_fixture()
        cache=dict(features=torch.eye(5),patch_tokens=torch.zeros(5,3,2),
            anchor_logits=2*torch.eye(5),labels=torch.arange(5))
        fake_scaler=torch.amp.GradScaler('cpu',enabled=False)
        with tempfile.TemporaryDirectory() as temp:
            with patch.object(study,'build_resolver',side_effect=lambda *a:copy.deepcopy(prototype)),\
                    patch.object(study,'runtime',return_value=('cpu',False,torch.float32,fake_scaler)),\
                    patch.object(study,'prepare_datasets',return_value=({'train':object()}, {},'split')),\
                    patch.object(study,'loader',return_value=[]),\
                    patch.object(study,'train_one_epoch',side_effect=study.BudgetExceeded('stop')):
                result=study.train_job(plan,'pairwise',Path(temp),{},dict(selection=cache,train_probe=cache))
            last=torch.load(Path(temp)/'pairwise/last.pt',weights_only=False)
            self.assertEqual(result['status'],'budget_exhausted')
            self.assertEqual(result['completed_epochs'],0);self.assertEqual(last['epoch'],0)
            self.assertTrue(result['interrupted_epoch_replays_on_resume'])

    def test_cache_identity_forbids_other_anchor_or_reserved_split(self):
        plan=self.plan_fixture();cache=dict(identity=dict(study._cache_identity(plan),split='audit'),labels=torch.zeros(1))
        with patch.object(study.torch,'load',return_value=cache):
            with self.assertRaisesRegex(ValueError,'dimensions'):study.load_feature_cache(plan,'not-opened')
        cache['identity']['anchor_sha256']='different'
        with patch.object(study.torch,'load',return_value=cache):
            with self.assertRaisesRegex(ValueError,'identity'):study.load_feature_cache(plan,'not-opened')

    def test_interrupted_epoch_resume_matches_uninterrupted_weights_ema_optimizer_rng(self):
        torch.manual_seed(42);prototype=FakeResolver();plan=self.plan_fixture();plan['epochs']=2
        cache=dict(features=torch.eye(5),patch_tokens=torch.zeros(5,3,2),
            anchor_logits=2*torch.eye(5),labels=torch.arange(5))
        caches=dict(selection=cache,train_probe=cache)
        batches=[(torch.eye(5),torch.arange(5),[str(i) for i in range(5)])]
        def fake_runtime(config,seed):
            torch.manual_seed(seed)
            return 'cpu',False,torch.float32,torch.amp.GradScaler('cpu',enabled=False)
        def random_mix(x,y,*args):
            return x+torch.rand_like(x)*.01,F_onehot(y)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);a=root/'a';b=root/'b';a.mkdir();b.mkdir()
            with patch.object(study,'build_resolver',side_effect=lambda *args:copy.deepcopy(prototype)),\
                    patch.object(study,'runtime',side_effect=fake_runtime),\
                    patch.object(study,'prepare_datasets',return_value=({'train':object()}, {},'split')),\
                    patch.object(study,'loader',return_value=batches),\
                    patch.object(study,'mix_batch',side_effect=random_mix):
                study.train_job(plan,'pairwise',a,{},caches)
                ordinary=study.train_one_epoch;calls=[0]
                def interrupt_after_update(*args,**kwargs):
                    calls[0]+=1;result=ordinary(*args,**kwargs)
                    if calls[0]==2:raise study.BudgetExceeded('after optimizer update')
                    return result
                with patch.object(study,'train_one_epoch',side_effect=interrupt_after_update):
                    partial=study.train_job(plan,'pairwise',b,{},caches)
                self.assertEqual(partial['completed_epochs'],1)
                resumed=study.train_job(plan,'pairwise',b,{},caches)
                self.assertEqual(resumed['status'],'complete')
            left=torch.load(a/'pairwise/last.pt',weights_only=False)
            right=torch.load(b/'pairwise/last.pt',weights_only=False)
            for key in ('resolver_state','ema_resolver_state'):
                for name,value in left[key].items():self.assertTrue(torch.equal(value,right[key][name]))
            for parameter,state in left['optimizer']['state'].items():
                for name,value in state.items():
                    other=right['optimizer']['state'][parameter][name]
                    if isinstance(value,torch.Tensor):self.assertTrue(torch.equal(value,other))
                    else:self.assertEqual(value,other)
            self.assertTrue(torch.equal(left['rng']['torch'],right['rng']['torch']))
            self.assertEqual(left['scheduler'],right['scheduler'])

    def test_best_index_recovers_from_transaction_committed_last_checkpoint(self):
        plan=self.plan_fixture();config=study.job_config(plan,'pairwise');metadata={'identity':'x'}
        model=FakeResolver();snapshot=dict(resolver_state=study.resolver_state(model),epoch=1,
            source='raw',selection=dict(accuracy=.6),metadata=metadata,config=config)
        with tempfile.TemporaryDirectory() as temp:
            run=Path(temp)
            # Simulate interruption after last.pt commit and before best.pt replacement.
            study.atomic_save(run/'best.pt',dict(snapshot,epoch=0))
            committed=dict(epoch=1,best_snapshot=snapshot,best_trained_snapshot=snapshot)
            best,trained=study._restore_best_snapshots(run,committed,metadata,config)
            self.assertEqual(best['epoch'],1);self.assertEqual(trained['epoch'],1)
            self.assertEqual(torch.load(run/'best.pt',weights_only=False)['epoch'],1)
            self.assertTrue((run/'best_trained.pt').is_file())
            with self.assertRaisesRegex(ValueError,'outside'):
                study._restore_best_snapshots(run,dict(committed,epoch=0),metadata,config)

    def test_deadline_during_initial_evaluation_allows_owned_directory_retry(self):
        torch.manual_seed(42);prototype=FakeResolver();plan=self.plan_fixture()
        cache=dict(features=torch.eye(5),patch_tokens=torch.zeros(5,3,2),
            anchor_logits=2*torch.eye(5),labels=torch.arange(5))
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            with patch.object(study,'build_resolver',side_effect=lambda *a:copy.deepcopy(prototype)),\
                    patch.object(study,'runtime',side_effect=lambda *a:('cpu',False,torch.float32,torch.amp.GradScaler('cpu',enabled=False))),\
                    patch.object(study,'prepare_datasets',return_value=({'train':object()}, {},'split')),\
                    patch.object(study,'loader',return_value=[]):
                with patch.object(study,'evaluate_cached',side_effect=study.BudgetExceeded('initial')):
                    with self.assertRaises(study.BudgetExceeded):
                        study.train_job(plan,'pairwise',root,{},dict(selection=cache,train_probe=cache))
                self.assertTrue((root/'pairwise/metadata.json').is_file())
                self.assertFalse((root/'pairwise/last.pt').exists())
                with patch.object(study,'train_one_epoch',side_effect=study.BudgetExceeded('later')):
                    result=study.train_job(plan,'pairwise',root,{},dict(selection=cache,train_probe=cache))
                self.assertEqual(result['completed_epochs'],0)


def F_onehot(labels):return torch.nn.functional.one_hot(labels,5).float()


if __name__=='__main__':unittest.main()
