import copy
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

import hcsi_tiny_experiment as study


class SmallJoint(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=nn.Sequential(nn.Linear(5,8),nn.BatchNorm1d(8),nn.GELU(),nn.Dropout(.2))
        self.classifier=nn.Linear(8,5)
    def forward(self,images):
        return self.classifier(self.encoder(images))


class SmallFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=nn.Linear(5,5)
        self.vit=nn.Linear(5,5)
        self.classifier=nn.Linear(10,5)
        self.interaction=nn.Linear(5,5)


class HCSITinyExperimentTests(unittest.TestCase):
    def fixture(self,epochs=2):
        config=dict(dataset=dict(batch_size=64,num_workers=0,data_dir='data',split_seed=7),
            augmentation=dict(crop_scale=[.6,1.],randaugment_ops=2,randaugment_magnitude=5),
            training=dict(epochs=100,seed=42,probe_per_class=20,mixed_precision=False,
                learning_rate=5e-4,weight_decay=.05,scheduler={},gradient_clip_norm=5.,
                mixup_alpha=.2,cutmix_alpha=1.,mix_probability=.5,label_smoothing=.1))
        provenance=dict(source_cnn_accuracy=.648,source_vit_accuracy=.5018,
            parent_checkpoint_sha256='cnn',vit_checkpoint_sha256='vit')
        reference=dict(suite='reference',checkpoint='reference/best.pt',checkpoint_sha256='ref',
            selection=dict(accuracy=.6545,nll=1.79),epoch=7,source='ema')
        plan=dict(protocol=study.PROTOCOL,config=config,config_recipe_hash=study.recipe_hash(config),
            seed=42,split_hash='split',epochs=epochs,jobs=list(study.JOBS),
            total_new_model_epochs=len(study.JOBS)*epochs,source_hashes={'new':'sha'},
            source_provenance=provenance,source_files={},model_policy=copy.deepcopy(study.MODEL_POLICY),
            optimizer_policy=copy.deepcopy(study.OPTIMIZER_POLICY),reference=reference,
            shutdown_after_completion=False,audit_partition_evaluated=False,
            official_validation_evaluated=False,author_benchmark_reproduction=False,
            fusion_augmentation_epoch_offset=100,smoke=False,examples=0)
        return plan

    def patches(self,prototype,batches):
        def fake_runtime(config,seed):
            torch.manual_seed(seed)
            return 'cpu',False,torch.float32,torch.amp.GradScaler('cpu',enabled=False)
        return (patch.object(study,'build_model',side_effect=lambda *a:copy.deepcopy(prototype)),
            patch.object(study,'runtime',side_effect=fake_runtime),
            patch.object(study,'loader',return_value=batches),
            patch.object(study,'mix_batch',side_effect=lambda x,y,*a:(x+torch.rand_like(x)*.01,
                torch.nn.functional.one_hot(y,5).float())))

    def test_five_jobs_share_views_horizon_and_learning_rate_recipe(self):
        plan=self.fixture();before=copy.deepcopy(plan)
        configs=[study.job_config(plan,job) for job in study.JOBS]
        self.assertEqual(study.JOBS,('cnn_continue','late_mlp','hcsi','cnn_mlp','late_mlp_random_vit'))
        self.assertTrue(all(c==configs[0] for c in configs))
        self.assertEqual(configs[0]['training']['encoder_learning_rate'],5e-5)
        self.assertEqual(configs[0]['training']['fusion_learning_rate'],5e-4)
        self.assertEqual(configs[0]['training']['epochs'],2)
        self.assertEqual(plan,before)
        with self.assertRaises(ValueError):study.job_config(plan,'new_seed')

    def test_optimizer_updates_both_encoders_and_assigns_head_and_gates_higher_lr(self):
        model=SmallFusion();config=study.job_config(self.fixture(),'hcsi')
        optimizer=study.make_optimizer(model,config)
        roles={id(p):(g['role'],g['lr'],g['weight_decay']) for g in optimizer.param_groups for p in g['params']}
        self.assertEqual(set(roles),{id(p) for p in model.parameters()})
        for p in model.encoder.parameters():self.assertEqual(roles[id(p)][:2],('encoder',5e-5))
        for p in model.vit.parameters():self.assertEqual(roles[id(p)][:2],('vit',5e-5))
        for module in (model.classifier,model.interaction):
            for p in module.parameters():self.assertEqual(roles[id(p)][:2],('fusion',5e-4))
        self.assertEqual(roles[id(model.encoder.bias)][2],0.)
        model.vit.weight.requires_grad_(False)
        with self.assertRaises(ValueError):study.make_optimizer(model,config)

    def test_preserved_cnn_linear_head_gets_classifier_learning_rate(self):
        model=SmallJoint();model.encoder.head=nn.Module();model.encoder.head.dense=nn.Linear(8,5)
        del model.classifier
        optimizer=study.make_optimizer(model,study.job_config(self.fixture(),'cnn_continue'))
        head_ids={id(p) for p in model.encoder.head.dense.parameters()}
        for group in optimizer.param_groups:
            for p in group['params']:
                self.assertEqual(group['role'],'fusion' if id(p) in head_ids else 'encoder')

    def test_clean_evaluation_is_fp32_complete_and_finite(self):
        model=nn.Identity();batches=[(2*torch.eye(5)[:3],torch.arange(3),['a','b','c']),
            (2*torch.eye(5)[3:],torch.arange(3,5),['d','e'])]
        with patch.object(study,'loader',return_value=batches):
            result=study.evaluate_model(model,object(),self.fixture()['config'],42,'cpu')
        self.assertEqual(result['n'],5);self.assertEqual(result['accuracy'],1.)
        self.assertEqual(result['top5_accuracy'],1.)
        with patch.object(study,'loader',return_value=batches):
            with self.assertRaises(study.BudgetExceeded):
                study.evaluate_model(model,object(),self.fixture()['config'],42,'cpu',time.time()-1)

    def test_checkpoint_weights_validate_finite_and_exact_layout(self):
        model=SmallJoint();state=study.cpu_state(model)
        state['classifier.weight'].fill_(float('nan'))
        with self.assertRaises(ValueError):study.validate_weights(model,state)
        with self.assertRaises(ValueError):study.validate_weights(model,{})

    def test_plan_records_full_training_and_known_reference_without_repeating_it(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);base=self.fixture(30)
            sources=dict(cnn_suite=str(root/'cnn'),vit_suite=str(root/'vit'),
                config=base['config'],seed=42,split_hash='split',provenance=base['source_provenance'],
                reference=dict(record=dict(base['reference'],suite=str(root/'reference'))),
                source_files={},parent_completed_epochs=213)
            args=study.arguments().parse_args(['--output',str(root/'new')])
            with patch.object(study,'load_sources',return_value=sources),\
                    patch.object(study,'_validate_data',return_value={}),\
                    patch.object(study,'source_hashes',return_value={'new':'sha'}):
                plan=study.make_plan(args)
                self.assertEqual(plan['total_new_model_epochs'],150)
                self.assertEqual(plan['reference']['selection']['accuracy'],.6545)
                self.assertEqual(plan['model_policy']['beta_pos'],.2)
                self.assertFalse(plan['shutdown_after_completion'])
                self.assertFalse(plan['author_benchmark_reproduction'])
                suite=study.create_suite(args,plan)
                self.assertEqual(study.validate_resume(suite),plan)
                changed=copy.deepcopy(plan);changed['optimizer_policy']['encoder_learning_rate']=1.
                (suite/'suite.json').write_text(json.dumps(changed))
                with self.assertRaisesRegex(ValueError,'recipe'):study.validate_resume(suite)

    def test_output_cannot_overlap_any_source(self):
        sources=dict(cnn_suite='source/cnn',vit_suite='source/vit',reference=dict(record=dict(suite='reference')))
        for target in ('source','source/cnn','source/vit/child','reference/child'):
            with self.assertRaises(ValueError):study._separate_output(target,sources)
        study._separate_output('unrelated/new',sources)

    def test_full_model_resume_matches_continuous_model_ema_optimizer_scheduler_rng(self):
        torch.manual_seed(19);prototype=SmallJoint();plan=self.fixture()
        batches=[(torch.eye(5),torch.arange(5),list('abcde'))]
        datasets=dict(train=object(),selection=object(),train_probe=object())
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);left=root/'left';right=root/'right';left.mkdir();right.mkdir()
            a,b,c,d=self.patches(prototype,batches)
            with a,b,c,d:
                study.train_job(plan,'hcsi',left,{},datasets)
                ordinary=study.train_one_epoch;calls=[0]
                def interrupted(*args,**kwargs):
                    calls[0]+=1;result=ordinary(*args,**kwargs)
                    if calls[0]==2:raise study.BudgetExceeded('after optimizer update')
                    return result
                with patch.object(study,'train_one_epoch',side_effect=interrupted):
                    partial=study.train_job(plan,'hcsi',right,{},datasets)
                self.assertEqual(partial['completed_epochs'],1)
                resumed=study.train_job(plan,'hcsi',right,{},datasets)
                self.assertEqual(resumed['status'],'complete')
            x=torch.load(left/'hcsi/last.pt',weights_only=False)
            y=torch.load(right/'hcsi/last.pt',weights_only=False)
            for key in ('model','ema'):
                for name,value in x[key].items():self.assertTrue(torch.equal(value,y[key][name]),(key,name))
            for parameter,state in x['optimizer']['state'].items():
                for name,value in state.items():
                    other=y['optimizer']['state'][parameter][name]
                    if isinstance(value,torch.Tensor):self.assertTrue(torch.equal(value,other))
                    else:self.assertEqual(value,other)
            self.assertEqual(x['scheduler'],y['scheduler'])
            self.assertTrue(torch.equal(x['rng']['torch'],y['rng']['torch']))
            self.assertGreater(resumed['best_trained_epoch'],0)
            self.assertTrue((right/'hcsi/best_trained.pt').is_file())

    def test_deadline_during_initial_evaluation_can_retry_owned_directory(self):
        torch.manual_seed(19);prototype=SmallJoint();plan=self.fixture(1)
        batches=[(torch.eye(5),torch.arange(5),list('abcde'))]
        datasets=dict(train=object(),selection=object(),train_probe=object())
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);a,b,c,d=self.patches(prototype,batches)
            with a,b,c,d:
                with patch.object(study,'evaluate_model',side_effect=study.BudgetExceeded('initial')):
                    with self.assertRaises(study.BudgetExceeded):study.train_job(plan,'hcsi',root,{},datasets)
                self.assertFalse((root/'hcsi/last.pt').exists())
                with patch.object(study,'train_one_epoch',side_effect=study.BudgetExceeded('later')):
                    result=study.train_job(plan,'hcsi',root,{},datasets)
                self.assertEqual(result['status'],'budget_exhausted')
                self.assertEqual(result['completed_epochs'],0)

    def test_committed_last_recovers_best_indexes_after_interruption(self):
        plan=self.fixture();config=study.job_config(plan,'hcsi');metadata={'identity':'same'}
        snapshot=dict(model=study.cpu_state(SmallJoint()),epoch=1,source='ema',
            selection=dict(accuracy=.6),metadata=metadata,config=config)
        with tempfile.TemporaryDirectory() as temp:
            run=Path(temp)
            study.atomic_save(run/'best.pt',dict(snapshot,epoch=0))
            best,trained=study._restore_best_snapshots(run,dict(epoch=1,best_snapshot=snapshot,
                best_trained_snapshot=snapshot),metadata,config)
            self.assertEqual(best['epoch'],1);self.assertEqual(trained['epoch'],1)
            self.assertEqual(torch.load(run/'best.pt',weights_only=False)['epoch'],1)
            with self.assertRaisesRegex(ValueError,'outside'):
                study._restore_best_snapshots(run,dict(epoch=0,best_snapshot=snapshot),metadata,config)

    def test_last_ema_loader_uses_ema_weights_and_metrics(self):
        plan=self.fixture(1);model=SmallJoint();metadata=study._metadata(plan,'hcsi',model)
        weights=study.cpu_state(model);averaged={name:value.clone() for name,value in weights.items()}
        averaged['classifier.weight'].add_(.1)
        score=dict(accuracy=.6,top5_accuracy=1.,nll=.8,n=5)
        state=dict(epoch=1,source='raw',selection=score,model=weights,ema=averaged,
            metadata=metadata,config=study.job_config(plan,'hcsi'),
            history=[{'ema_selection_'+k:v for k,v in score.items()}])
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);run=root/'hcsi';run.mkdir()
            study.save_json(run/'metadata.json',metadata);study.atomic_save(run/'last.pt',state)
            with patch.object(study,'build_model',side_effect=lambda *a:SmallJoint()):
                loaded,value=study.load_model(plan,root,'hcsi','last_ema')
            self.assertEqual(value['source'],'ema')
            self.assertTrue(torch.equal(loaded.classifier.weight,averaged['classifier.weight']))
            self.assertEqual(value['selection'],score)

    def test_source_reference_and_independent_weights_pinned_on_resume(self):
        plan=self.fixture();plan.update(cnn_suite='cnn',vit_suite='vit',reference=dict(plan['reference'],suite='joint'))
        sources=dict(source_files={},selected={},provenance={})
        reference=dict(record=plan['reference'],state={},plan={})
        with patch.object(study,'load_best_sources',return_value=sources),\
                patch.object(study,'_reference_sources',return_value=(reference,[])):
            self.assertEqual(study.load_sources(plan)['reference']['record'],plan['reference'])
            changed=copy.deepcopy(plan);changed['source_files']={'unexpected':'sha'}
            with self.assertRaisesRegex(ValueError,'Pinned'):study.load_sources(changed)

    def test_random_vit_control_changes_only_initial_vit_not_head_cnn_or_views(self):
        import hcsi_tiny_models
        plan=self.fixture()
        torch.manual_seed(31);cnn=nn.Linear(5,5);trained_vit=nn.Linear(5,5)
        with torch.no_grad():
            for p in trained_vit.parameters():p.fill_(4.)
        sources=dict(parent_model=study.cpu_state(cnn),selected=dict(profile='vit_small_p8_augreg',
            state=dict(model=study.cpu_state(trained_vit))))
        class Assembly(nn.Module):
            def __init__(self,encoder,vit,variant,**kwargs):
                super().__init__();self.encoder=encoder;self.vit=vit;self.variant=variant
                self.classifier=nn.Linear(10,5)
        def random_vit(profile,seed):
            with torch.random.fork_rng(devices=[]):
                torch.default_generator.manual_seed(seed)
                return nn.Linear(5,5)
        with patch('models.spatial_preact.build_spatial_model',side_effect=lambda *a:copy.deepcopy(cnn)),\
                patch('models.vit_augreg.build_vit',side_effect=random_vit),\
                patch.object(hcsi_tiny_models,'HCSITinyFusion',Assembly):
            learned=study.build_model(plan,'late_mlp',sources)
            random=study.build_model(plan,'late_mlp_random_vit',sources)
        self.assertEqual(learned.variant,random.variant)
        self.assertTrue(torch.equal(learned.classifier.weight,random.classifier.weight))
        self.assertTrue(torch.equal(learned.classifier.bias,random.classifier.bias))
        self.assertTrue(torch.equal(learned.encoder.weight,random.encoder.weight))
        self.assertTrue(torch.equal(learned.vit.weight,trained_vit.weight))
        self.assertFalse(torch.equal(learned.vit.weight,random.vit.weight))
        self.assertNotEqual(learned.vit_initialization_digest,random.vit_initialization_digest)
        self.assertEqual(study.job_config(plan,'late_mlp'),study.job_config(plan,'late_mlp_random_vit'))
        self.assertIn('original random ViT',study._metadata(plan,'late_mlp_random_vit',random)['initialization'])

    def test_independent_jobs_continue_after_one_failure_and_suite_is_not_complete(self):
        import sys
        plan=self.fixture(1);calls=[]
        def fake_train(plan,job,*args):
            calls.append(job)
            if job=='hcsi':raise RuntimeError('isolated candidate failure')
            return dict(status='complete',completed_epochs=1)
        diagnostic=SimpleNamespace(run_hcsi_diagnostics=lambda *a,**k:dict(status='partial'))
        with tempfile.TemporaryDirectory() as temp:
            with patch.object(study,'validate_resume',return_value=plan),\
                    patch.object(study,'load_sources',return_value={}),\
                    patch.object(study,'prepare_datasets',return_value=({}, {},'split')),\
                    patch.object(study,'train_job',side_effect=fake_train),\
                    patch.object(study,'training_report'),\
                    patch.dict(sys.modules,{'hcsi_tiny_diagnostics':diagnostic}):
                result=study.run_suite(temp)
            self.assertEqual(calls,list(study.JOBS))
            self.assertEqual(result['status'],'failed')
            self.assertEqual(result['outcomes']['late_mlp_random_vit']['status'],'complete')
            self.assertTrue((Path(temp)/'training/hcsi_failure.json').is_file())


if __name__=='__main__':
    unittest.main()
