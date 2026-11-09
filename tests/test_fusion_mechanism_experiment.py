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

import fusion_mechanism_experiment as study


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


class FixtureCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem=nn.Linear(5,2048)
        self.stages=nn.ModuleList([nn.Identity() for _ in range(4)])
        self.final_reduction=nn.Identity()
        self.head=nn.Module()
        self.head.bn=nn.BatchNorm1d(2048)
        self.head.relu=nn.ReLU()
        self.head.gap=nn.Identity()
        self.head.dense=nn.Linear(2048,5)


class FixtureViT(nn.Module):
    embed_dim=8
    num_patches=3
    attention_flops_per_image=0
    def __init__(self):
        super().__init__()
        self.projection=nn.Linear(5,8)
        self.classifier=nn.Linear(8,5)
    def forward_tokens(self,images):
        values=self.projection(images)
        return torch.stack([values,2*values,3*values,4*values],dim=1)


class FusionMechanismExperimentTests(unittest.TestCase):
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
            seed=42,split_hash='split',epochs=epochs,fusion_epochs=epochs,jobs=list(study.JOBS),
            total_new_model_epochs=len(study.JOBS)*epochs,source_hashes={'new':'sha'},
            source_provenance=provenance,source_files={},model_policy=copy.deepcopy(study.MODEL_POLICY),
            selected_vit={},
            optimizer_policy=copy.deepcopy(study.OPTIMIZER_POLICY),reference=reference,
            shutdown_after_completion=False,audit_partition_evaluated=False,
            official_validation_evaluated=False,complete_all=True,max_hours=None,
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

    def test_three_jobs_share_winning_views_horizon_and_learning_rate_recipe(self):
        plan=self.fixture();before=copy.deepcopy(plan)
        configs=[study.job_config(plan,job) for job in study.JOBS]
        self.assertEqual(study.JOBS,('random_vit_cls_linear','reset_cnn_head_cls_linear','patch_mean_linear'))
        self.assertTrue(all(c==configs[0] for c in configs))
        self.assertEqual(configs[0]['training']['encoder_learning_rate'],5e-5)
        self.assertEqual(configs[0]['training']['fusion_learning_rate'],5e-4)
        self.assertEqual(configs[0]['training']['epochs'],2)
        self.assertEqual(plan,before)
        with self.assertRaises(ValueError):study.job_config(plan,'new_seed')

    def test_optimizer_updates_both_encoders_and_assigns_head_and_gates_higher_lr(self):
        model=SmallFusion();config=study.job_config(self.fixture(),'random_vit_cls_linear')
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
                source_files={},selected={},parent_completed_epochs=213)
            args=study.arguments().parse_args(['--output',str(root/'new')])
            with patch.object(study,'load_sources',return_value=sources),\
                    patch.object(study,'_validate_data',return_value={}),\
                    patch.object(study,'source_hashes',return_value={'new':'sha'}):
                plan=study.make_plan(args)
                self.assertEqual(plan['total_new_model_epochs'],90)
                self.assertEqual(plan['reference']['selection']['accuracy'],.6545)
                self.assertEqual(plan['model_policy']['readout']['patch_mean_linear'],'normalized_patch_mean')
                self.assertFalse(plan['shutdown_after_completion'])
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
                study.train_job(plan,'random_vit_cls_linear',left,{},datasets)
                ordinary=study.train_one_epoch;calls=[0]
                def interrupted(*args,**kwargs):
                    calls[0]+=1;result=ordinary(*args,**kwargs)
                    if calls[0]==2:raise study.BudgetExceeded('after optimizer update')
                    return result
                with patch.object(study,'train_one_epoch',side_effect=interrupted):
                    partial=study.train_job(plan,'random_vit_cls_linear',right,{},datasets)
                self.assertEqual(partial['completed_epochs'],1)
                resumed=study.train_job(plan,'random_vit_cls_linear',right,{},datasets)
                self.assertEqual(resumed['status'],'complete')
            x=torch.load(left/'random_vit_cls_linear/last.pt',weights_only=False)
            y=torch.load(right/'random_vit_cls_linear/last.pt',weights_only=False)
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
            self.assertTrue((right/'random_vit_cls_linear/best_trained.pt').is_file())

    def test_deadline_during_initial_evaluation_can_retry_owned_directory(self):
        torch.manual_seed(19);prototype=SmallJoint();plan=self.fixture(1)
        batches=[(torch.eye(5),torch.arange(5),list('abcde'))]
        datasets=dict(train=object(),selection=object(),train_probe=object())
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);a,b,c,d=self.patches(prototype,batches)
            with a,b,c,d:
                with patch.object(study,'evaluate_model',side_effect=study.BudgetExceeded('initial')):
                    with self.assertRaises(study.BudgetExceeded):study.train_job(plan,'random_vit_cls_linear',root,{},datasets)
                self.assertFalse((root/'random_vit_cls_linear/last.pt').exists())
                with patch.object(study,'train_one_epoch',side_effect=study.BudgetExceeded('later')):
                    result=study.train_job(plan,'random_vit_cls_linear',root,{},datasets)
                self.assertEqual(result['status'],'budget_exhausted')
                self.assertEqual(result['completed_epochs'],0)

    def test_committed_last_recovers_best_indexes_after_interruption(self):
        plan=self.fixture();config=study.job_config(plan,'random_vit_cls_linear');metadata={'identity':'same'}
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
        plan=self.fixture(1);model=SmallJoint();metadata=study._metadata(plan,'random_vit_cls_linear',model)
        weights=study.cpu_state(model);averaged={name:value.clone() for name,value in weights.items()}
        averaged['classifier.weight'].add_(.1)
        score=dict(accuracy=.6,top5_accuracy=1.,nll=.8,n=5)
        state=dict(epoch=1,source='raw',selection=score,model=weights,ema=averaged,
            metadata=metadata,config=study.job_config(plan,'random_vit_cls_linear'),
            history=[{'ema_selection_'+k:v for k,v in score.items()}])
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);run=root/'random_vit_cls_linear';run.mkdir()
            study.save_json(run/'metadata.json',metadata);study.atomic_save(run/'last.pt',state)
            with patch.object(study,'build_model',side_effect=lambda *a:copy.deepcopy(model)):
                loaded,value=study.load_model(plan,root,'random_vit_cls_linear','last_ema')
            self.assertEqual(value['source'],'ema')
            self.assertTrue(torch.equal(loaded.classifier.weight,averaged['classifier.weight']))
            self.assertEqual(value['selection'],score)

    def test_source_reference_and_independent_weights_pinned_on_resume(self):
        plan=self.fixture();plan.update(cnn_suite='cnn',vit_suite='vit',reference=dict(plan['reference'],suite='joint'))
        sources=dict(source_files={},selected={},provenance=plan['source_provenance'],seed=42,split_hash='split')
        reference=dict(record=plan['reference'],state={},plan={})
        with patch.object(study,'load_best_sources',return_value=sources),\
                patch.object(study,'_reference_sources',return_value=(reference,[])):
            self.assertEqual(study.load_sources(plan)['reference']['record'],plan['reference'])
            changed=copy.deepcopy(plan);changed['source_files']={'unexpected':'sha'}
            with self.assertRaisesRegex(ValueError,'Pinned'):study.load_sources(changed)

    def test_independent_jobs_continue_after_one_failure_and_suite_is_not_complete(self):
        import sys
        plan=self.fixture(1);calls=[]
        def fake_train(plan,job,*args):
            calls.append(job)
            if job=='random_vit_cls_linear':raise RuntimeError('isolated candidate failure')
            return dict(status='complete',completed_epochs=1)
        diagnostic=SimpleNamespace(run_fusion_mechanism_diagnostics=lambda *a,**k:dict(status='partial'))
        with tempfile.TemporaryDirectory() as temp:
            with patch.object(study,'validate_resume',return_value=plan),\
                    patch.object(study,'load_sources',return_value={}),\
                    patch.object(study,'prepare_datasets',return_value=({}, {},'split')),\
                    patch.object(study,'train_job',side_effect=fake_train),\
                    patch.object(study,'training_report'),\
                    patch.dict(sys.modules,{'fusion_mechanism_diagnostics':diagnostic}):
                result=study.run_suite(temp)
            self.assertEqual(calls,list(study.JOBS))
            self.assertEqual(result['status'],'failed')
            self.assertEqual(result['outcomes']['patch_mean_linear']['status'],'complete')
            self.assertTrue((Path(temp)/'training/random_vit_cls_linear_failure.json').is_file())

    def test_exact_winner_construction_changes_only_each_declared_factor(self):
        from vit_augreg_experiment import build_job_model
        plan=self.fixture()
        def cnn_factory(*args):
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(42)
                return FixtureCNN()
        def vit_factory(profile,seed):
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                return FixtureViT()
        cnn=cnn_factory();vit=vit_factory('fixture',42)
        with torch.no_grad():
            cnn.stem.weight.add_(.7)
            vit.projection.weight.add_(.9)
        sources=dict(parent_model=study.cpu_state(cnn),selected=dict(profile='fixture',
            state=dict(model=study.cpu_state(vit))))
        with patch('vit_augreg_experiment.build_spatial_model',side_effect=cnn_factory),\
                patch('models.vit_augreg.build_vit',side_effect=vit_factory):
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(42)
                canonical=build_job_model('new_joint_concat',sources,sources['selected'],42)
                expected_rng=torch.get_rng_state().clone()
            sources['reference']=dict(record=dict(initial_state_sha256=study.state_digest(canonical)))
            before=torch.get_rng_state().clone()
            random,reset,mean=[study.build_model(plan,job,sources) for job in study.JOBS]
            self.assertTrue(torch.equal(before,torch.get_rng_state()))
        canonical_state=canonical.state_dict()
        for model in (random,reset,mean):
            self.assertEqual(set(model.state_dict()),set(canonical_state))
            self.assertTrue(all(p.requires_grad for p in model.parameters()))
            self.assertTrue(torch.equal(model.canonical_training_rng_cpu,expected_rng))
            self.assertTrue(torch.equal(model.fusion_classifier.weight[:,2048:],canonical.fusion_classifier.weight[:,2048:]))
        for name,value in random.state_dict().items():
            if not name.startswith('vit.'):
                self.assertTrue(torch.equal(value,canonical_state[name]),name)
        self.assertFalse(torch.equal(random.vit.projection.weight,canonical.vit.projection.weight))
        for name,value in reset.state_dict().items():
            if name not in ('fusion_classifier.weight','fusion_classifier.bias'):
                self.assertTrue(torch.equal(value,canonical_state[name]),name)
        self.assertFalse(torch.equal(reset.fusion_classifier.weight[:,:2048],canonical.fusion_classifier.weight[:,:2048]))
        self.assertFalse(torch.equal(reset.fusion_classifier.bias,canonical.fusion_classifier.bias))
        for name,value in mean.state_dict().items():
            self.assertTrue(torch.equal(value,canonical_state[name]),name)
        images=torch.randn(4,5)
        mean.eval();canonical.eval()
        cnn_features,readout,patches=mean.extract_features(images)
        self.assertTrue(torch.equal(readout,patches.mean(1)))
        self.assertFalse(torch.equal(readout,canonical.extract_features(images)[1]))
        with torch.no_grad():
            manual=mean.fusion_classifier(torch.cat((cnn_features,patches.mean(1)),dim=-1))
            self.assertTrue(torch.equal(manual,mean(images)))

    def test_initialization_guard_rejects_changed_winning_initial_model(self):
        sources=dict(reference=dict(record=dict(initial_state_sha256='wrong')),selected={})
        with patch('vit_augreg_experiment.build_job_model',return_value=SmallJoint()):
            with self.assertRaisesRegex(ValueError,'winning reference'):
                study.build_model(self.fixture(),study.JOBS[0],sources)

    def test_reference_loader_keeps_original_dependency_scope(self):
        plan=self.fixture();sources=dict(reference=dict(record=plan['reference']))
        canonical={'protocol':'old'}
        with patch('best_pair_joint.validate_resume',return_value=canonical),\
                patch('best_pair_joint.load_best_pair_model',return_value=('model','state')) as load:
            self.assertEqual(study.load_reference_model(plan,sources),('model','state'))
            self.assertEqual(load.call_args.args[0],canonical)
            self.assertNotIn('sources',load.call_args.kwargs)

    def test_completed_train_reuse_runs_no_new_optimizer_steps(self):
        torch.manual_seed(19);prototype=SmallJoint();plan=self.fixture(1)
        batches=[(torch.eye(5),torch.arange(5),list('abcde'))]
        datasets=dict(train=object(),selection=object(),train_probe=object())
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);a,b,c,d=self.patches(prototype,batches)
            with a,b,c,d:
                result=study.train_job(plan,study.JOBS[0],root,{},datasets)
                before=(root/study.JOBS[0]/'last.pt').read_bytes()
                with patch.object(study,'train_one_epoch',side_effect=AssertionError('retrained')):
                    reused=study.train_job(plan,study.JOBS[0],root,{},datasets)
                self.assertEqual(reused,result)
                self.assertEqual(before,(root/study.JOBS[0]/'last.pt').read_bytes())

    def test_foreign_unstarted_files_are_rejected(self):
        prototype=SmallJoint();plan=self.fixture(1)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);run=root/study.JOBS[0];run.mkdir()
            study.save_json(run/'metadata.json',study._metadata(plan,study.JOBS[0],prototype))
            (run/'foreign.txt').write_text('foreign')
            a,b,c,d=self.patches(prototype,[])
            with a,b,c,d:
                with self.assertRaisesRegex(ValueError,'unrelated files'):
                    study.train_job(plan,study.JOBS[0],root,{},dict(selection=object()))

    def test_training_root_manifest_matches_suite_and_rejects_foreign_plan(self):
        import sys
        plan=self.fixture(1)
        def fake_train(plan,job,*args):
            marker=Path(args[0])/'suite.json'
            self.assertEqual(json.loads(marker.read_text()),plan)
            return dict(status='complete',completed_epochs=1)
        diagnostic=SimpleNamespace(run_fusion_mechanism_diagnostics=lambda *a,**k:dict(status='complete'))
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            with patch.object(study,'validate_resume',return_value=plan),\
                    patch.object(study,'load_sources',return_value={}),\
                    patch.object(study,'prepare_datasets',return_value=({}, {},'split')),\
                    patch.object(study,'train_job',side_effect=fake_train),\
                    patch.object(study,'training_report'),\
                    patch.dict(sys.modules,{'fusion_mechanism_diagnostics':diagnostic}):
                result=study.run_suite(root)
                self.assertEqual(result['status'],'complete')
                changed=dict(plan,seed=43)
                study.save_json(root/'training/suite.json',changed)
                with self.assertRaisesRegex(ValueError,'manifest differs'):
                    study.run_suite(root)

    def test_successful_retry_preserves_old_failure_evidence_in_archive(self):
        import sys
        plan=self.fixture(1)
        diagnostic=SimpleNamespace(run_fusion_mechanism_diagnostics=lambda *a,**k:dict(status='complete'))
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);training=root/'training';training.mkdir()
            study.save_json(training/'suite.json',plan)
            study.save_json(training/(study.JOBS[0]+'_failure.json'),dict(protocol=study.PROTOCOL,job=study.JOBS[0],error='previous'))
            study.save_json(root/'failure.json',dict(protocol=study.PROTOCOL,error='old analysis'))
            with patch.object(study,'validate_resume',return_value=plan),\
                    patch.object(study,'load_sources',return_value={}),\
                    patch.object(study,'prepare_datasets',return_value=({}, {},'split')),\
                    patch.object(study,'train_job',return_value=dict(status='complete',completed_epochs=1)),\
                    patch.object(study,'training_report'),\
                    patch.dict(sys.modules,{'fusion_mechanism_diagnostics':diagnostic}):
                self.assertEqual(study.run_suite(root)['status'],'complete')
            self.assertFalse((root/'failure.json').exists())
            self.assertFalse((training/(study.JOBS[0]+'_failure.json')).exists())
            self.assertEqual(len(list((root/'recovered_failures').glob('*.json'))),1)
            self.assertEqual(len(list((training/'recovered_failures').glob('*.json'))),1)


if __name__=='__main__':
    unittest.main()
