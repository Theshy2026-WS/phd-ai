import argparse
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock,patch

import run_fusion_overnight as runner


class FakeChild:
    def __init__(self,child,code=0,status='complete',events=None,after_wait=None):
        self.child,self.code,self.status=child,code,status
        self.stdout=iter(['Experiment core done\n'])
        self.done=False;self.events=events;self.after_wait=after_wait
    def __enter__(self):return self
    def __exit__(self,*args):return False
    def poll(self):return self.code if self.done else None
    def wait(self,timeout=None):
        self.done=True
        (Path(self.child['suite'])/'completion.json').write_text(json.dumps(dict(status=self.status)))
        if self.events is not None:self.events.append(self.child['name']+'_done')
        if self.after_wait:self.after_wait()
        return self.code
    def terminate(self):self.done=True
    def kill(self):self.done=True


class QueueTests(unittest.TestCase):
    def fixture(self,root):
        queue=root/'queue';queue.mkdir()
        children=[]
        for name in runner.ORDER:
            suite=queue/name/'suite_test';suite.mkdir(parents=True)
            children.append(dict(name=name,suite=str(suite),core=runner.CORES[name]))
        return queue,dict(protocol=runner.PROTOCOL,epochs=30,children=children)

    def test_default_is_uncapped_and_child_command_never_invokes_mail_wrappers(self):
        args=runner.arguments().parse_args([])
        self.assertIsNone(args.max_hours)
        budget=runner.queue_budget(None,None,100)
        self.assertIsNone(budget['child_deadline_monotonic'])
        self.assertIsNone(budget['total_seconds'])
        for name in runner.ORDER:
            command=runner.child_command(dict(name=name,suite='suite',core=runner.CORES[name]))
            self.assertEqual(Path(command[2]).name,runner.CORES[name])
            self.assertNotIn('--deadline-unix',command)
            self.assertNotIn('--to',command)
        command=runner.child_command(dict(name='hcsi',suite='suite',core=runner.CORES['hcsi']),1234)
        self.assertIn('--deadline-unix',command)
        with self.assertRaises(ValueError):runner.child_command(dict(name='hcsi',suite='suite',core='run_hcsi_tiny.py'))

    def test_uncapped_accounting_finalizes_before_stop_and_excludes_clean_idle(self):
        with tempfile.TemporaryDirectory() as temp:
            queue=Path(temp)
            budget=runner.queue_budget(None,None,time.monotonic()-5)
            runner.record_budget(queue,budget,invocation_start_unix=time.time()-5,finished=True)
            record=json.loads((queue/'wall_budget.json').read_text())
            self.assertIsNone(record['cap_seconds']);self.assertTrue(record['invocation_finished'])
            self.assertEqual(record['instance_policy'],'email then managed stop')
            resumed=runner.queue_budget(queue,None,time.monotonic(),now_unix=time.time()+36000)
            self.assertAlmostEqual(resumed['prior_spent_seconds'],record['elapsed_active_seconds'])
            self.assertIsNone(resumed['child_deadline_monotonic'])

    def test_optional_cap_and_unfinished_resume_do_not_reset_elapsed_time(self):
        with tempfile.TemporaryDirectory() as temp:
            queue=Path(temp)
            (queue/'wall_budget.json').write_text(json.dumps(dict(cap_seconds=None,
                elapsed_active_seconds=30,invocation_finished=False,invocation_start_unix=100)))
            resumed=runner.queue_budget(queue,10,1000,now_unix=200)
            self.assertEqual(resumed['prior_spent_seconds'],130)
            self.assertEqual(resumed['remaining_seconds'],36000-130)
            (queue/'wall_budget.json').write_text(json.dumps(dict(cap_seconds=7200,
                elapsed_active_seconds=3600,invocation_finished=True)))
            self.assertEqual(runner.queue_budget(queue,10,1000)['total_seconds'],7200)

    def test_success_runs_pairwise_then_hcsi_without_watchdog_or_intermediate_delivery(self):
        with tempfile.TemporaryDirectory() as temp:
            queue,record=self.fixture(Path(temp));events=[]
            def spawn(command,**kwargs):
                name=next(n for n in runner.ORDER if Path(command[2]).name==runner.CORES[n])
                child=next(c for c in record['children'] if c['name']==name)
                events.append(name+'_start')
                self.assertNotIn('--deadline-unix',command)
                return FakeChild(child,events=events)
            with patch.object(runner.subprocess,'Popen',side_effect=spawn),\
                    patch.object(runner,'ChildWatchdog') as watchdog:
                result=runner.run_children(queue,record,io.StringIO(),runner.queue_budget(None,None,0),lambda:events.append('armed'))
            self.assertEqual(result[0],0);self.assertFalse(result[2]);watchdog.assert_not_called()
            self.assertEqual(events,['pairwise_start','armed','pairwise_done','hcsi_start','armed','hcsi_done'])

    def test_first_failure_does_not_discard_second_study(self):
        with tempfile.TemporaryDirectory() as temp:
            queue,record=self.fixture(Path(temp));children=record['children']
            with patch.object(runner.subprocess,'Popen',side_effect=[
                    FakeChild(children[0],7,'failed'),FakeChild(children[1])]):
                code,outcomes,limited=runner.run_children(queue,record,io.StringIO(),runner.queue_budget(None,None,0),lambda:None)
            self.assertEqual(code,1);self.assertFalse(limited)
            self.assertEqual(outcomes['pairwise']['status'],'failed')
            self.assertEqual(outcomes['hcsi']['status'],'complete')

    def test_explicit_time_cap_does_not_start_second_child_after_budget_is_spent(self):
        with tempfile.TemporaryDirectory() as temp:
            queue,record=self.fixture(Path(temp));budget=runner.budget_deadlines(time.monotonic(),10)
            def expire():budget['child_deadline_monotonic']=-1
            with patch.object(runner.subprocess,'Popen',return_value=FakeChild(record['children'][0],after_wait=expire)) as spawn:
                code,outcomes,limited=runner.run_children(queue,record,io.StringIO(),budget,lambda:None)
            spawn.assert_called_once();self.assertEqual(code,124);self.assertTrue(limited)
            self.assertEqual(outcomes['hcsi']['status'],'not_started')

    def test_completed_children_are_reused_without_retraining(self):
        with tempfile.TemporaryDirectory() as temp:
            queue,record=self.fixture(Path(temp))
            for child in record['children']:
                (Path(child['suite'])/'completion.json').write_text(json.dumps(dict(status='complete')))
            with patch.object(runner.subprocess,'Popen') as spawn:
                code,outcomes,_=runner.run_children(queue,record,io.StringIO(),runner.queue_budget(None,None,0),lambda:None)
            spawn.assert_not_called();self.assertEqual(code,0)
            self.assertTrue(all(v['reused_completed'] for v in outcomes.values()))

    def test_main_delivers_once_and_budget_is_finished_before_stop_after_both(self):
        with tempfile.TemporaryDirectory() as temp:
            queue,record=self.fixture(Path(temp));events=[]
            def spawn(command,**kwargs):
                child=next(c for c in record['children'] if Path(command[2]).name==c['core'])
                return FakeChild(child,events=events)
            def finish(*args,**kwargs):
                events.append('mail');self.assertIsNone(kwargs['budget'])
                kwargs['before_stop']()
                self.assertTrue(json.loads((queue/'wall_budget.json').read_text())['invocation_finished'])
                events.append('stop');return True
            delivery=SimpleNamespace(finish=MagicMock(side_effect=finish))
            with patch.dict(sys.modules,{'fusion_overnight_delivery':delivery}),\
                    patch('torch.cuda.is_available',return_value=True),\
                    patch.object(runner,'make_plan',return_value=record),\
                    patch.object(runner,'create_queue',return_value=(queue,record)),\
                    patch.object(runner,'check_email_ready'),patch.object(runner,'preflight',return_value='target'),\
                    patch.object(runner.subprocess,'Popen',side_effect=spawn):
                with self.assertRaises(SystemExit) as result:runner.main([])
            self.assertEqual(result.exception.code,0);delivery.finish.assert_called_once()
            self.assertEqual(events,['pairwise_done','hcsi_done','mail','stop'])
            self.assertTrue(json.loads((queue/'completion.json').read_text())['shutdown_after_completion'])

    def test_check_only_and_failed_preflight_never_send_or_stop(self):
        with tempfile.TemporaryDirectory() as temp:
            queue,record=self.fixture(Path(temp));delivery=SimpleNamespace(finish=MagicMock())
            with patch.dict(sys.modules,{'fusion_overnight_delivery':delivery}),\
                    patch('torch.cuda.is_available',return_value=True),\
                    patch.object(runner,'make_plan',return_value=record),\
                    patch.object(runner,'create_queue') as create,\
                    patch.object(runner,'check_email_ready'),patch.object(runner,'preflight') as preflight:
                runner.main(['--check-only']);create.assert_not_called()
                preflight.side_effect=RuntimeError('No managed stop permission')
                with self.assertRaises(RuntimeError):runner.main([])
            delivery.finish.assert_not_called()

    def test_start_failure_before_first_child_never_arms_stop(self):
        with tempfile.TemporaryDirectory() as temp:
            queue,record=self.fixture(Path(temp));delivery=SimpleNamespace(finish=MagicMock())
            with patch.dict(sys.modules,{'fusion_overnight_delivery':delivery}),\
                    patch('torch.cuda.is_available',return_value=True),\
                    patch.object(runner,'make_plan',return_value=record),\
                    patch.object(runner,'create_queue',return_value=(queue,record)),\
                    patch.object(runner,'check_email_ready'),patch.object(runner,'preflight',return_value='target'),\
                    patch.object(runner.subprocess,'Popen',side_effect=OSError('Cannot start')):
                with self.assertRaises(SystemExit) as result:runner.main([])
            self.assertEqual(result.exception.code,1);delivery.finish.assert_not_called()

    def test_completed_queue_resume_redelivers_and_stops_without_new_gpu_child(self):
        with tempfile.TemporaryDirectory() as temp:
            queue,record=self.fixture(Path(temp))
            runner.mark_execution_started(queue,record)
            for child in record['children']:
                (Path(child['suite'])/'completion.json').write_text(json.dumps(dict(status='complete')))
            def finish(*args,**kwargs):kwargs['before_stop']();return True
            delivery=SimpleNamespace(finish=MagicMock(side_effect=finish))
            with patch.dict(sys.modules,{'fusion_overnight_delivery':delivery}),\
                    patch('torch.cuda.is_available',return_value=True),\
                    patch.object(runner,'validate_resume',return_value=record),\
                    patch.object(runner,'check_email_ready'),patch.object(runner,'preflight',return_value='target'),\
                    patch.object(runner.subprocess,'Popen') as spawn:
                with self.assertRaises(SystemExit) as result:runner.main(['--resume',str(queue)])
            self.assertEqual(result.exception.code,0);spawn.assert_not_called()
            delivery.finish.assert_called_once()
            self.assertTrue(json.loads((queue/'wall_budget.json').read_text())['invocation_finished'])

    def test_execution_marker_is_bound_to_exact_parent_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            queue,record=self.fixture(Path(temp))
            self.assertFalse(runner.execution_has_started(queue,record))
            runner.mark_execution_started(queue,record)
            self.assertTrue(runner.execution_has_started(queue,record))
            changed=copy.deepcopy(record);changed['epochs']=20
            with self.assertRaisesRegex(ValueError,'marker differs'):
                runner.execution_has_started(queue,changed)

    def test_parent_plan_creation_resume_and_reference_tamper_checks(self):
        import pairwise_resolver_experiment as pairwise
        import hcsi_tiny_experiment as hcsi
        plans={
            'pairwise':dict(anchor=dict(checkpoint_sha256='ref'),epochs=30,seed=42,split_hash='split',
                shutdown_after_completion=False,smoke=False),
            'hcsi':dict(reference=dict(checkpoint_sha256='ref'),epochs=30,seed=42,split_hash='split',
                shutdown_after_completion=False,smoke=False)}
        def create(args,plan=None):
            target=Path(args.output)/'suite_test';target.mkdir(parents=True)
            (target/'suite.json').write_text(json.dumps(plan));return target
        def validate(path):return json.loads((Path(path)/'suite.json').read_text())
        modules={name:SimpleNamespace(arguments=(pairwise if name=='pairwise' else hcsi).arguments,
            make_plan=MagicMock(return_value=copy.deepcopy(plan)),create_suite=create,validate_resume=validate)
            for name,plan in plans.items()}
        with tempfile.TemporaryDirectory() as temp:
            args=runner.arguments().parse_args(['--output',str(Path(temp)/'new')])
            with patch.object(runner,'engines',return_value=modules),\
                    patch.object(runner,'source_hashes',return_value={'new':'hash'}):
                plan=runner.make_plan(args)
                self.assertTrue(plan['shutdown_after_completion'])
                self.assertTrue(all(p['shutdown_after_completion'] is False for p in plan['child_plans'].values()))
                self.assertTrue(all('standalone wrapper10h limit is not applied' in p['timing_policy']
                    for p in plan['child_plans'].values()))
                queue,record=runner.create_queue(args,plan)
                self.assertEqual(runner.validate_resume(queue),record)
                changed=copy.deepcopy(record);changed['total_new_model_epochs']=1
                (queue/'queue.json').write_text(json.dumps(changed))
                with self.assertRaisesRegex(ValueError,'identity'):runner.validate_resume(queue)
                changed=copy.deepcopy(record);changed['children'][0]['suite']=str(Path(temp)/'elsewhere')
                (queue/'queue.json').write_text(json.dumps(changed))
                with self.assertRaisesRegex(ValueError,'escaped'):runner.validate_resume(queue)

    def test_child_creation_pins_both_to_same_reference_without_changing_child_stop_policy(self):
        def parser(name):
            p=argparse.ArgumentParser()
            for flag in ('cnn-suite','vit-suite','output','examples','epochs','data-dir','num-workers'):
                p.add_argument('--'+flag)
            p.add_argument('--base-suite' if name=='pairwise' else '--reference-suite')
            p.add_argument('--smoke',action='store_true');p.add_argument('--allow-download',action='store_true')
            return p
        modules={name:SimpleNamespace(arguments=lambda n=name:parser(n)) for name in runner.ORDER}
        args=runner.arguments().parse_args(['--reference-suite','chosen','--epochs','30'])
        pair=runner.child_arguments(args,'pairwise','new/pairwise',modules)
        hcsi=runner.child_arguments(args,'hcsi','new/hcsi',modules)
        self.assertEqual(pair.base_suite,hcsi.reference_suite)
        self.assertEqual(pair.base_suite,'chosen')


if __name__=='__main__':unittest.main()
