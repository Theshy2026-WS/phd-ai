import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import fusion_continuation_suite as runner


class Sized:
    def __init__(self,n): self.n=n
    def __len__(self): return self.n


class ContinuationSuiteTests(unittest.TestCase):
    def fixture(self,root,extra=()):
        source,parent=root/'source',root/'parent'
        source.mkdir(exist_ok=True);parent.mkdir(exist_ok=True)
        original=parent/'best.pt'
        original.write_bytes(b'parent')
        config=dict(dataset=dict(data_dir='data',download=False,batch_size=64,num_workers=4),
                    training=dict(epochs=100,learning_rate=.0005,seed=42))
        supplied=dict(source_suite=str(source),parent_run=str(parent),config=config,seed=42,
            split_hash='split',source_files={str(original):runner.file_hash(original)},
            provenance=dict(source_execution_environment={}))
        args=runner.arguments().parse_args(['--source-suite',str(source),'--output',str(root/'new'),*extra])
        return args,supplied

    def plan(self,args,supplied):
        with patch.object(runner,'load_sources',return_value=supplied), \
                patch.object(runner,'clean_datasets',return_value=(Sized(80000),Sized(10000),[],'split',
                    dict(train_clean=['train/a'],selection=['selection/b']))), \
                patch.object(runner,'execution_environment',return_value={}), \
                patch.object(runner,'source_hashes',return_value={'new':'source'}):
            return runner.make_plan(args)

    def test_extra100_not_total100_and_no_expensive_route(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);args,supplied=self.fixture(root)
            original=copy.deepcopy(supplied['config'])
            plan=self.plan(args,supplied)
            self.assertEqual(plan['continue_epochs'],100)
            self.assertEqual(plan['final_joint_epochs'],130)
            self.assertEqual(plan['total_new_model_epochs'],260)
            self.assertEqual(plan['jobs'],['dual_continue','cnn_head_control','cnn_head_continue','dual_frozen_head'])
            self.assertFalse(any('shallow' in job or '64' in job for job in plan['jobs']))
            self.assertEqual(plan['continuation_policy']['augmentation_epochs'],[131,230])
            self.assertEqual(plan['continuation_policy']['optimizer'],'preserved AdamW moments/steps')
            self.assertEqual(supplied['config'],original)
            self.assertTrue(plan['complete_all']);self.assertIsNone(plan['max_hours'])
            suite=runner.create_suite(args,plan)
            self.assertEqual(json.loads((suite/'suite.json').read_text())['final_joint_epochs'],130)

    def test_source_directory_and_ancestors_never_become_new_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);args,supplied=self.fixture(root)
            for output in (str(root/'source'),str(root),str(root/'source'/'nested')):
                args.output=output
                with self.assertRaisesRegex(ValueError,'separate'):
                    self.plan(args,supplied)

    def test_smoke_and_invalid_horizons(self):
        with tempfile.TemporaryDirectory() as temp:
            args,supplied=self.fixture(Path(temp),['--smoke'])
            plan=self.plan(args,supplied)
            self.assertEqual((plan['continue_epochs'],plan['control_epochs']),(1,1))
            self.assertEqual(plan['total_new_model_epochs'],4)
            args.smoke=False;args.continue_epochs=101
            with self.assertRaisesRegex(ValueError,'continuation1..100'):
                self.plan(args,supplied)

    def test_different_training_environment_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            args,supplied=self.fixture(Path(temp))
            supplied['provenance']['source_execution_environment']={'device':'otherGPU'}
            with self.assertRaisesRegex(ValueError,'original GPU'):
                self.plan(args,supplied)

    def run_stages(self,root,training=None,analysis=None):
        original=root/'original.pt';original.write_bytes(b'original')
        plan=dict(source_suite='source',config={},continue_epochs=100,control_epochs=30,
            examples=0,smoke=False,source_files={str(original):runner.file_hash(original)},
            excluded_training_routes=['shallow64'])
        trained=training or Mock(return_value=dict(status='complete'))
        analyzed=analysis or Mock(return_value=dict(status='complete'))
        with patch.object(runner,'validate_resume',return_value=plan), \
                patch('fusion_continuation_experiment.run_continuation_study',trained), \
                patch('fusion_continuation_diagnostics.run_continuation_diagnostics',analyzed), \
                patch.object(runner.torch.cuda,'is_available',return_value=True), \
                patch.object(runner.torch.cuda,'empty_cache'):
            result=runner.run_suite(root)
        return result,trained,analyzed

    def test_training_and_analysis_keep_new_phase_horizon(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);result,trained,analyzed=self.run_stages(root)
            self.assertEqual(result['status'],'complete')
            self.assertEqual(result['final_joint_epochs'],130)
            self.assertEqual(trained.call_args.kwargs['continue_epochs'],100)
            self.assertEqual(analyzed.call_args.args[2],root/'feature_analysis')
            self.assertTrue(result['source_checkpoints_unchanged'])

    def test_failed_training_still_analyzes_available_results(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);analyzed=Mock(return_value=dict(status='complete'))
            with self.assertRaisesRegex(RuntimeError,'contains failures'):
                self.run_stages(root,training=Mock(side_effect=RuntimeError('fixture')),analysis=analyzed)
            analyzed.assert_called_once()
            self.assertEqual(json.loads((root/'completion.json').read_text())['status'],'failed')

    def test_resume_source_change_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            (root/'suite.json').write_text(json.dumps(dict(protocol=runner.PROTOCOL,source_hashes={'old':'source'})))
            with patch.object(runner,'source_hashes',return_value={'new':'source'}):
                with self.assertRaisesRegex(ValueError,'source files'):
                    runner.validate_resume(root)


if __name__=='__main__':
    unittest.main()
