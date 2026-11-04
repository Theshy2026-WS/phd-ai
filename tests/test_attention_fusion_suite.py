import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import attention_fusion_suite as runner


class Sized:
    def __init__(self, n): self.n=n
    def __len__(self): return self.n


class AttentionSuiteTests(unittest.TestCase):
    def test_default_is_attention_only_complete_all_and_parent_recipe_is_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); parent=root/'parent'; parent.mkdir()
            (parent/'best.pt').write_bytes(b'parent')
            config=dict(dataset=dict(data_dir='data',download=True,batch_size=64,num_workers=4),
                        training=dict(epochs=100,learning_rate=.0005,seed=42))
            loaded=(parent,dict(seed=42,split_hash='split',best_epoch=100),dict(config=config),'hash')
            args=runner.arguments().parse_args(['--parent-run',str(parent),'--output',str(root/'new')])
            with patch.object(runner,'load_spatial_parent',return_value=loaded), \
                    patch.object(runner,'clean_datasets',return_value=(Sized(80000),Sized(10000),[], 'split',
                        dict(train_clean=['train/a'],selection=['selection/b']))), \
                    patch.object(runner,'execution_environment',return_value={}), \
                    patch.object(runner,'source_hashes',return_value={'x':'y'}):
                plan=runner.make_plan(args)
            self.assertEqual(plan['stages'],['training','attention_analysis'])
            self.assertTrue(plan['complete_all']); self.assertIsNone(plan['max_hours'])
            self.assertEqual((plan['warm_epochs'],plan['fresh_epochs']),(30,100))
            self.assertEqual(plan['config']['training']['epochs'],100)
            self.assertFalse(plan['config']['dataset']['download'])
            self.assertFalse((root/'new').exists())
            path=runner.create_suite(args,plan)
            self.assertTrue((path/'suite.json').is_file())

    def run_stages(self, root, training=None, analysis=None):
        parent=root/'parent'; parent.mkdir(); (parent/'best.pt').write_bytes(b'parent')
        plan=dict(parent_run=str(parent),parent_checkpoint_sha256=runner.file_hash(parent/'best.pt'),
            config={},warm_epochs=30,fresh_epochs=100,max_hours=None,complete_all=True,examples=0,smoke=False)
        trained=training or Mock(return_value=dict(status='complete'))
        analyzed=analysis or Mock(return_value=dict(status='complete'))
        with patch.object(runner,'validate_resume',return_value=plan), \
                patch('attention_fusion_experiment.run_attention_study',trained), \
                patch('attention_fusion_diagnostics.run_attention_diagnostics',analyzed), \
                patch.object(runner.torch.cuda,'is_available',return_value=True), \
                patch.object(runner.torch.cuda,'empty_cache'):
            result=runner.run_suite(root)
        return result,trained,analyzed

    def test_training_then_analysis_never_launches_generalization_queue(self):
        with tempfile.TemporaryDirectory() as temp:
            result,trained,analyzed=self.run_stages(Path(temp))
            self.assertEqual(result['status'],'complete')
            self.assertIsNone(trained.call_args.kwargs['deadline'])
            self.assertIsNone(analyzed.call_args.kwargs['deadline'])
            self.assertEqual(trained.call_args.kwargs['warm_epochs'],30)
            self.assertEqual(trained.call_args.kwargs['fresh_epochs'],100)

    def test_training_failure_still_analyzes_available_checkpoints_and_reports_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            analyzed=Mock(return_value=dict(status='complete'))
            with self.assertRaisesRegex(RuntimeError,'contains failures'):
                self.run_stages(Path(temp),training=Mock(side_effect=RuntimeError('fixture')),analysis=analyzed)
            analyzed.assert_called_once()
            status=json.loads((Path(temp)/'completion.json').read_text())
            self.assertEqual(status['outcomes']['training']['status'],'failed')


if __name__ == '__main__': unittest.main()
