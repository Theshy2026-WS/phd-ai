import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
import argparse
import torch
import pandas as pd
from PIL import Image
from torchvision.datasets import ImageFolder
from models.audit_repvit import AuditRepViT
from datasets.utility_tiny import UtilityDataset, loader
from utility_experiment import sample_metrics, pair_predictions, occlude, train, audit
from utility_diagnostics import intervention, checkpoint_diagnostic, paired_tables, auc_rank


class UtilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_models_shared_weights_shapes_and_gradients(self):
        baseline = AuditRepViT('cnn', 42)
        shared = baseline.backbone.state_dict()
        states = []
        for variant in ('conv','attention'):
            model = AuditRepViT(variant,42)
            for k,v in model.backbone.state_dict().items():
                self.assertTrue(torch.equal(v,shared[k]),k)
            seen = []
            h = model.addon.register_forward_pre_hook(lambda m, args: seen.append(args[0].shape))
            output = model(torch.randn(2,3,64,64))
            h.remove()
            self.assertEqual(tuple(output.shape),(2,200))
            self.assertEqual(tuple(seen[0]),(2,192,4,4))
            output.square().mean().backward()
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in model.addon.parameters()))
            if variant == 'attention':
                self.assertGreater(model.addon.mixer.in_proj_weight.grad.abs().sum().item(), 0)
            states.append(model.addon.state_dict())
            model.eval()
            x = torch.randn(2,3,64,64)
            before = {k:v.clone() for k,v in model.state_dict().items()}
            with torch.no_grad():
                full = model(x)
                with intervention(model, 'mixer_off'):
                    off = model(x)
                self.assertFalse(torch.equal(full, off))
                with intervention(model, 'addon_off'):
                    bypass = model(x)
                expected = model.backbone.stem(x)
                for stage in model.backbone.stages:
                    expected = stage(expected)
                expected = model.backbone.forward_head(expected)
                torch.testing.assert_close(bypass, expected)
                torch.testing.assert_close(model(x), full, rtol=0, atol=0)
            for k,v in model.state_dict().items():
                self.assertTrue(torch.equal(v,before[k]),k)
        for k in states[0]:
            if not k.startswith('mixer.'):
                self.assertTrue(torch.equal(states[0][k],states[1][k]),k)

    def test_pairing_and_utility(self):
        frames = [pd.DataFrame(dict(sample_id=['a'],target=[0],occlusion=[0.],
                    nll=[v],true_margin=[-v])) for v in [3.,2.,1.]]
        result = pair_predictions(frames)
        self.assertEqual(result.utility_vs_cnn.iloc[0],2.)
        self.assertEqual(result.utility_vs_conv.iloc[0],1.)
        frames[2]['sample_id'] = 'b'
        with self.assertRaises(ValueError):
            pair_predictions(frames)
        m=sample_metrics(torch.tensor([[5.,0.,0.,0.,0.]]),torch.tensor([0]))
        self.assertTrue(m['correct'].item())
        self.assertEqual(occlude(torch.ones(1,3,64,64),0.).sum(),3*64*64)

    def test_synthetic_end_to_end(self):
        # No real dataset download; exercise train/checkpoint/profiling/audit on images.
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'images'/'class0').mkdir(parents=True)
            for i in range(8):
                Image.new('RGB',(64,64),(i*20,30,40)).save(root/'images'/'class0'/f'{i}.png')
            folder=ImageFolder(root/'images')
            def fake_data(config,seed):
                ds={k:UtilityDataset(folder,ids,k,seed) for k,ids in
                    [('train',[0,1,2,3]),('selection',[4,5]),('audit',[6,7])]}
                return ds, {'synthetic':True}, 'synthetic'
            import yaml
            config=yaml.safe_load(Path('configs/tiny_imagenet_utility.yaml').read_text())
            config['dataset'].update(batch_size=2,num_workers=0)
            config['profiling'].update(warmup_iterations=0,benchmark_iterations=1)
            path=root/'config.yaml'; path.write_text(yaml.safe_dump(config))
            with patch('utility_experiment.make_datasets',fake_data):
                runs={}
                for variant in ('cnn','conv','attention'):
                    runs[variant]=str(train(argparse.Namespace(config=str(path),variant=variant,
                        seed=42,epochs=1,output=str(root/'runs'))))
                audit_dir = audit(argparse.Namespace(**runs,output=str(root/'audits')))
            with patch('datasets.utility_tiny.make_datasets', fake_data):
                diagnostic = checkpoint_diagnostic(argparse.Namespace(
                    audit_dir=str(audit_dir), output=str(root/'diagnostics')))
            replay = pd.read_csv(diagnostic/'replay_check.csv')
            self.assertEqual(replay.prediction_mismatches.sum(), 0)
            self.assertLess(replay.max_abs_nll_change.max(), 1e-5)
            summary = pd.read_csv(diagnostic/'intervention_summary.csv')
            self.assertEqual(len(summary), 18)
            paired=pd.read_csv(next((root/'audits').glob('*/paired_utility.csv')))
            self.assertEqual(len(paired),6)
            self.assertTrue(paired.utility_vs_conv.notna().all())
            ds=fake_data({},42)[0]['train']
            first=ds[0][0]
            torch.rand(100)
            self.assertTrue(torch.equal(first,ds[0][0]))
            a=[ids for _,_,ids in loader(ds,{'batch_size':2,'num_workers':0},42)]
            torch.rand(100)
            b=[ids for _,_,ids in loader(ds,{'batch_size':2,'num_workers':0},42)]
            self.assertEqual(a,b)

    def test_diagnostic_pair_counts_and_auc(self):
        frame = pd.DataFrame(dict(sample_id=['a','b','c','d'], target=[0]*4, occlusion=[0.]*4))
        for name, pred in [('cnn',[0,0,1,1]),('conv',[0,1,0,1]),('attention',[1,0,0,0])]:
            frame[name+'_pred'] = pred
            frame[name+'_correct'] = frame[name+'_pred'].eq(0)
            frame[name+'_nll'] = [1.,2.,3.,4.]
            frame[name+'_entropy'] = [0.5,1.5,2.5,3.5]
            frame[name+'_confidence'] = [0.9,0.7,0.5,0.3]
            frame[name+'_uncertainty_margin'] = [0.8,0.6,0.4,0.2]
        pairs, bins, signals = paired_tables(frame)
        row = pairs.iloc[0]
        self.assertEqual((row.corrected,row.harmed,row.oracle_accuracy),(2,1,1.))
        self.assertEqual(len(pairs),3)
        self.assertEqual(auc_rank([0,1],[1.,1.]),0.5)
        self.assertEqual(auc_rank([0,1],[0.,1.]),1.)
        self.assertTrue(pd.isna(auc_rank([1,1],[0.,1.])))
        with self.assertRaises(ValueError):
            paired_tables(pd.concat([frame,frame.iloc[:1]]))


if __name__ == '__main__':
    unittest.main()
