import argparse
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import torch
import yaml
from PIL import Image
from torchvision.datasets import ImageFolder

from datasets.utility_tiny import UtilityDataset
from datasets.resnet_tiny import prepare_datasets, mix_batch
from models.audit_resnet18 import AuditResNet18, LateSelfAttention
from resnet_experiment import train, audit, pair_predictions
from resnet_experiment import atomic_save


class ResNetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_shared_initialization_spatial_shape_and_gradients(self):
        cnn, att = AuditResNet18('cnn'), AuditResNet18('attention')
        for key, value in cnn.backbone.state_dict().items():
            self.assertTrue(torch.equal(value,att.backbone.state_dict()[key]),key)
        shape = []
        hook = att.addon.register_forward_pre_hook(lambda m,args: shape.append(tuple(args[0].shape)))
        x = torch.randn(2,3,64,64)
        output = att(x)
        hook.remove()
        self.assertEqual(shape,[(2,512,8,8)])
        self.assertEqual(tuple(output.shape),(2,200))
        output.square().mean().backward()
        self.assertGreater(att.addon.attention.in_proj_weight.grad.abs().sum().item(),0)
        cnn.eval()
        with torch.no_grad():
            torch.testing.assert_close(cnn(x),cnn.backbone(x),rtol=0,atol=0)
        with self.assertRaises(ValueError):
            LateSelfAttention(heads=0)

    def test_pair_rejects_label_mismatch(self):
        a = pd.DataFrame(dict(sample_id=['a'],occlusion=[0.],target=[0],correct=[False],nll=[2.]))
        b = a.copy(); b['correct']=True; b['nll']=1.
        p = pair_predictions(a,b)
        self.assertTrue(p.rescued.iloc[0])
        self.assertEqual(p.utility_nll.iloc[0],1.)
        b['target']=1
        with self.assertRaises(ValueError):
            pair_predictions(a,b)

    def test_synthetic_train_resume_and_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'images'/'class0').mkdir(parents=True)
            for i in range(8):
                Image.new('RGB',(64,64),(i*20,40,60)).save(root/'images'/'class0'/f'{i}.png')
            folder=ImageFolder(root/'images')
            def fake_data(config,seed):
                ds={k:UtilityDataset(folder,ids,k,seed) for k,ids in
                    [('train',[0,1,2,3]),('selection',[4,5]),('audit',[6,7])]}
                return ds, {'synthetic':True}, 'synthetic'
            config=yaml.safe_load(Path('configs/tiny_imagenet_resnet18.yaml').read_text())
            config['dataset'].update(batch_size=2,num_workers=0)
            config['profiling'].update(warmup_iterations=0,benchmark_iterations=1)
            config['training']['mixed_precision']=False
            path=root/'config.yaml'; path.write_text(yaml.safe_dump(config))
            with patch('datasets.resnet_tiny.make_datasets',fake_data):
                ds,_,_=prepare_datasets(config,42)
                first=ds['train'][0][0]
                torch.randn(50)
                torch.testing.assert_close(first,ds['train'][0][0],rtol=0,atol=0)
                x=torch.randn(2,3,64,64); y=torch.tensor([0,1])
                mixed,targets=mix_batch(x.clone(),y,config['training'],42,1,0)
                other,other_targets=mix_batch(x.clone(),y,config['training'],42,1,0)
                torch.testing.assert_close(mixed,other,rtol=0,atol=0)
                torch.testing.assert_close(targets,other_targets,rtol=0,atol=0)
                torch.testing.assert_close(targets.sum(1),torch.ones(2))
                runs={}
                for variant in ('cnn','attention'):
                    args=argparse.Namespace(config=str(path),output=str(root/'runs'),seed=42,
                                            epochs=1,variant=variant,resume=None)
                    runs[variant]=train(args)
                    args.resume=str(runs[variant])
                    self.assertEqual(train(args),runs[variant])
                    frame=pd.read_csv(runs[variant]/'epochs.csv')
                    self.assertEqual(len(frame),1)
                    self.assertIn('clean_accuracy_gap',frame)
                    self.assertNotIn('train_accuracy',frame)
                destination=audit(runs['cnn'],runs['attention'],root/'runs')
                self.assertEqual(len(pd.read_csv(destination/'paired_utility.csv')),6)
                self.assertEqual(len(pd.read_csv(destination/'metrics.csv')),6)
                # An interrupted two-epoch schedule must resume like the uninterrupted run.
                args=argparse.Namespace(config=str(path),output=str(root/'interrupted'),seed=42,
                                        epochs=2,variant='cnn',resume=None)
                def save_then_stop(path,state):
                    atomic_save(path,state)
                    if path.name=='last.pt':
                        raise InterruptedError('Synthetic interruption at epoch boundary')
                with patch('resnet_experiment.atomic_save',save_then_stop):
                    with self.assertRaises(InterruptedError):
                        train(args)
                partial=next((root/'interrupted').iterdir())
                args.resume=str(partial)
                resumed=train(args)
                args.resume=None; args.output=str(root/'uninterrupted')
                complete=train(args)
                a=torch.load(resumed/'last.pt',weights_only=False,map_location='cpu')
                b=torch.load(complete/'last.pt',weights_only=False,map_location='cpu')
                self.assertEqual(a['epoch'],2)
                for key,value in a['model'].items():
                    torch.testing.assert_close(value,b['model'][key],rtol=0,atol=0)


if __name__=='__main__':
    unittest.main()
