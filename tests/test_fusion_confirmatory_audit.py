import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch import nn

import fusion_confirmatory_audit as audit


TEMP = Path(__file__).resolve().parents[1]/'runs/fusion_confirmatory_audit_tests_tmp'


class TinyModel(nn.Module):
    def __init__(self, offset=0.):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.bn = nn.BatchNorm1d(3)
        self.classifier = nn.Linear(3, 2)
        with torch.no_grad():
            self.classifier.bias.add_(torch.tensor([offset, -offset]))

    def forward(self, image):
        if self.training or torch.is_grad_enabled():
            raise AssertionError('Confirmation requires eval/inference mode')
        return self.classifier(self.bn(self.pool(image).flatten(1)))


class OfficialAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        TEMP.mkdir(parents=True, exist_ok=True)

    def official(self, root):
        directory = root/'tiny-imagenet-200/val/images'
        directory.mkdir(parents=True)
        lines = []
        classes = {'n00000000':0, 'n00000001':1}
        for index in range(4):
            name = f'val_{index}.JPEG'
            Image.new('RGB', (64, 64), (20+index*20, 50, 90)).save(directory/name)
            label = 'n00000000' if index < 2 else 'n00000001'
            lines.append(f'{name}\t{label}\t0\t0\t63\t63')
        (directory.parent/'val_annotations.txt').write_text('\n'.join(reversed(lines))+'\n', encoding='utf-8')
        dataset = audit.OfficialValidation(directory.parents[1], classes, expected_classes=2, images_per_class=2)
        return dataset, directory, classes

    def fixture(self, root):
        dataset, directory, classes = self.official(root)
        for name in ('cnn', 'vit', 'reference', 'mechanism'):
            (root/name).mkdir()
        config = dict(dataset=dict(data_dir=str(root), batch_size=64, num_workers=0, split_seed=42),
            training=dict(seed=42, mixed_precision=False))
        inherited = dict(cnn_suite=str(root/'cnn'), vit_suite=str(root/'vit'),
            reference=dict(suite=str(root/'reference')), config=config,
            seed=42, split_hash='fixture-split', execution_environment={}, source_provenance={'cnn':'source'})
        torch.manual_seed(19)
        models = {name:TinyModel(index*.01).eval() for index, name in enumerate(audit.PHYSICAL_MODELS)}
        specifications, files = {}, {}
        for name in audit.PHYSICAL_MODELS:
            filename = audit.CHECKPOINTS.get(name, 'best.pt')
            run = root/'mechanism/training'/name
            run.mkdir(parents=True)
            path = run/filename
            torch.save(models[name].state_dict(), path)
            specs = dict(checkpoint=str(path), checkpoint_sha256=audit.file_hash(path), epoch=1, source='raw')
            specifications[name] = specs
            files[str(path)] = specs['checkpoint_sha256']
        inputs = dict(mechanism_plan=inherited, sources={}, models=specifications, source_files=files)
        args = audit.arguments().parse_args(['--mechanism-suite', str(root/'mechanism'),
            '--output', str(root/'confirmation')])
        return args, inputs, dataset, models

    def contexts(self, root, inputs, dataset):
        return (patch.object(audit, 'ROOT', root),
            patch.object(audit, '_inputs', return_value=inputs),
            patch.object(audit, '_official_data', return_value=dataset),
            patch.object(audit, 'source_hashes', return_value={'new':'source'}),
            patch.object(audit, 'execution_environment', return_value={}),
            patch.object(torch.cuda, 'is_available', return_value=False))

    def test_official_annotation_mapping_ids_normalization_and_content_seal(self):
        with tempfile.TemporaryDirectory(dir=TEMP) as temporary:
            dataset, directory, classes = self.official(Path(temporary))
            self.assertEqual(dataset.split, 'official_validation')
            self.assertEqual(dataset.image_ids, [f'official_val/val_{i}.JPEG' for i in range(4)])
            self.assertEqual(dataset.targets.tolist(), [0, 0, 1, 1])
            x, y, identifier = dataset[0]
            self.assertEqual(tuple(x.shape), (3, 64, 64))
            self.assertEqual(y, 0)
            self.assertEqual(identifier, 'official_val/val_0.JPEG')
            self.assertFalse(dataset.identity['augmentation'])
            self.assertEqual(json.loads(json.dumps(dataset.identity)), dataset.identity)
            initial = copy.deepcopy(dataset.identity)
            Image.new('RGB', (64, 64), 'red').save(directory/'val_0.JPEG')
            changed = audit.OfficialValidation(directory.parents[1], classes, expected_classes=2, images_per_class=2)
            self.assertNotEqual(changed.identity['image_manifest_sha256'], initial['image_manifest_sha256'])
            self.assertEqual(changed.identity['annotation_sha256'], initial['annotation_sha256'])

    def test_official_annotations_reject_duplicates_unsafe_unknown_and_missing_images(self):
        for modification in ('duplicate', 'unsafe', 'unknown', 'missing', 'resized'):
            with self.subTest(modification=modification), tempfile.TemporaryDirectory(dir=TEMP) as temporary:
                dataset, directory, classes = self.official(Path(temporary))
                annotation = directory.parent/'val_annotations.txt'
                text = annotation.read_text()
                if modification == 'duplicate':
                    annotation.write_text(text+text.splitlines()[0]+'\n')
                elif modification == 'unsafe':
                    annotation.write_text(text.replace('val_0.JPEG', '../val_0.JPEG'))
                elif modification == 'unknown':
                    annotation.write_text(text.replace('n00000000', 'unknown'))
                elif modification == 'missing':
                    (directory/'val_0.JPEG').unlink()
                else:
                    Image.new('RGB', (32, 32)).save(directory/'val_0.JPEG')
                with self.assertRaises(ValueError):
                    audit.OfficialValidation(directory.parents[1], classes, expected_classes=2, images_per_class=2)

    def test_official_factory_validates_original_split_and_evaluates_no_internal_images(self):
        class Internal:
            def __init__(self, count, folder): self.count, self.folder = count, folder
            def __len__(self): return self.count
            def __getitem__(self, index): raise AssertionError('Internal split was evaluated')
        with tempfile.TemporaryDirectory(dir=TEMP) as temporary:
            root = Path(temporary)
            dataset, directory, classes = self.official(root)
            from types import SimpleNamespace
            folder = SimpleNamespace(root=directory.parents[1]/'train', class_to_idx=classes)
            splits = {name:Internal(count, folder) for name, count in
                (('train',80000), ('selection',10000), ('audit',10000))}
            manifest = dict(train=['train/a'], selection=['selection/a'], audit=['audit/a'])
            with patch.object(audit, 'make_datasets', return_value=(splits, manifest, 'same')), \
                    patch.object(audit, 'OfficialValidation', return_value=dataset) as factory:
                self.assertIs(audit._official_data({'dataset':{}},42,'same'), dataset)
                self.assertEqual(factory.call_args.args[1], classes)
                with self.assertRaisesRegex(ValueError, 'identity'):
                    audit._official_data({'dataset':{}},42,'changed')
                manifest['selection'] = ['train/a']
                with self.assertRaisesRegex(ValueError, 'identity'):
                    audit._official_data({'dataset':{}},42,'same')

    def test_readonly_plan_registers_before_inference_and_blocks_new_output_for_same_split(self):
        with tempfile.TemporaryDirectory(dir=TEMP) as temporary:
            root = Path(temporary)
            args, inputs, dataset, models = self.fixture(root)
            a,b,c,d,e,f = self.contexts(root, inputs, dataset)
            with a,b,c,d,e,f:
                plan = audit.make_plan(args)
                self.assertFalse(Path(args.output).exists())
                self.assertFalse(audit._registration_path(plan).exists())
                self.assertEqual(plan['training_epochs'], 0)
                self.assertEqual(plan['evaluation_provenance']['target'], 'official_validation')
                self.assertTrue(plan['evaluation_provenance']['legacy_internal_audit_previously_evaluated'])
                suite = audit.create_suite(args, plan)
                self.assertFalse((suite/'evaluation_started.json').exists())
                self.assertEqual(audit.validate_resume(suite), plan)
                args.output = str(root/'another-output')
                with self.assertRaisesRegex(ValueError, 'already registered'):
                    audit.create_suite(args, plan)

    def test_real_plan_six_model_inference_stats_and_exact_completed_resume(self):
        with tempfile.TemporaryDirectory(dir=TEMP) as temporary:
            root = Path(temporary)
            args, inputs, dataset, models = self.fixture(root)
            calls = []
            a,b,c,d,e,f = self.contexts(root, inputs, dataset)
            with a,b,c,d,e,f:
                plan = audit.make_plan(args)
                suite = audit.create_suite(args, plan)
                def load(plan, name, sources):
                    self.assertTrue((suite/'audit_registration.json').is_file())
                    self.assertTrue((suite/'evaluation_started.json').is_file())
                    calls.append(name)
                    return copy.deepcopy(models[name])
                with patch.object(audit, '_load_model', side_effect=load):
                    result = audit.run_suite(suite)
                self.assertEqual(result['status'], 'complete', result['failures'])
                self.assertEqual(calls, list(audit.PHYSICAL_MODELS))
                self.assertTrue(result['official_validation_evaluated'])
                self.assertFalse(result['audit_partition_evaluated'])
                self.assertEqual(result['training_epochs'], 0)
                self.assertIsNone(result['failure'])
                self.assertEqual(len(result['outcomes']), 7)
                self.assertTrue((suite/'feature_analysis/report.md').is_file())
                before = {str(p):audit.file_hash(p) for p in (suite/'feature_analysis/predictions').rglob('*') if p.is_file()}
                with patch.object(audit, '_load_model', side_effect=AssertionError('Repeated evaluation')), \
                        patch.object(audit, 'evaluate_plain', side_effect=AssertionError('Repeated inference')):
                    second = audit.run_suite(suite)
                self.assertEqual(second['status'], 'complete', second['failures'])
                after = {str(p):audit.file_hash(p) for p in (suite/'feature_analysis/predictions').rglob('*') if p.is_file()}
                self.assertEqual(before, after)

    def test_registration_changed_weights_data_or_statistical_plan_rejected(self):
        for mutation in ('weights', 'data', 'plan'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory(dir=TEMP) as temporary:
                root = Path(temporary)
                args, inputs, dataset, models = self.fixture(root)
                a,b,c,d,e,f = self.contexts(root, inputs, dataset)
                with a,b,c,d,e,f:
                    plan = audit.make_plan(args)
                    suite = audit.create_suite(args, plan)
                    if mutation == 'weights':
                        inputs['models']['existing_joint']['checkpoint_sha256'] = 'changed'
                    elif mutation == 'data':
                        dataset.identity['annotation_sha256'] = 'changed'
                    else:
                        plan['statistical_plan']['alpha'] = .2
                        audit.save_json(suite/'suite.json', plan)
                    with self.assertRaises(ValueError):
                        audit.validate_resume(suite)

    def test_failure_partial_evidence_can_resume_without_repeating_successful_models(self):
        with tempfile.TemporaryDirectory(dir=TEMP) as temporary:
            root = Path(temporary)
            args, inputs, dataset, models = self.fixture(root)
            a,b,c,d,e,f = self.contexts(root, inputs, dataset)
            with a,b,c,d,e,f:
                suite = audit.create_suite(args, audit.make_plan(args))
                def partial(plan, name, sources):
                    if name == 'existing_joint': raise RuntimeError('interrupted inference')
                    return copy.deepcopy(models[name])
                with patch.object(audit, '_load_model', side_effect=partial):
                    first = audit.run_suite(suite)
                self.assertEqual(first['status'], 'failed')
                self.assertEqual(set(first['failures']), {'existing_joint'})
                called = []
                def recover(plan, name, sources):
                    called.append(name)
                    return copy.deepcopy(models[name])
                with patch.object(audit, '_load_model', side_effect=recover):
                    final = audit.run_suite(suite)
                self.assertEqual(final['status'], 'complete', final['failures'])
                self.assertEqual(called, ['existing_joint'])

    def test_cached_prediction_corruption_is_reported_not_silently_reevaluated(self):
        with tempfile.TemporaryDirectory(dir=TEMP) as temporary:
            root = Path(temporary)
            args, inputs, dataset, models = self.fixture(root)
            a,b,c,d,e,f = self.contexts(root, inputs, dataset)
            with a,b,c,d,e,f:
                suite = audit.create_suite(args, audit.make_plan(args))
                with patch.object(audit, '_load_model', side_effect=lambda p,n,s:copy.deepcopy(models[n])):
                    self.assertEqual(audit.run_suite(suite)['status'], 'complete')
                artifact = suite/'feature_analysis/predictions/existing_joint/metrics.json'
                artifact.write_text('{}')
                with patch.object(audit, '_load_model', side_effect=AssertionError('Reevaluated corrupt cache')):
                    final = audit.run_suite(suite)
                self.assertEqual(final['status'], 'failed')
                self.assertIn('existing_joint', final['failures'])

    def test_owned_registration_crash_recovery_is_readonly_until_execution(self):
        with tempfile.TemporaryDirectory(dir=TEMP) as temporary:
            root = Path(temporary)
            args, inputs, dataset, models = self.fixture(root)
            a,b,c,d,e,f = self.contexts(root, inputs, dataset)
            with a,b,c,d,e,f:
                plan = audit.make_plan(args)
                suite = audit.create_suite(args, plan)
                registry = audit._registration_path(plan)
                registry.unlink()
                self.assertEqual(audit.validate_resume(suite), plan)
                self.assertFalse(registry.exists(), 'Readonly preflight repaired a registration')
                with patch.object(audit, '_load_model', side_effect=lambda p,n,s:copy.deepcopy(models[n])):
                    self.assertEqual(audit.run_suite(suite)['status'], 'complete')
                self.assertTrue(registry.is_file())
                registry.unlink()
                with self.assertRaisesRegex(ValueError, 'after evaluation began'):
                    audit.validate_resume(suite)

    def test_failed_statistical_reporting_is_authoritative_and_resumes_from_cached_predictions(self):
        with tempfile.TemporaryDirectory(dir=TEMP) as temporary:
            root = Path(temporary)
            args, inputs, dataset, models = self.fixture(root)
            a,b,c,d,e,f = self.contexts(root, inputs, dataset)
            with a,b,c,d,e,f:
                suite = audit.create_suite(args, audit.make_plan(args))
                with patch.object(audit, '_load_model', side_effect=lambda p,n,s:copy.deepcopy(models[n])), \
                        patch('fusion_audit_statistics.write_report', side_effect=RuntimeError('interrupted report')):
                    first = audit.run_suite(suite)
                self.assertEqual(first['status'], 'failed')
                self.assertIn('statistical_report', first['failures'])
                self.assertEqual(json.loads((suite/'completion.json').read_text())['status'], 'failed')
                self.assertEqual(len(first['outcomes']), 7)
                with patch.object(audit, '_load_model', side_effect=AssertionError('Repeated model inference')), \
                        patch.object(audit, 'evaluate_plain', side_effect=AssertionError('Repeated model inference')):
                    final = audit.run_suite(suite)
                self.assertEqual(final['status'], 'complete', final['failures'])
                self.assertEqual(final['failures'], {})


if __name__ == '__main__':
    unittest.main()
