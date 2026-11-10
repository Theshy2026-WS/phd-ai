"""Fixed, read-only confirmation on the official Tiny ImageNet validation set.

All checkpoints and statistical comparisons are registered before inference.
Legacy internal audit images were already evaluated by early experiment code;
they are deliberately not the confirmation set used here. No model is trained.
"""
import argparse
from collections import Counter
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
from torchvision.transforms import functional as TF

from best_pair_diagnostics import evaluate_plain
from correction_experiment import file_hash
from datasets.utility_tiny import make_datasets
from fusion_continuation_diagnostics import (_validate_arrays, _write_result,
    load_probabilities, probability_result)
from literature_experiment import ROOT
from models.attention_correction import state_digest
from spatial_experiment import execution_environment


PROTOCOL = 'tiny-fixed-official-validation-confirmation-v1'
DEFAULT_MECHANISM = ROOT/'runs/fusion_mechanism_tiny/suite_20261109T165758_03cb08'
PHYSICAL_MODELS = ('source_cnn', 'source_vit', 'existing_joint',
    'random_vit_cls_linear', 'reset_cnn_head_cls_linear', 'patch_mean_linear')
ENSEMBLE = 'probability_90_10'
CHECKPOINTS = {'random_vit_cls_linear': 'best_trained.pt',
    'reset_cnn_head_cls_linear': 'best.pt', 'patch_mean_linear': 'best.pt'}
PINNED_MODELS = {
    'source_cnn': dict(checkpoint_sha256='9a230237ad61f204c76a7df009da04607fc3ebbf3cc33d3258c76d8f8d8592eb', epoch=83, source='ema'),
    'source_vit': dict(checkpoint_sha256='7728fa7c2b13887034e6c4133534b180fd4b5e2457ab4841b0efee828c8346ed', epoch=273, source='raw'),
    'existing_joint': dict(checkpoint_sha256='8172f674cb23d57c13c048630f786546fe46f95e8814acae164d105d4b96af48', epoch=7, source='ema'),
    'random_vit_cls_linear': dict(checkpoint_sha256='27f6b33ca20836dc0c475f9c2950010fcea361deb1486e81c961befec1f4c328', epoch=1, source='ema'),
    'reset_cnn_head_cls_linear': dict(checkpoint_sha256='ff26b5eabdd7e38d06429f286bdca6871e6879a32d512d9a29ba798a2db8da1a', epoch=20, source='ema'),
    'patch_mean_linear': dict(checkpoint_sha256='9f6aa427d064b711dc5ac48488ff5ce73b44ad1abacae28e26b7db416d8d9386', epoch=30, source='raw')}
EVALUATION_PROVENANCE = dict(target='official_validation',
    legacy_internal_audit_previously_evaluated=True,
    prior_official_validation_use='not_detected_in_provided_records',
    interpretation='A fixed confirmation evaluation of preselected models. Earlier internal audit '
        'metrics were exposed by legacy code; that partition is not treated as untouched. '
        'Available records do not prove the official validation set was never viewed elsewhere.')


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _json(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f'Need a regular JSON record: {path}')
    return json.loads(path.read_text(encoding='utf-8'))


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    for attempt in range(5):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(.05*(attempt+1))


def _real_directory(path):
    path = Path(path).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('Evaluation paths must not contain symlinks')
    return path.resolve()


class OfficialValidation(Dataset):
    """Strict native64 official-val adapter using the training class mapping."""
    split = 'official_validation'

    def __init__(self, root, class_to_idx, *, expected_classes=200, images_per_class=50):
        self.root = _real_directory(root)
        self.class_to_idx = dict(class_to_idx)
        if (len(self.class_to_idx) != expected_classes
                or set(self.class_to_idx.values()) != set(range(expected_classes))):
            raise ValueError('Official validation requires the original dense training class mapping')
        self.class_names = [name for name, _ in sorted(self.class_to_idx.items(), key=lambda item:item[1])]
        self.annotation = self.root/'val/val_annotations.txt'
        directory = _real_directory(self.root/'val/images')
        if not self.annotation.is_file() or self.annotation.is_symlink() or not directory.is_dir():
            raise ValueError('Need complete official val/images and val_annotations.txt; no fallback partition')
        entries = {}
        for line in self.annotation.read_text(encoding='utf-8').splitlines():
            fields = line.split('\t')
            if len(fields) != 6:
                raise ValueError('Official validation annotation must have six tab-separated fields')
            name, label = fields[:2]
            if (not name or Path(name).name != name or '/' in name or '\\' in name
                    or name in ('.', '..') or name in entries or label not in self.class_to_idx):
                raise ValueError('Duplicate, unsafe filename or unknown official validation class')
            try:
                tuple(int(value) for value in fields[2:])
            except ValueError as exc:
                raise ValueError('Malformed official validation bounding-box annotation') from exc
            entries[name] = self.class_to_idx[label]
        counts = Counter(entries.values())
        if (len(entries) != expected_classes*images_per_class
                or counts != Counter({i:images_per_class for i in range(expected_classes)})):
            raise ValueError('Expected exactly 50 official validation images per training class')
        actual = {p.name for p in directory.iterdir() if p.is_file()}
        if actual != set(entries):
            raise ValueError('Official annotation and image filenames differ')
        self.samples, self.image_ids, records = [], [], []
        for name in sorted(entries):
            path = directory/name
            if path.is_symlink() or not path.resolve().is_relative_to(directory):
                raise ValueError('Official validation image escaped the fixed directory')
            with Image.open(path) as image:
                if image.size != (64, 64):
                    raise ValueError('Official validation must remain native64; no resizing')
            identifier = 'official_val/'+name
            self.samples.append((path, entries[name]))
            self.image_ids.append(identifier)
            records.append(dict(image_id=identifier, target=entries[name], sha256=file_hash(path)))
        self.targets = np.asarray([target for _, target in self.samples], dtype=np.int64)
        self.epoch = 0
        self.identity = dict(partition=self.split, n=len(self), class_names=self.class_names,
            annotation_sha256=file_hash(self.annotation), image_manifest_sha256=_digest(records),
            image_ids_sha256=_digest(self.image_ids), targets_sha256=_digest(self.targets.tolist()),
            class_counts={str(key):value for key,value in sorted(counts.items())}, native_size=[3, 64, 64],
            normalization=dict(mean=[.485, .456, .406], std=[.229, .224, .225]),
            augmentation=False, resized=False)
        self.manifest = records

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        image = TF.normalize(TF.to_tensor(default_loader(str(path))),
            [.485, .456, .406], [.229, .224, .225])
        return image, target, self.image_ids[index]


def source_hashes():
    from fusion_mechanism_experiment import source_hashes as inherited
    names = ('fusion_confirmatory_audit.py', 'fusion_audit_statistics.py',
        'run_fusion_audit.py', 'fusion_audit_delivery.py')
    return {**inherited(), **{name:file_hash(ROOT/name) for name in names}}


def _inputs(mechanism_suite):
    import fusion_mechanism_experiment as mechanism
    suite = _real_directory(mechanism_suite)
    plan = mechanism.validate_resume(suite)
    completed = _json(suite/'completion.json')
    if (plan['epochs'] != 30 or plan.get('smoke') or completed.get('status') != 'complete'
            or completed.get('failure') is not None
            or completed.get('official_validation_evaluated') is not False
            or completed.get('outcomes', {}).get('feature_analysis', {}).get('status') != 'complete'
            or any(completed.get('outcomes', {}).get(job) != dict(status='complete', completed_epochs=30)
                   for job in mechanism.JOBS)):
        raise ValueError('Require the completed three-control30 suite before fixed confirmation')
    sources = mechanism.load_sources(plan)
    files = dict(sources['source_files'])
    for path in (suite/'suite.json', suite/'completion.json', suite/'feature_analysis/completion.json'):
        files[str(path)] = file_hash(path)
    specifications = dict(
        source_cnn=dict(checkpoint=str(Path(sources['parent_run'])/'best.pt'),
            checkpoint_sha256=sources['provenance']['parent_checkpoint_sha256'],
            epoch=sources['provenance']['source_cnn_best_epoch'], source=sources['provenance']['source_cnn_best_source']),
        source_vit=dict(checkpoint=sources['selected']['checkpoint'],
            checkpoint_sha256=sources['selected']['checkpoint_sha256'],
            epoch=sources['selected']['best_epoch'], source=sources['selected']['best_source']),
        existing_joint=dict(checkpoint=plan['reference']['checkpoint'],
            checkpoint_sha256=plan['reference']['checkpoint_sha256'],
            epoch=plan['reference']['epoch'], source=plan['reference']['source']))
    for job, filename in CHECKPOINTS.items():
        run = suite/'training'/job
        summary = _json(run/'summary.json')
        model, state = mechanism.load_model(plan, suite/'training', job, filename, sources=sources)
        del model
        prefix = 'best_trained' if filename == 'best_trained.pt' else 'best'
        if (summary.get('status') != 'complete' or summary.get('completed_epochs') != 30
                or state['epoch'] < 1 or summary[prefix+'_epoch'] != state['epoch']
                or summary[prefix+'_source'] != state['source']
                or summary[prefix+'_selection'] != state['selection']):
            raise ValueError('Preselected trained checkpoint differs from its completed control summary')
        specifications[job] = dict(checkpoint=str(run/filename), checkpoint_sha256=file_hash(run/filename),
            epoch=state['epoch'], source=state['source'], checkpoint_rule=filename,
            development_selection=state['selection'])
        for name in ('metadata.json', 'summary.json', 'epochs.csv', filename):
            files[str(run/name)] = file_hash(run/name)
    for record in specifications.values():
        if file_hash(record['checkpoint']) != record['checkpoint_sha256']:
            raise ValueError('Pinned source checkpoint changed')
        files[record['checkpoint']] = record['checkpoint_sha256']
    for name, record in specifications.items():
        if any(record.get(key) != value for key, value in PINNED_MODELS[name].items()):
            raise ValueError('Confirmation requires the specifically preselected development checkpoint, not a new candidate')
    return dict(mechanism_plan=plan, sources=sources, models=specifications, source_files=files)


def _official_data(config, seed, expected_split):
    datasets, manifest, fingerprint = make_datasets(config['dataset'], seed)
    if (fingerprint != expected_split or set(datasets) != {'train', 'selection', 'audit'}
            or [len(datasets[key]) for key in ('train', 'selection', 'audit')] != [80000, 10000, 10000]
            or any(set(manifest[a]) & set(manifest[b])
                for a, b in (('train','selection'), ('train','audit'), ('selection','audit')))):
        raise ValueError('Original training/development/internal-audit split identity changed')
    folder = datasets['train'].folder
    result = OfficialValidation(Path(folder.root).parent, folder.class_to_idx)
    # Only official validation is ever indexed by the evaluator.
    return result


def _separate_output(output, inputs):
    output = _real_directory(output)
    originals = [inputs['mechanism_plan']['cnn_suite'], inputs['mechanism_plan']['vit_suite'],
        inputs['mechanism_plan']['reference']['suite'], inputs['mechanism_plan'].get('suite_directory')]
    originals.append(str(Path(inputs['models']['random_vit_cls_linear']['checkpoint']).parents[2]))
    for value in originals:
        if value is None:
            continue
        path = _real_directory(value)
        if output == path or output.is_relative_to(path) or path.is_relative_to(output):
            raise ValueError('Confirmation output must be separate from immutable source suites')
    return output


def make_plan(args):
    from fusion_audit_statistics import statistical_plan
    inputs = _inputs(args.mechanism_suite)
    inherited = inputs['mechanism_plan']
    config = copy.deepcopy(inherited['config'])
    data = Path(args.data_dir or config['dataset']['data_dir']).expanduser()
    config['dataset']['data_dir'] = str((data if data.is_absolute() else ROOT/data).resolve())
    config['dataset']['download'] = False
    if args.num_workers is not None:
        config['dataset']['num_workers'] = args.num_workers
    if config['dataset']['num_workers'] < 0 or config['dataset']['batch_size'] != 64:
        raise ValueError('Require source batch64 and nonnegative workers')
    _separate_output(args.output, inputs)
    dataset = _official_data(config, inherited['seed'], inherited['split_hash'])
    environment = execution_environment(config, inherited['seed'])
    if environment != inherited['execution_environment']:
        raise ValueError('Confirmation requires the recorded software, GPU and precision environment')
    return dict(protocol=PROTOCOL, mechanism_suite=str(Path(args.mechanism_suite).resolve()),
        source_files=inputs['source_files'], source_hashes=source_hashes(), models=inputs['models'],
        model_order=list(PHYSICAL_MODELS), source_provenance=inherited['source_provenance'],
        config=config, seed=inherited['seed'], split_hash=inherited['split_hash'],
        execution_environment=environment, dataset_identity=dataset.identity,
        statistical_plan=statistical_plan(), evaluation_provenance=copy.deepcopy(EVALUATION_PROVENANCE),
        probability_ensemble=dict(name=ENSEMBLE, cnn_weight=.9, vit_weight=.1,
            weights_selected_on='previous_development_grid', tuning_on_official_validation=False),
        training_epochs=0, official_validation_planned=True, internal_audit_evaluated_by_this_run=False,
        shutdown_after_completion=False, no_model_selection_on_confirmation=True,
        registration_policy='One registered plan per original split; exact resume reuses checkpoint predictions.')


def _registration_path(plan):
    return ROOT/'runs/fusion_official_validation_registry'/f'{plan["split_hash"]}.json'


def _registration(plan, suite):
    return dict(protocol=PROTOCOL, plan_sha256=_digest(plan), suite=str(Path(suite).resolve()),
        target='official_validation', split_hash=plan['split_hash'])


def _validate_registration(suite, plan, *, repair=False):
    expected = _registration(plan, suite)
    if _json(suite/'audit_registration.json') != expected:
        raise ValueError('Local confirmation registration differs')
    registry = _registration_path(plan)
    if registry.exists():
        if _json(registry) != expected:
            raise ValueError('Confirmation registration differs; do not create or retune another run')
        return
    prediction_root = suite/'feature_analysis/predictions'
    if ((suite/'evaluation_started.json').exists()
            or (prediction_root.exists() and any(prediction_root.iterdir()))):
        raise ValueError('Global registration disappeared after evaluation began; refusing a new confirmation')
    # A crash between writing local receipts and the exclusive global receipt
    # precedes any inference. Validation is read-only; execution repairs exactly
    # that owned registration after the wrapper's normal preflight.
    if repair:
        registry.parent.mkdir(parents=True, exist_ok=True)
        try:
            with registry.open('x', encoding='utf-8') as stream:
                json.dump(expected, stream, indent=2)
                stream.flush(); os.fsync(stream.fileno())
        except FileExistsError:
            if _json(registry) != expected:
                raise ValueError('A competing confirmation registration was created')


def create_suite(args, plan=None):
    plan = make_plan(args) if plan is None else plan
    suite = _real_directory(args.output)
    registry = _registration_path(plan)
    if registry.exists():
        prior = _json(registry)
        raise ValueError(f'Official validation already registered; use --resume {prior.get("suite")}')
    if suite.exists() and any(suite.iterdir()):
        raise ValueError('Refusing an existing confirmation directory; use its exact --resume path')
    suite.mkdir(parents=True, exist_ok=True)
    record = _registration(plan, suite)
    save_json(suite/'suite.json', plan)
    save_json(suite/'audit_registration.json', record)
    registry.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents simultaneous or differently located fresh runs.
    with registry.open('x', encoding='utf-8') as stream:
        json.dump(record, stream, indent=2)
        stream.flush(); os.fsync(stream.fileno())
    print(f'Fixed official-validation suite: {suite}', flush=True)
    print('Six fixed checkpoints, one fixed90/10 probability ensemble; zero training epochs.', flush=True)
    return suite


def validate_resume(path):
    from fusion_audit_statistics import statistical_plan
    suite = _real_directory(path)
    plan = _json(suite/'suite.json')
    if (plan.get('protocol') != PROTOCOL or plan.get('source_hashes') != source_hashes()
            or plan.get('model_order') != list(PHYSICAL_MODELS)
            or plan.get('statistical_plan') != statistical_plan()
            or plan.get('evaluation_provenance') != EVALUATION_PROVENANCE
            or plan.get('training_epochs') != 0 or plan.get('shutdown_after_completion') is not False
            or plan.get('no_model_selection_on_confirmation') is not True
            or plan.get('probability_ensemble') != dict(name=ENSEMBLE, cnn_weight=.9, vit_weight=.1,
                weights_selected_on='previous_development_grid', tuning_on_official_validation=False)):
        raise ValueError('Registered confirmation sources, comparisons or evaluation-only policy changed')
    _validate_registration(suite, plan)
    inputs = _inputs(plan['mechanism_suite'])
    if inputs['source_files'] != plan['source_files'] or inputs['models'] != plan['models']:
        raise ValueError('Preselected checkpoint or historical source identity changed')
    dataset = _official_data(plan['config'], plan['seed'], plan['split_hash'])
    if dataset.identity != plan['dataset_identity']:
        raise ValueError('Official validation image content, annotations or class mapping changed')
    if execution_environment(plan['config'], plan['seed']) != plan['execution_environment']:
        raise ValueError('Registered confirmation environment changed')
    return plan


def _load_model(plan, name, inputs):
    import fusion_mechanism_experiment as mechanism
    if name == 'source_cnn':
        from models.spatial_preact import build_spatial_model
        model = build_spatial_model('cnn8', plan['seed'])
        model.load_state_dict(inputs['sources']['parent_model'], strict=True)
    elif name == 'source_vit':
        from models.vit_augreg import build_vit
        selected = inputs['sources']['selected']
        model = build_vit(selected['profile'], plan['seed'])
        model.load_state_dict(selected['state']['model'], strict=True)
    elif name == 'existing_joint':
        model, _ = mechanism.load_reference_model(inputs['mechanism_plan'], sources=inputs['sources'])
    else:
        model, _ = mechanism.load_model(inputs['mechanism_plan'],
            Path(plan['mechanism_suite'])/'training', name, CHECKPOINTS[name], sources=inputs['sources'])
    return model


def _cache_identity(plan, name):
    return dict(plan_sha256=_digest(plan), model=name,
        checkpoint_sha256=plan['models'][name]['checkpoint_sha256'] if name in PHYSICAL_MODELS else None)


def _load_cache(path, identity, dataset):
    _real_directory(path)
    marker = path/'record.json'
    if not marker.exists():
        return None
    record = _json(marker)
    if record.get('identity') != identity or not record.get('files'):
        raise ValueError('Cached prediction belongs to another registered model or plan')
    for relative, digest in record['files'].items():
        piece = Path(relative)
        item = path/piece
        if (piece.is_absolute() or '..' in piece.parts or item.is_symlink()
                or not item.resolve().is_relative_to(path.resolve()) or file_hash(item) != digest):
            raise ValueError('Cached official-validation predictions changed')
    return load_probabilities(path/'predictions.npz', dataset.class_names,
        dataset.image_ids, dataset.targets)


def _save_result(path, identity, result):
    _real_directory(path)
    if path.exists():
        previous = path.with_name(path.name+'__incomplete_'+str(time.time_ns()))
        path.rename(previous)
    _write_result(path, result)
    files = {p.relative_to(path).as_posix():file_hash(p) for p in path.rglob('*') if p.is_file()}
    save_json(path/'record.json', dict(identity=identity, files=files))


def run_suite(path):
    from fusion_audit_statistics import evaluate_audit_results, write_report
    suite = _real_directory(path)
    plan = validate_resume(suite)
    _validate_registration(suite, plan, repair=True)
    inputs = _inputs(plan['mechanism_suite'])
    dataset = _official_data(plan['config'], plan['seed'], plan['split_hash'])
    output = suite/'feature_analysis'
    output.mkdir(exist_ok=True)
    save_json(output/'evaluation_manifest.json', dataset.manifest)
    results, failures, outcomes = {}, {}, {}
    began = time.perf_counter()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for name in PHYSICAL_MODELS:
        model = None
        try:
            cache = output/'predictions'/name
            identity = _cache_identity(plan, name)
            value = _load_cache(cache, identity, dataset)
            if value is None:
                # This durable marker precedes all access to evaluation logits.
                if not (suite/'evaluation_started.json').exists():
                    save_json(suite/'evaluation_started.json', dict(protocol=PROTOCOL,
                        plan_sha256=_digest(plan), target='official_validation'))
                model = _load_model(plan, name, inputs).to(device).float().eval()
                before = state_digest(model)
                value = evaluate_plain(model, dataset, plan['config'], plan['seed'], device, dataset.class_names)
                _validate_arrays(value, dataset.class_names, dataset.image_ids, dataset.targets)
                if state_digest(model) != before:
                    raise RuntimeError('Read-only confirmation changed model parameters or normalization buffers')
                _save_result(cache, identity, value)
            results[name] = value
            outcomes[name] = dict(status='complete', metrics=value['metrics'])
            print(f'{name}: official Top1={100*value["metrics"]["accuracy"]:.2f}%', flush=True)
        except Exception as exc:
            failures[name] = dict(error_type=type(exc).__name__, traceback=traceback.format_exc())
            outcomes[name] = dict(status='failed')
        finally:
            if model is not None:
                del model
            gc.collect()
            if device == 'cuda':
                torch.cuda.empty_cache()
    if 'source_cnn' in results and 'source_vit' in results:
        try:
            cnn, vit = results['source_cnn'], results['source_vit']
            _validate_arrays(vit, cnn['class_names'], cnn['image_ids'], cnn['targets'])
            result = probability_result(.9*cnn['probabilities']+.1*vit['probabilities'],
                cnn['targets'], cnn['image_ids'], cnn['class_names'])
            cache = output/'predictions'/ENSEMBLE
            identity = _cache_identity(plan, ENSEMBLE)
            if _load_cache(cache, identity, dataset) is None:
                _save_result(cache, identity, result)
            results[ENSEMBLE] = result
            outcomes[ENSEMBLE] = dict(status='complete', metrics=result['metrics'])
        except Exception as exc:
            failures[ENSEMBLE] = dict(error_type=type(exc).__name__, traceback=traceback.format_exc())
            outcomes[ENSEMBLE] = dict(status='failed')
    summary = None
    if len(results) == len(PHYSICAL_MODELS)+1 and not failures:
        try:
            summary = evaluate_audit_results(results, plan=plan['statistical_plan'])
            summary['evaluation_provenance'] = plan['evaluation_provenance']
            write_report(output, summary)
        except Exception as exc:
            failures['statistical_report'] = dict(error_type=type(exc).__name__, traceback=traceback.format_exc())
    unchanged = all(Path(name).is_file() and file_hash(name) == digest
                    for name, digest in plan['source_files'].items())
    try:
        fresh = _official_data(plan['config'], plan['seed'], plan['split_hash'])
        if fresh.identity != plan['dataset_identity']:
            raise ValueError('Validation data changed during evaluation')
    except Exception as exc:
        failures['dataset_integrity'] = dict(error_type=type(exc).__name__, traceback=traceback.format_exc())
    if not unchanged:
        failures['source_integrity'] = dict(error_type='ValueError', message='Pinned source artifacts changed')
    complete = summary is not None and not failures
    result = dict(protocol=PROTOCOL, status='complete' if complete else 'failed',
        outcomes=outcomes, failure=None if not failures else dict(message='Incomplete fixed-model confirmation'),
        failures=failures, statistics=summary, elapsed_seconds=time.perf_counter()-began,
        source_checkpoints_unchanged=unchanged, training_epochs=0,
        official_validation_evaluated=(suite/'evaluation_started.json').is_file(),
        complete_official_validation_evaluation=len(results)==len(PHYSICAL_MODELS)+1,
        audit_partition_evaluated=False, legacy_internal_audit_previously_evaluated=True,
        evaluation_provenance=plan['evaluation_provenance'], shutdown_after_completion=False,
        no_model_selection_on_confirmation=True)
    save_json(suite/'completion.json', result)
    save_json(output/'completion.json', result)
    save_json(output/'failures.json', failures)
    (suite/'report.md').write_text('# Fixed official-validation confirmation\n\n'
        f'Execution status: **{result["status"]}**. Zero training epochs.\n\n'
        'Six development-selected checkpoints and one fixed90/10 probability ensemble. '
        'No checkpoint, mixture weight or model is selected using these results.\n\n'
        '[Statistical report](feature_analysis/report.md)\n\n'
        'The experiment core performs no email or instance operation. '
        'run_fusion_audit.py handles final report email followed by managed stop.\n', encoding='utf-8')
    return result


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mechanism-suite', default=str(DEFAULT_MECHANISM))
    parser.add_argument('--data-dir')
    parser.add_argument('--num-workers', type=int)
    parser.add_argument('--output', '--output-dir', default=str(ROOT/'runs/fusion_confirmatory_audit'))
    parser.add_argument('--resume')
    return parser


def main(argv=None):
    args = arguments().parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; no official-validation inference or registration started')
    suite = Path(args.resume) if args.resume else create_suite(args)
    result = run_suite(suite)
    return 0 if result['status'] == 'complete' else 2


if __name__ == '__main__':
    raise SystemExit(main())
