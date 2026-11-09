"""Recover HCSI analysis only, email the existing nine-job reports, then stop.

No training function or experiment subprocess is called by this entry point.
Original source hashes, checkpoint identities and completed horizons are checked
before the repaired analysis is allowed to run. Historical failure markers are
copied into an immutable recovery attempt before current status is refreshed.
"""
import argparse
import copy
import gc
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

import torch

from correction_experiment import file_hash
from fusion_overnight_delivery import finish
from literature_experiment import new_name
from overnight_twostage import preflight
from run_hcsi_tiny import _write_json
from run_vit_augreg_overnight import check_email_ready


ROOT = Path(__file__).resolve().parent
PROTOCOL = 'tiny-fusion-analysis-only-recovery-v1'
HISTORICAL_MARKERS = (
    'completion.json', 'status.json', 'report.md', 'failure.json',
    'delivery_status.json', 'wall_budget.json',
)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue', '--resume', required=True, dest='queue',
        help='The completed nine-job queue; no new training is scheduled')
    parser.add_argument('--to', default='shenwenfei@gmail.com')
    parser.add_argument('--examples', type=int)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--fallback-os-shutdown', action='store_true')
    return parser


def _regular(path, root):
    path, root = Path(path).absolute(), Path(root).absolute()
    if (not path.is_file() or path.is_symlink()
            or any(part.is_symlink() for part in path.parents)
            or not path.resolve().is_relative_to(root.resolve())):
        raise ValueError('Recovery inputs must be regular files inside their declared study')
    return path


def _read(path, root):
    return json.loads(_regular(path, root).read_text(encoding='utf-8'))


def artifact_inventory(children):
    """Pin every pairwise artifact and every HCSI training artifact, streamed."""
    inventory = {}
    for name, suite in children.items():
        suite = Path(suite).resolve()
        directories = (suite,) if name == 'pairwise' else (suite/'training',)
        for directory in directories:
            if directory.is_symlink() or not directory.is_dir():
                raise ValueError('Expected a real completed study artifact directory')
            for current, folders, files in os.walk(directory, followlinks=False):
                current = Path(current)
                if any((current/folder).is_symlink() for folder in folders):
                    raise ValueError('Refusing symlink study artifacts')
                for filename in sorted(files):
                    path = _regular(current/filename, suite)
                    key = name+'/'+path.relative_to(suite).as_posix()
                    inventory[key] = dict(path=str(path), sha256=file_hash(path))
    return inventory


def verify_inventory(inventory):
    for key, value in inventory.items():
        path = Path(value['path'])
        if not path.is_file() or path.is_symlink() or file_hash(path) != value['sha256']:
            raise RuntimeError(f'Recovery altered a completed artifact: {key}')


def validate_completed_queue(queue):
    """Read old protocol validators and full checkpoints; never call training."""
    from run_fusion_overnight import validate_resume
    import pairwise_resolver_experiment as pairwise
    import hcsi_tiny_experiment as hcsi

    queue = Path(queue).absolute()
    if queue.is_symlink() or any(path.is_symlink() for path in queue.parents):
        raise ValueError('Recovery queue cannot contain symlink ancestors')
    queue = queue.resolve()
    record = validate_resume(queue)
    children = {child['name']: Path(child['suite']).resolve() for child in record['children']}
    pair_status = _read(children['pairwise']/'completion.json', children['pairwise'])
    pair_analysis = _read(children['pairwise']/'feature_analysis/completion.json', children['pairwise'])
    if pair_status.get('status') != 'complete' or pair_analysis.get('status') != 'complete':
        raise ValueError('Pairwise training and analysis must already be complete; recovery will not retrain them')
    counts = {}
    for name, module in (('pairwise', pairwise), ('hcsi', hcsi)):
        plan = record['child_plans'][name]
        training = children[name]/'training'
        for job in module.JOBS:
            run = training/job
            summary = _read(run/'summary.json', children[name])
            if (summary.get('status') != 'complete'
                    or summary.get('completed_epochs') != plan['epochs']):
                raise ValueError(f'{name}/{job} did not finish its planned horizon; analysis-only recovery refuses to train')
            loader = module.load_resolver_model if name == 'pairwise' else module.load_model
            model, last = loader(plan, training, job, checkpoint='last.pt')
            if (last.get('epoch') != plan['epochs']
                    or [row['epoch'] for row in last.get('history', [])] != list(range(plan['epochs']+1))):
                raise ValueError(f'{name}/{job} has no complete atomic last checkpoint')
            del model, last
            gc.collect()
            model, best = loader(plan, training, job, checkpoint='best.pt')
            if (best.get('epoch') != summary.get('best_epoch')
                    or best.get('source') != summary.get('best_source')
                    or best.get('selection') != summary.get('best_selection')):
                raise ValueError(f'{name}/{job} best checkpoint differs from its completed summary')
            del model, best
            gc.collect()
            counts[name+'/'+job] = dict(completed_epochs=summary['completed_epochs'],
                planned_epochs=plan['epochs'])
    if len(counts) != 9:
        raise ValueError('Expected exactly nine completed model jobs')
    return queue, record, children, counts, artifact_inventory(children)


def create_attempt(queue, children, record, counts, inventory):
    attempt = Path(queue)/'analysis_recovery'/new_name('attempt')
    attempt.mkdir(parents=True, exist_ok=False)
    backups = {}
    for name, root in (('queue', Path(queue)), ('hcsi', children['hcsi'])):
        for filename in HISTORICAL_MARKERS:
            source = root/filename
            if not source.exists():
                continue
            _regular(source, root)
            destination = attempt/'original'/name/filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(source.read_bytes())
            backups[name+'/'+filename] = dict(path=str(destination), sha256=file_hash(destination))
    manifest = dict(protocol=PROTOCOL, queue=str(queue),
        queue_source_hashes=record['source_hashes'], counts=counts,
        checkpoint_and_pairwise_artifacts=inventory, historical_markers=backups,
        training_scheduled=False, new_model_epochs=0,
        pairwise_analysis_reused=True, only_recomputed_component='hcsi feature_analysis',
        old_source_files_modified=False, instance_policy='email then managed stop')
    _write_json(attempt/'manifest.json', manifest)
    _write_json(Path(queue)/'analysis_recovery_manifest.json',
        dict(manifest, recovery_attempt=str(attempt)))
    return attempt


def run_repaired_analysis(plan, training, output, examples):
    from hcsi_analysis_recovery import run_hcsi_recovery_diagnostics
    return run_hcsi_recovery_diagnostics(plan, training, output, examples=examples)


def remove_archived_failure_markers(queue, children, attempt):
    """Clear stale current failures only after checking their preserved copies."""
    manifest = _read(Path(attempt)/'manifest.json', attempt)
    pending = []
    for name, root in (('queue', Path(queue)), ('hcsi', children['hcsi'])):
        current = root/'failure.json'
        if not current.exists():
            continue
        _regular(current, root)
        saved = manifest.get('historical_markers', {}).get(name+'/failure.json')
        if not saved:
            raise RuntimeError('A current failure marker has no verified historical backup')
        backup = _regular(saved['path'], attempt)
        if file_hash(backup) != saved['sha256'] or file_hash(current) != saved['sha256']:
            raise RuntimeError('Historical failure backup or current marker changed')
        pending.append(current)
    for current in pending:
        current.unlink()


def publish_recovery(queue, record, children, attempt, result):
    if result.get('status') != 'complete':
        raise RuntimeError('Recovered HCSI analysis is incomplete; completed scores are not fabricated')
    child = children['hcsi']
    previous = _read(child/'completion.json', child)
    remove_archived_failure_markers(queue, children, attempt)
    updated = copy.deepcopy(previous)
    updated.update(status='complete', failure=None,
        analysis_recovery=dict(protocol=PROTOCOL, directory=str(attempt), training_repeated=False,
            historical_failure_preserved=True))
    updated.setdefault('outcomes', {})['feature_analysis'] = dict(status='complete', recovered=True)
    _write_json(child/'completion.json', updated)
    _write_json(child/'status.json', updated)
    (child/'report.md').write_text('# HCSI analysis recovered without retraining\n\n'
        'All five training jobs were already complete. Only feature analysis was recomputed. '
        'The previous failure markers are preserved in the parent analysis_recovery attempt.\n\n'
        '- [Training](training/report.md)\n- [Recovered feature evidence](feature_analysis/report.md)\n', encoding='utf-8')
    outcomes = {name: dict(status='complete', exit_code=0, child_status='complete',
        reused_completed=True, analysis_recovered=name == 'hcsi') for name in children}
    completion = dict(protocol=record['protocol'], status='complete', exit_code=0,
        outcomes=outcomes, failure=None, shutdown_after_completion=True,
        instance_policy='email then managed stop', total_model_jobs=9,
        audit_partition_evaluated=False, official_validation_evaluated=False,
        analysis_recovery=dict(protocol=PROTOCOL, directory=str(attempt),
            new_model_epochs=0, original_training_unchanged=True))
    _write_json(Path(queue)/'completion.json', completion)
    _write_json(Path(queue)/'status.json', completion)
    text = '# Combined fusion results: analysis recovered\n\n'
    text += 'All nine training jobs were already complete. Pairwise results were reused; '
    text += 'only the failed HCSI analysis was repaired. No model was trained again.\n\n'
    for name, suite in children.items():
        relative = suite.relative_to(queue).as_posix()
        text += f'- [{name}]({relative}/report.md)\n'
    text += '\nHistorical failure records remain in analysis_recovery/.\n'
    (Path(queue)/'report.md').write_text(text, encoding='utf-8')


def main(argv=None):
    args = arguments().parse_args(argv)
    if args.examples is not None and args.examples < 0:
        raise ValueError('Examples must be nonnegative')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for the recorded inference environment; no analysis or shutdown started')
    queue, record, children, counts, inventory = validate_completed_queue(args.queue)
    check_email_ready(args.to)
    if args.fallback_os_shutdown:
        subprocess.run(['sudo', '-n', '-l', '/usr/sbin/shutdown', '-h', 'now'], check=True)
    url = preflight()
    if args.check_only:
        print('All nine completed jobs validated. No training, analysis, transmitted email or shutdown.', flush=True)
        return 0
    attempt = create_attempt(queue, children, record, counts, inventory)
    log_path = attempt/'run.log'
    started = time.perf_counter()
    code, armed, failure = 1, False, None
    result = None
    def before_stop():
        status = dict(protocol=PROTOCOL, status='complete' if code == 0 else 'failed',
            recovery_attempt=str(attempt), new_model_epochs=0,
            training_repeated=False, pairwise_analysis_reused=True,
            elapsed_analysis_seconds=time.perf_counter()-started,
            invocation_finished=True, failure=failure,
            instance_policy='email then managed stop')
        _write_json(attempt/'completion.json', status)
        _write_json(queue/'analysis_recovery_status.json', status)
    try:
        with log_path.open('a', encoding='utf-8', buffering=1) as log:
            try:
                armed = True
                print('All nine models are complete. Recovering HCSI analysis only; no retraining.', flush=True)
                plan = record['child_plans']['hcsi']
                examples = plan['examples'] if args.examples is None else args.examples
                result = run_repaired_analysis(plan, children['hcsi']/'training',
                    children['hcsi']/'feature_analysis', examples)
                verify_inventory(inventory)
                publish_recovery(queue, record, children, attempt, result)
                code = 0
            except BaseException as exc:
                failure = dict(error_type=type(exc).__name__, traceback=traceback.format_exc())
                log.write(failure['traceback'])
                # Check artifact integrity even when analysis itself fails.
                try:
                    verify_inventory(inventory)
                except BaseException as integrity:
                    failure['artifact_integrity_error'] = str(integrity)
            finally:
                log.flush()
                os.fsync(log.fileno())
    finally:
        if armed:
            try:
                before_stop()
            except Exception as exc:
                print(f'Could not write recovery status ({type(exc).__name__}); final delivery and stop still follow.', flush=True)
            delivered = finish(queue, log_path, args.to, code, url,
                args.fallback_os_shutdown, before_stop=before_stop)
            if code == 0 and not delivered:
                code = 2
        else:
            before_stop()
    return code


if __name__ == '__main__':
    raise SystemExit(main())
