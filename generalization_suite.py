"""Tiny ImageNet data audit and generalization repair study.

One seed, independent hypotheses, matched controls, and auditable best/last
models. Complete every finite scheduled job by default; an explicit time
budget remains available. The wrapper handles one email and Workbench stop.
"""
import argparse
import copy
import gc
import json
import math
from pathlib import Path
import threading
import time
import traceback

import torch

from baseline_diagnostics import clean_datasets
from correction_experiment import file_hash
from literature_experiment import ROOT, new_name, recipe_hash, save_json
from mechanism_diagnostics import load_spatial_parent
from spatial_experiment import execution_environment, source_hashes as spatial_sources


PROTOCOL = 'tiny-generalization-overnight-v1'
DEFAULT_PARENT = ROOT/'runs/spatial_tiny/suite_20261103T175622_a19c86/cnn8_seed42'
DEFAULT_PREVIOUS = ROOT/'runs/mechanism_tiny/suite_20261104T014344_7caf47/fusion30'
NEW_SOURCES = ('generalization_suite.py', 'generalization_audit.py',
               'generalization_last.py', 'generalization_experiment.py',
               'models/generalization_preact.py', 'run_generalization_overnight.py',
               'fusion30_experiment.py', 'models/spatial_fusion.py')


def source_hashes():
    return {**spatial_sources(), **{name: file_hash(ROOT/name) for name in NEW_SOURCES}}


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


class ActiveBudget:
    """Persist active runtime, with an optional cutoff retained across resume."""
    def __init__(self, suite, max_hours, interval=30.):
        self.path = Path(suite)/'budget.json'
        self.max_seconds = None if max_hours is None else max_hours*3600
        self.previous = 0.
        if self.path.exists():
            state = json.loads(self.path.read_text(encoding='utf-8'))
            if state.get('protocol') != PROTOCOL or state.get('max_seconds') != self.max_seconds:
                raise ValueError('Saved active budget has a different protocol or time limit')
            self.previous = float(state['elapsed_active_seconds'])
            if not math.isfinite(self.previous) or self.previous < 0:
                raise ValueError('Invalid saved active runtime')
        self.started = time.monotonic()
        self.interval = interval
        self.stop_event = threading.Event()
        self.worker = None

    def elapsed(self):
        return self.previous+time.monotonic()-self.started

    def remaining(self):
        if self.max_seconds is None:
            return None
        return max(0., self.max_seconds-self.elapsed())

    def write(self):
        atomic_json(self.path, dict(protocol=PROTOCOL, max_seconds=self.max_seconds,
            elapsed_active_seconds=self.elapsed(),
            complete_all=self.max_seconds is None,
            note=('Active child runtime; heartbeat30s. Complete all finite scheduled jobs without time cutoffs.'
                  if self.max_seconds is None else
                  'Active child runtime; heartbeat30s. Cooperative epoch cutoff, not a hung-process watchdog.')))

    def _heartbeat(self):
        while not self.stop_event.wait(self.interval):
            try:
                self.write()
            except OSError as exc:
                print(f'Budget heartbeat write failed ({type(exc).__name__}).', flush=True)

    def __enter__(self):
        self.write()
        self.worker = threading.Thread(target=self._heartbeat, daemon=True)
        self.worker.start()
        return self

    def __exit__(self, *exc):
        self.stop_event.set()
        if self.worker is not None:
            self.worker.join(timeout=5.)
        self.write()


def _outside(output, parent, previous):
    for source in (parent, previous):
        if output == source or source in output.parents:
            raise ValueError('New output must be outside the original checkpoint/suite directories')


def make_plan(args):
    warm = 1 if args.smoke else (30 if args.warm_epochs is None else args.warm_epochs)
    fresh = 1 if args.smoke else (100 if args.fresh_epochs is None else args.fresh_epochs)
    complete_all = bool(getattr(args, 'complete_all', False) or args.max_hours is None)
    if getattr(args, 'complete_all', False) and args.max_hours is not None:
        raise ValueError('Choose complete-all or an explicit max_hours budget, not both')
    hours = None if complete_all else args.max_hours
    if not 1 <= warm <= 30 or not 1 <= fresh <= 100:
        raise ValueError('Warm epochs must be1..30 and fresh epochs1..100')
    if hours is not None and (not math.isfinite(hours) or not .25 <= hours <= 48):
        raise ValueError('max_hours must be a finite number in0.25..48')
    maximum_reserve = 120 if hours is None else min(120, hours*30)
    if not 5 <= args.reserve_minutes <= maximum_reserve:
        raise ValueError('Reserve5..120 minutes, at most half the total budget, for final analysis/email')
    if args.examples < 0:
        raise ValueError('Examples must be nonnegative')
    parent, summary, state, checkpoint_hash = load_spatial_parent(args.parent_run)
    config = copy.deepcopy(state['config'])
    del state
    data = Path(args.data_dir or config['dataset']['data_dir']).expanduser()
    config['dataset']['data_dir'] = str((data if data.is_absolute() else ROOT/data).resolve())
    config['dataset']['download'] = bool(args.allow_download)
    if args.num_workers is not None:
        config['dataset']['num_workers'] = args.num_workers
    if config['dataset']['num_workers'] < 0 or config['dataset']['batch_size'] % 2:
        raise ValueError('Need nonnegative workers and an even batch size')
    previous = Path(args.previous_fusion).expanduser().resolve()
    _outside(Path(args.output).resolve(), parent, previous)
    train, selection, _, fingerprint, _ = clean_datasets(config, summary['seed'], summary['split_hash'], 0)
    if len(train) != 80000 or len(selection) != 10000:
        raise ValueError('Need the complete80K/10K development partitions')
    return dict(protocol=PROTOCOL, parent_run=str(parent), previous_fusion=str(previous),
        parent_checkpoint_sha256=checkpoint_hash, seed=summary['seed'], split_hash=fingerprint,
        config=config, config_recipe_hash=recipe_hash(config), source_hashes=source_hashes(),
        execution_environment=execution_environment(config, summary['seed']),
        warm_epochs=warm, fresh_epochs=fresh, max_hours=hours, complete_all=complete_all,
        reserve_minutes=args.reserve_minutes,
        examples=args.examples, smoke=bool(args.smoke), stages=['data_audit', 'previous_last', 'training'],
        audit_partition_evaluated=False, official_validation_evaluated=False,
        note='Development screening: fix only supported problems; a large train/selection gap is not proof of a dataset bug')


def create_suite(args, plan=None):
    plan = make_plan(args) if plan is None else plan
    root = Path(args.output).resolve()
    _outside(root, Path(plan['parent_run']), Path(plan['previous_fusion']))
    suite = root/new_name('smoke' if plan['smoke'] else 'suite')
    suite.mkdir(parents=True, exist_ok=False)
    save_json(suite/'suite.json', plan)
    print(f'Generalization suite: {suite}', flush=True)
    return suite


def validate_resume(path):
    suite = Path(path).resolve()
    plan = json.loads((suite/'suite.json').read_text(encoding='utf-8'))
    if plan.get('protocol') != PROTOCOL or plan.get('source_hashes') != source_hashes():
        raise ValueError('Resume requires the recorded generalization protocol and source files')
    if recipe_hash(plan['config']) != plan['config_recipe_hash']:
        raise ValueError('Saved configuration changed')
    complete_all = plan.get('complete_all', plan['max_hours'] is None)
    if complete_all != (plan['max_hours'] is None):
        raise ValueError('Saved complete-all/time-budget mode is inconsistent')
    _, summary, state, checkpoint_hash = load_spatial_parent(plan['parent_run'])
    del state
    if (checkpoint_hash != plan['parent_checkpoint_sha256'] or summary['seed'] != plan['seed']
            or summary['split_hash'] != plan['split_hash']):
        raise ValueError('Parent checkpoint/split/seed identity changed')
    if execution_environment(plan['config'], plan['seed']) != plan['execution_environment']:
        raise ValueError('Resume requires the recorded software, GPU and precision environment')
    clean_datasets(plan['config'], plan['seed'], plan['split_hash'], 0)
    return plan


def run_suite(path):
    from generalization_audit import run_data_audit
    from generalization_last import run_previous_last
    from generalization_experiment import run_training_study
    suite = Path(path).resolve()
    plan = validate_resume(suite)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for the overnight generalization study')
    outcomes, fatal_data = {}, False
    complete_all = plan.get('complete_all', plan['max_hours'] is None)
    with ActiveBudget(suite, plan['max_hours']) as budget:
        work_deadline = (None if complete_all else
            time.monotonic()+max(0., budget.remaining()-plan['reserve_minutes']*60))

        def diagnostic_deadline():
            # Complete-all removes the old45-minute diagnostic stage limits as
            # well as the total cutoff. Each stage still has a finite workload.
            return None if complete_all else min(work_deadline, time.monotonic()+45*60)

        stages = (
            ('data_audit', lambda: run_data_audit(plan['parent_run'], suite/'data_audit',
                config=copy.deepcopy(plan['config']), examples=plan['examples'],
                previous_fusion=plan['previous_fusion'],
                deadline=diagnostic_deadline())),
            ('previous_last', lambda: run_previous_last(plan['parent_run'], plan['previous_fusion'],
                suite/'previous_last', config=copy.deepcopy(plan['config']), examples=plan['examples'],
                deadline=diagnostic_deadline())),
            ('training', lambda: run_training_study(plan['parent_run'], suite/'training',
                config=copy.deepcopy(plan['config']), epochs=plan['warm_epochs'],
                fresh_epochs=plan['fresh_epochs'], deadline=work_deadline,
                previous_fusion=plan['previous_fusion'], examples=plan['examples'],
                smoke=plan['smoke'], resume=True)),
        )
        for name, action in stages:
            if fatal_data:
                outcomes[name] = dict(status='blocked', reason='Concrete fatal data-integrity errors in the audit')
                continue
            if work_deadline is not None and time.monotonic() >= work_deadline:
                outcomes[name] = dict(status='deferred', reason='Preserving the final analysis/email reserve')
                continue
            timing = ('complete-all, no time cutoff' if complete_all else
                      f'remaining total budget={budget.remaining()/3600:.2f}h')
            print(f'\nGeneralization stage: {name}; {timing}', flush=True)
            atomic_json(suite/'status.json', dict(protocol=PROTOCOL, active_stage=name, outcomes=outcomes))
            try:
                result = action()
                stage_status = result.get('status', 'complete') if isinstance(result, dict) else 'complete'
                outcomes[name] = dict(status=stage_status)
                if name == 'data_audit' and isinstance(result, dict) and result.get('training_allowed') is False:
                    fatal_data = True
                    outcomes[name] = dict(status='failed', reason='Data-integrity audit disallows new training')
            except TimeoutError:
                outcomes[name] = dict(status='budget_exhausted', reason='Stage time limit; partial files retained')
            except Exception as exc:
                failure = dict(status='failed', error_type=type(exc).__name__, traceback=traceback.format_exc())
                outcomes[name] = failure
                directory = suite/name
                directory.mkdir(exist_ok=True)
                save_json(directory/'stage_failure.json', failure)
                print(f'Stage {name} failed ({type(exc).__name__}); independent stages will still be attempted.', flush=True)
            gc.collect()
            torch.cuda.empty_cache()
        unchanged = file_hash(Path(plan['parent_run'])/'best.pt') == plan['parent_checkpoint_sha256']
        failed = not unchanged or any(item['status'] in ('failed', 'blocked') for item in outcomes.values())
        partial = any(item['status'] not in ('complete', 'unavailable') for item in outcomes.values())
        status = 'failed' if failed else ('budget_exhausted' if partial else 'complete')
        result = dict(protocol=PROTOCOL, status=status, outcomes=outcomes,
            elapsed_active_hours=budget.elapsed()/3600, max_hours=plan['max_hours'], complete_all=complete_all,
            parent_checkpoint_unchanged=unchanged, smoke=plan['smoke'],
            warm_epochs=plan['warm_epochs'], fresh_epochs=plan['fresh_epochs'],
            audit_partition_evaluated=False, official_validation_evaluated=False)
        atomic_json(suite/'status.json', result)
        atomic_json(suite/'completion.json', result)
        timing_description = ('complete all finite scheduled jobs; no time cutoff'
                              if complete_all else f'active budget {plan["max_hours"]}h')
        (suite/'report.md').write_text('# Overnight generalization study\n\n'
            f'Status: **{status}**. Seed {plan["seed"]}; {timing_description}.\n\n'
            '- [Data/source/augmentation audit](data_audit/report.md)\n'
            '- [Actual previous last checkpoints](previous_last/report.md)\n'
            '- [Matched training remedies](training/report.md)\n\n'
            'Warm-start and from-scratch results have separate controls and horizons. '
            'A lower training score alone does not establish better generalization. '
            'Check all deferred/failed jobs before interpreting results.\n', encoding='utf-8')
        if failed:
            raise RuntimeError('Overnight study contains failed stages; see the packaged reports/status')
    print(f'Generalization study ended: {status}; {suite}', flush=True)
    return result


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-run', default=str(DEFAULT_PARENT))
    parser.add_argument('--previous-fusion', default=str(DEFAULT_PREVIOUS))
    parser.add_argument('--data-dir')
    parser.add_argument('--allow-download', action='store_true')
    parser.add_argument('--num-workers', type=int)
    parser.add_argument('--warm-epochs', '--epochs', dest='warm_epochs', type=int, help='Default30, maximum30 additional epochs')
    parser.add_argument('--fresh-epochs', type=int, help='Default100, maximum100 fresh epochs')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--complete-all', action='store_true',
                      help='Finish all finite scheduled jobs without total or stage time cutoffs (default)')
    mode.add_argument('--max-hours', type=float,
                      help='Optional total active child budget; retained on resume')
    parser.add_argument('--reserve-minutes', type=float, default=45,
                        help='With --max-hours, reserve for final evaluation/packaging; default45')
    parser.add_argument('--examples', type=int, default=8)
    parser.add_argument('--smoke', action='store_true', help='Separate one-epoch training screen; full data audit')
    parser.add_argument('--output', default=str(ROOT/'runs/generalization_tiny'))
    parser.add_argument('--resume')
    return parser


def main(argv=None):
    args = arguments().parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; no suite created')
    if args.resume:
        plan = validate_resume(args.resume)
        if args.smoke or any(getattr(args, key) is not None and getattr(args, key) != plan[key]
            for key in ('warm_epochs', 'fresh_epochs', 'max_hours')):
            raise ValueError('Resume retains the saved horizons and active time budget')
        if args.complete_all and not plan.get('complete_all', plan['max_hours'] is None):
            raise ValueError('Resume cannot change an existing time-budgeted suite to complete-all')
        suite = Path(args.resume)
    else:
        suite = create_suite(args)
    run_suite(suite)


if __name__ == '__main__':
    main()
