"""Run pairwise and HCSI sequentially, email their combined reports, then stop Workbench.

Only this parent owns delivery and shutdown. The two existing experiment cores
remain unchanged and never send mail or stop the instance themselves.
"""
import argparse
import copy
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from correction_experiment import file_hash
from literature_experiment import new_name
from overnight_twostage import preflight
from run_hcsi_tiny import (ChildWatchdog, _stop_child, _write_json,
    budget_deadlines, hours_argument)
from run_vit_augreg_overnight import check_email_ready, child_environment


ROOT = Path(__file__).resolve().parent
PROTOCOL = 'tiny-combined-fusion-overnight-v1'
ORDER = ('pairwise', 'hcsi')
CORES = dict(pairwise='pairwise_resolver_experiment.py', hcsi='hcsi_tiny_experiment.py')


def engines():
    import pairwise_resolver_experiment
    import hcsi_tiny_experiment
    return dict(pairwise=pairwise_resolver_experiment, hcsi=hcsi_tiny_experiment)


def arguments():
    from best_pair_sources import DEFAULT_CNN_SOURCE, DEFAULT_VIT_SOURCE
    from hcsi_tiny_experiment import DEFAULT_REFERENCE
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cnn-suite', default=str(DEFAULT_CNN_SOURCE))
    p.add_argument('--vit-suite', default=str(DEFAULT_VIT_SOURCE))
    p.add_argument('--reference-suite', default=str(DEFAULT_REFERENCE))
    p.add_argument('--data-dir'); p.add_argument('--num-workers', type=int)
    p.add_argument('--epochs', type=int)
    p.add_argument('--examples', type=int, default=8)
    p.add_argument('--allow-download', action='store_true')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--max-hours', type=hours_argument,
        help='Optional cumulative time cap; omitted runs all nine jobs to completion')
    p.add_argument('--output', '--output-dir', default=str(ROOT/'runs/fusion_overnight'))
    p.add_argument('--resume')
    p.add_argument('--to', default='shenwenfei@gmail.com')
    p.add_argument('--check-only', action='store_true')
    p.add_argument('--fallback-os-shutdown', action='store_true')
    return p


def source_hashes():
    names = ('run_fusion_overnight.py', 'fusion_overnight_delivery.py',
        'run_hcsi_tiny.py', 'run_vit_augreg_overnight.py', 'overnight_twostage.py')
    return {name: file_hash(ROOT/name) for name in names}


def child_arguments(args, name, output, modules=None):
    modules = engines() if modules is None else modules
    values = ['--cnn-suite', str(args.cnn_suite), '--vit-suite', str(args.vit_suite),
        '--base-suite' if name == 'pairwise' else '--reference-suite', str(args.reference_suite),
        '--output', str(output), '--examples', str(args.examples)]
    for flag, value in (('--epochs', args.epochs), ('--data-dir', args.data_dir),
                       ('--num-workers', args.num_workers)):
        if value is not None:
            values += [flag, str(value)]
    if args.smoke:
        values.append('--smoke')
    if args.allow_download:
        values.append('--allow-download')
    return modules[name].arguments().parse_args(values)


def make_plan(args):
    modules = engines()
    plans = {name: modules[name].make_plan(child_arguments(args, name,
        Path(args.output)/name, modules)) for name in ORDER}
    pair, hcsi = plans['pairwise'], plans['hcsi']
    if (pair['anchor']['checkpoint_sha256'] != hcsi['reference']['checkpoint_sha256']
            or pair['epochs'] != hcsi['epochs'] or pair['seed'] != hcsi['seed']
            or pair['split_hash'] != hcsi['split_hash']):
        raise ValueError('Both studies must share the completed joint reference, split, seed and horizon')
    if any(plan.get('shutdown_after_completion') is not False for plan in plans.values()):
        raise ValueError('Only the parent may request shutdown')
    for plan in plans.values():
        plan['timing_policy'] = ('Execution and final email/stop are managed by the combined parent. '
            'The standalone wrapper10h limit is not applied; only an explicit parent cap applies.')
    return dict(protocol=PROTOCOL, order=list(ORDER), epochs=pair['epochs'],
        smoke=pair['smoke'], source_hashes=source_hashes(), child_plans=plans,
        total_model_jobs=9, total_new_model_epochs=9*pair['epochs'],
        execution='Sequential experiment cores; no child email wrappers',
        runtime_policy='Run all nine jobs to completion by default; standalone wrapper budgets are not used. An explicit parent cap is optional.',
        shutdown_after_completion=True, instance_policy='email then managed stop',
        audit_partition_evaluated=False, official_validation_evaluated=False)


def _real_directory(path):
    path = Path(path).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('Queue paths must not contain symlinks')
    return path.resolve()


def create_queue(args, plan):
    root = _real_directory(args.output)
    queue = root/new_name('smoke' if plan['smoke'] else 'queue')
    queue.mkdir(parents=True, exist_ok=False)
    modules = engines()
    children = []
    for name in ORDER:
        child_args = child_arguments(args, name, queue/name, modules)
        suite = modules[name].create_suite(child_args, plan=copy.deepcopy(plan['child_plans'][name]))
        children.append(dict(name=name, suite=str(Path(suite).resolve()), core=CORES[name]))
    record = json.loads(json.dumps(dict(plan, children=children)))
    _write_json(queue/'queue.json', record)
    print(f'Combined queue: {queue}', flush=True)
    return queue, record


def validate_resume(queue):
    raw = Path(queue)
    if raw.is_symlink() or (raw/'queue.json').is_symlink():
        raise ValueError('Refusing symlink queue manifest')
    queue = _real_directory(raw)
    value = json.loads((queue/'queue.json').read_text(encoding='utf-8'))
    if (value.get('protocol') != PROTOCOL or value.get('order') != list(ORDER)
            or value.get('source_hashes') != source_hashes()
            or value.get('shutdown_after_completion') is not True
            or value.get('instance_policy') != 'email then managed stop'
            or len(value.get('children', [])) != 2
            or not isinstance(value.get('epochs'),int) or not 1 <= value['epochs'] <= 30
            or value.get('total_model_jobs') != 9
            or value.get('total_new_model_epochs') != 9*value['epochs']):
        raise ValueError('Queue identity, source, order or final-stop policy changed')
    modules = engines()
    for name, child in zip(ORDER, value['children']):
        path = _real_directory(child['suite'])
        if (child.get('name') != name or child.get('core') != CORES[name]
                or path.is_symlink() or not path.resolve().is_relative_to(queue/name)):
            raise ValueError('Child suite escaped its own queue directory')
        child_plan = modules[name].validate_resume(path)
        if (json.loads(json.dumps(child_plan)) != value['child_plans'][name]
                or child_plan.get('shutdown_after_completion') is not False
                or child_plan['epochs'] != value['epochs']):
            raise ValueError('Recorded child plan or no-child-shutdown policy changed')
    pair,hcsi = value['child_plans']['pairwise'],value['child_plans']['hcsi']
    if (pair['anchor']['checkpoint_sha256'] != hcsi['reference']['checkpoint_sha256']
            or pair['seed'] != hcsi['seed'] or pair['split_hash'] != hcsi['split_hash']):
        raise ValueError('Resumed studies no longer share the reference, seed and split')
    return value


def record_budget(queue, budget, *, invocation_start_unix, finished):
    elapsed = max(0., time.monotonic()-budget['start_monotonic'])
    spent = budget['prior_spent_seconds']+(elapsed if finished else 0.)
    _write_json(Path(queue)/'wall_budget.json', dict(cap_seconds=budget['total_seconds'],
        max_hours=(budget['total_seconds']/3600. if budget['total_seconds'] is not None else None), elapsed_active_seconds=spent,
        invocation_start_unix=float(invocation_start_unix), invocation_finished=bool(finished),
        delivery_reserve_seconds=budget['delivery_reserve_seconds'],
        scope='One cumulative parent queue: preflight, both studies and email; clean idle gaps excluded.',
        instance_policy='email then managed stop', shutdown_after_completion=True))


def queue_budget(queue, hours, started, *, now_unix=None):
    """Persist elapsed active time without inventing a deadline for uncapped runs."""
    cap, spent = hours*3600. if hours is not None else None, 0.
    if queue is not None:
        path = Path(queue)/'wall_budget.json'
        if path.is_symlink():
            raise ValueError('Refusing symlink wall accounting')
        if path.is_file():
            value = json.loads(path.read_text(encoding='utf-8'))
            old_cap = value['cap_seconds']
            if old_cap is not None:
                old_cap = float(old_cap)
                if not math.isfinite(old_cap) or not 360 <= old_cap <= 36000:
                    raise ValueError('Invalid optional queue cap')
                cap = old_cap if cap is None else min(cap,old_cap)
            spent = float(value['elapsed_active_seconds'])
            if not math.isfinite(spent) or spent < 0:
                raise ValueError('Invalid elapsed active wall time')
            if not value.get('invocation_finished',False):
                now = time.time() if now_unix is None else now_unix
                previous = float(value['invocation_start_unix'])
                if not math.isfinite(previous) or previous > now+5:
                    raise ValueError('Invalid unfinished invocation timestamp')
                spent += max(0.,now-previous)
    if cap is not None:
        return budget_deadlines(started,cap/3600.,spent)
    return dict(start_monotonic=started,total_seconds=None,prior_spent_seconds=spent,
        total_deadline_monotonic=None,child_deadline_monotonic=None,
        remaining_seconds=None,delivery_reserve_seconds=0.)


def child_command(child, deadline_unix=None):
    if child.get('name') not in ORDER or child.get('core') != CORES[child['name']]:
        raise ValueError('Unexpected experiment child')
    command = [sys.executable, '-u', str(ROOT/child['core']), '--resume', child['suite']]
    if deadline_unix is not None:
        command += ['--deadline-unix', str(float(deadline_unix))]
    return command


def child_completion(child):
    path = Path(child['suite'])/'completion.json'
    if not path.is_file() or path.is_symlink():
        return None
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None
    return value.get('status') if isinstance(value, dict) else None


def execution_identity(record):
    return hashlib.sha256(json.dumps(record,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def execution_has_started(queue,record):
    marker=Path(queue)/'execution_started.json'
    if marker.is_symlink():
        raise ValueError('Refusing symlink execution marker')
    if not marker.is_file():
        return False
    value=json.loads(marker.read_text(encoding='utf-8'))
    if value.get('protocol') != PROTOCOL or value.get('queue_identity') != execution_identity(record):
        raise ValueError('Execution marker differs from this immutable queue')
    return True


def mark_execution_started(queue,record):
    if not execution_has_started(queue,record):
        _write_json(Path(queue)/'execution_started.json',dict(protocol=PROTOCOL,
            queue_identity=execution_identity(record),first_child_started_unix=time.time()))


def run_children(queue, record, log, budget, on_started):
    outcomes, timed_out = {}, False
    for child in record['children']:
        name = child['name']
        if child_completion(child) == 'complete':
            outcomes[name] = dict(status='complete', exit_code=0, reused_completed=True)
            continue
        deadline = budget['child_deadline_monotonic']
        if deadline is not None and time.monotonic() >= deadline:
            timed_out = True
            outcomes[name] = dict(status='not_started', reason='Shared queue time budget')
            continue
        try:
            deadline_unix = time.time()+deadline-time.monotonic() if deadline is not None else None
            header = f'\nStarting {name}; no intermediate email or instance stop.\n'
            log.write(header); print(header, end='', flush=True)
            with subprocess.Popen(child_command(child, deadline_unix), cwd=ROOT,
                    env=child_environment(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, errors='replace', bufsize=1,
                    **({'start_new_session': True} if os.name == 'posix' else {})) as process:
                process._resolver_owned_group = os.name == 'posix'
                with (ChildWatchdog(process,deadline) if deadline is not None else nullcontext(None)) as watchdog:
                    try:
                        on_started()
                        for line in process.stdout:
                            log.write(line); print(line, end='', flush=True)
                        code = process.wait()
                    except BaseException:
                        _stop_child(process)
                        raise
                stopped = watchdog.timed_out.is_set() if watchdog is not None else False
            state = child_completion(child)
            limited = stopped or state == 'budget_exhausted'
            timed_out = timed_out or limited
            outcomes[name] = dict(status='complete' if code == 0 and state == 'complete'
                else ('budget_exhausted' if limited else 'failed'),
                exit_code=124 if stopped else code, child_status=state, reused_completed=False)
        except (KeyboardInterrupt, SystemExit):
            outcomes[name] = dict(status='interrupted')
            _write_json(Path(queue)/'status.json', dict(protocol=PROTOCOL,status='interrupted',outcomes=outcomes))
            raise
        except Exception as exc:
            outcomes[name] = dict(status='failed', error_type=type(exc).__name__,
                traceback=traceback.format_exc())
        _write_json(Path(queue)/'status.json', dict(protocol=PROTOCOL,status='running',outcomes=outcomes))
    complete = len(outcomes) == 2 and all(v['status'] == 'complete' for v in outcomes.values())
    return (0 if complete else (124 if timed_out else 1)), outcomes, timed_out


def write_completion(queue, record, code, outcomes, timed_out, failure=None):
    status = 'complete' if code == 0 else ('budget_exhausted' if timed_out else 'failed')
    value = dict(protocol=PROTOCOL,status=status,exit_code=code,outcomes=outcomes,
        failure=failure,shutdown_after_completion=True,instance_policy='email then managed stop',
        total_model_jobs=9,audit_partition_evaluated=False,official_validation_evaluated=False)
    _write_json(Path(queue)/'completion.json', value)
    _write_json(Path(queue)/'status.json', value)
    text = ('# Combined pairwise and HCSI queue\n\n'
        f'Status: **{status}**. Four pairwise arms followed by five HCSI/continuation controls.\n\n'
        'Both study reports are delivered in one ordinary ZIP. Child experiment cores never '
        'send mail or stop the instance. The parent sends the combined completion/partial '
        'email, then requests managed Workbench stop. Incomplete work remains partial.\n\n')
    for child in record['children']:
        relative = Path(child['suite']).relative_to(queue).as_posix()
        text += f'- [{child["name"]}]({relative}/report.md)\n'
    (Path(queue)/'report.md').write_text(text,encoding='utf-8')


def main(argv=None):
    import torch
    from fusion_overnight_delivery import finish
    started, started_unix = time.monotonic(), time.time()
    args = arguments().parse_args(argv)
    budget = queue_budget(args.resume,args.max_hours,started)
    if not args.check_only and budget['remaining_seconds'] is not None and budget['remaining_seconds'] <= 0:
        raise RuntimeError('Combined queue budget is exhausted; no new training started')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no training or shutdown started')
    record = validate_resume(args.resume) if args.resume else make_plan(args)
    if args.resume and (args.smoke or (args.epochs is not None and args.epochs != record['epochs'])):
        raise ValueError('Resume retains the saved child horizons and smoke/full protocol')
    check_email_ready(args.to)
    if args.fallback_os_shutdown:
        subprocess.run(['sudo','-n','-l','/usr/sbin/shutdown','-h','now'],check=True)
    url = preflight()
    if args.check_only:
        print('Combined preflight OK; no suite, training, email or shutdown.',flush=True)
        return
    if budget['child_deadline_monotonic'] is not None and time.monotonic() >= budget['child_deadline_monotonic']:
        raise RuntimeError('Preflight exhausted compute budget; no training or shutdown started')
    if args.resume:
        queue = Path(args.resume).resolve()
    else:
        queue, record = create_queue(args,record)
    record_budget(queue,budget,invocation_start_unix=started_unix,finished=False)
    log_path = queue/'run.log'
    armed = execution_has_started(queue,record) if args.resume else False
    code, delivered, timed_out = 1, False, False
    outcomes, failure = {}, None
    def on_started():
        nonlocal armed
        armed = True
        mark_execution_started(queue,record)
    def before_stop():
        record_budget(queue,budget,invocation_start_unix=started_unix,finished=True)
    try:
        with log_path.open('a',encoding='utf-8',buffering=1) as log:
            try:
                code,outcomes,timed_out = run_children(queue,record,log,budget,on_started)
            except BaseException as exc:
                code = 130 if isinstance(exc,KeyboardInterrupt) else 1
                failure = dict(error_type=type(exc).__name__,traceback=traceback.format_exc())
                marker = queue/'status.json'
                if marker.is_file() and not marker.is_symlink():
                    try:
                        outcomes = json.loads(marker.read_text(encoding='utf-8')).get('outcomes',outcomes)
                    except (OSError,ValueError,AttributeError):
                        pass
                log.write(failure['traceback'])
            finally:
                log.flush(); os.fsync(log.fileno())
    finally:
        try:
            write_completion(queue,record,code,outcomes,timed_out,failure)
        finally:
            if armed:
                delivered = finish(queue,log_path,args.to,code,url,args.fallback_os_shutdown,
                    timed_out=timed_out,budget=budget if budget['total_seconds'] is not None else None,
                    before_stop=before_stop)
            else:
                before_stop()
    if code == 0 and not delivered:
        code = 2
    raise SystemExit(code)


if __name__ == '__main__':
    main()
