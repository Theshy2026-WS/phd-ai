"""Run three targeted fusion controls, email all analysis, then stop Workbench.

No total time cutoff applies by default. Email authentication, source/data
validation and managed stop permissions are checked before a suite is created.
Only a started child arms final email/stop. ``--check-only`` sends no message,
creates no suite, starts no training and does not stop the instance.
"""
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
from overnight_twostage import preflight
from run_hcsi_tiny import ChildWatchdog, _stop_child, _write_json
from run_vit_augreg_overnight import check_email_ready, child_environment


ROOT = Path(__file__).resolve().parent
PROTOCOL = 'tiny-fusion-mechanism-runner-v1'
CORE = 'fusion_mechanism_experiment.py'


def arguments():
    from fusion_mechanism_experiment import arguments as training_arguments
    parser = training_arguments()
    parser.description = __doc__
    parser.add_argument('--to', default='shenwenfei@gmail.com')
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--fallback-os-shutdown', action='store_true',
        help='Optional OS fallback only if the managed Workbench stop fails')
    return parser


def source_hashes():
    names = ('run_fusion_mechanism.py', 'fusion_mechanism_delivery.py',
        'run_vit_augreg_overnight.py', 'run_hcsi_tiny.py', 'overnight_twostage.py')
    return {name: file_hash(ROOT/name) for name in names}


def _suite_identity(suite):
    path = Path(suite)/'suite.json'
    if not path.is_file() or path.is_symlink():
        raise ValueError('Need the immutable experiment suite manifest')
    return file_hash(path)


def runner_record(suite):
    return dict(protocol=PROTOCOL, core=CORE, source_hashes=source_hashes(),
        suite_identity=_suite_identity(suite), shutdown_after_completion=True,
        instance_policy='email then managed stop', default_total_hours=None,
        audit_partition_evaluated=False, official_validation_evaluated=False)


def validate_runner(suite):
    suite = Path(suite)
    if any(path.is_symlink() for path in (suite, *suite.parents)):
        raise ValueError('Runner suite paths must not contain symlinks')
    marker = suite/'runner.json'
    if marker.is_symlink() or not marker.is_file():
        raise ValueError('Resume needs the original runner identity')
    record = json.loads(marker.read_text(encoding='utf-8'))
    if record != runner_record(suite):
        raise ValueError('Runner source, suite identity or email/stop policy changed')
    return record


def _record_digest(record):
    return hashlib.sha256(json.dumps(record, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def execution_has_started(suite, record):
    marker = Path(suite)/'execution_started.json'
    if marker.is_symlink():
        raise ValueError('Refusing a symlink execution marker')
    if not marker.exists():
        return False
    value = json.loads(marker.read_text(encoding='utf-8'))
    if value.get('protocol') != PROTOCOL or value.get('runner_identity') != _record_digest(record):
        raise ValueError('Execution marker belongs to a different runner')
    return True


def mark_execution_started(suite, record):
    if not execution_has_started(suite, record):
        _write_json(Path(suite)/'execution_started.json', dict(protocol=PROTOCOL,
            runner_identity=_record_digest(record), first_child_started_unix=time.time()))


def child_command(suite, deadline_unix=None):
    command = [sys.executable, '-u', str(ROOT/CORE), '--resume', str(suite)]
    if deadline_unix is not None:
        command += ['--deadline-unix', str(float(deadline_unix))]
    return command


def main(argv=None):
    import torch
    from fusion_mechanism_experiment import make_plan, create_suite, validate_resume
    from fusion_mechanism_delivery import finish
    args = arguments().parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no training, email or shutdown started')
    plan = validate_resume(args.resume) if args.resume else make_plan(args)
    if args.resume:
        if args.smoke or (args.epochs is not None and args.epochs != plan['epochs']):
            raise ValueError('Resume retains the saved full/smoke protocol and horizon')
        record = validate_runner(args.resume)
    else:
        record = None
    deadline_unix = getattr(args, 'deadline_unix', None)
    if deadline_unix is not None:
        if not math.isfinite(deadline_unix) or deadline_unix <= 0:
            raise ValueError('Explicit deadline must be a finite positive UNIX timestamp')
        if time.time() >= deadline_unix:
            raise RuntimeError('Explicit deadline has expired; no training or shutdown started')
    check_email_ready(args.to)
    if args.fallback_os_shutdown:
        subprocess.run(['sudo', '-n', '-l', '/usr/sbin/shutdown', '-h', 'now'], check=True)
    url = preflight()
    if args.check_only:
        print('Fusion mechanism preflight OK; no suite, training, email or shutdown.', flush=True)
        return
    if deadline_unix is not None and time.time() >= deadline_unix:
        raise RuntimeError('Preflight exhausted the explicit deadline; no training or shutdown started')
    if args.resume:
        suite = Path(args.resume).resolve()
    else:
        suite = Path(create_suite(args, plan=plan)).resolve()
        record = runner_record(suite)
        _write_json(suite/'runner.json', record)
    log_path = suite/'run.log'
    armed = execution_has_started(suite, record) if args.resume else False
    code, delivered = 1, False
    failure = None
    try:
        with log_path.open('a', encoding='utf-8', buffering=1) as log:
            try:
                print(f'Fusion mechanism suite: {suite}', flush=True)
                print('All three controls run to completion by default; one final email then managed stop.', flush=True)
                deadline = time.monotonic()+deadline_unix-time.time() if deadline_unix is not None else None
                with subprocess.Popen(child_command(suite, deadline_unix), cwd=ROOT,
                        env=child_environment(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        text=True, errors='replace', bufsize=1,
                        **({'start_new_session': True} if os.name == 'posix' else {})) as child:
                    child._resolver_owned_group = os.name == 'posix'
                    armed = True
                    try:
                        mark_execution_started(suite, record)
                        with (ChildWatchdog(child, deadline) if deadline is not None else nullcontext(None)) as watchdog:
                            for line in child.stdout:
                                log.write(line); print(line, end='', flush=True)
                            code = child.wait()
                        if watchdog is not None and watchdog.timed_out.is_set():
                            code = 124
                    except BaseException:
                        _stop_child(child)
                        raise
            except BaseException as exc:
                code = 130 if isinstance(exc, KeyboardInterrupt) else 1
                failure = dict(error_type=type(exc).__name__, traceback=traceback.format_exc())
                log.write(failure['traceback'])
            finally:
                log.write(f'\nExperiment child exit code: {code}\n')
                log.flush(); os.fsync(log.fileno())
    finally:
        try:
            _write_json(suite/'runner_status.json', dict(protocol=PROTOCOL,
                child_exit_code=code, failure=failure, execution_started=armed,
                instance_policy='email then managed stop', shutdown_after_completion=True))
        finally:
            if armed:
                delivered = finish(suite, log_path, args.to, code, url, args.fallback_os_shutdown)
    if code == 0 and not delivered:
        code = 2
    raise SystemExit(code)


if __name__ == '__main__':
    main()
