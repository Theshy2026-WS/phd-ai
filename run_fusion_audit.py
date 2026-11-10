"""Evaluate locked CNN/ViT checkpoints on official validation, email, then stop.

This entry point performs no training or model selection. Email authentication
and managed Workbench stop permissions are checked before audit registration.
``--check-only`` creates no suite, evaluates nothing and sends/stops nothing.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from correction_experiment import file_hash
from overnight_twostage import preflight
from run_hcsi_tiny import _stop_child, _write_json
from run_vit_augreg_overnight import check_email_ready, child_environment


ROOT = Path(__file__).resolve().parent
PROTOCOL = 'tiny-confirmatory-fusion-audit-runner-v1'
CORE = 'fusion_confirmatory_audit.py'


def arguments():
    from fusion_confirmatory_audit import arguments as audit_arguments
    parser = audit_arguments()
    parser.description = __doc__
    parser.add_argument('--to', default='shenwenfei@gmail.com')
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--fallback-os-shutdown', action='store_true',
        help='Optional OS fallback only if managed Workbench stop fails')
    return parser


def source_hashes():
    names = ('run_fusion_audit.py', 'fusion_audit_delivery.py',
        'run_vit_augreg_overnight.py', 'run_hcsi_tiny.py', 'overnight_twostage.py')
    return {name: file_hash(ROOT/name) for name in names}


def runner_record(suite):
    marker = Path(suite)/'suite.json'
    if marker.is_symlink() or not marker.is_file():
        raise ValueError('Need the immutable predeclared audit manifest')
    return dict(protocol=PROTOCOL, core=CORE, source_hashes=source_hashes(),
        suite_identity=file_hash(marker), evaluation_only=True, training_epochs=0,
        target_partition='official_validation', legacy_internal_audit_previously_evaluated=True,
        shutdown_after_completion=True, instance_policy='email then managed stop')


def validate_runner(suite):
    suite = Path(suite)
    if any(path.is_symlink() for path in (suite, *suite.parents)):
        raise ValueError('Audit suite paths must not contain symlinks')
    marker = suite/'runner.json'
    if marker.is_symlink() or not marker.is_file():
        raise ValueError('Resume requires the original audit runner identity')
    record = json.loads(marker.read_text(encoding='utf-8'))
    if record != runner_record(suite):
        raise ValueError('Audit runner source, manifest or email/stop policy changed')
    return record


def _record_digest(record):
    return hashlib.sha256(json.dumps(record, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def execution_has_started(suite, record):
    marker = Path(suite)/'execution_started.json'
    if marker.is_symlink():
        raise ValueError('Refusing symlink audit execution marker')
    if not marker.exists():
        return False
    value = json.loads(marker.read_text(encoding='utf-8'))
    if value.get('protocol') != PROTOCOL or value.get('runner_identity') != _record_digest(record):
        raise ValueError('Audit execution marker belongs to a different runner')
    return True


def mark_execution_started(suite, record):
    if not execution_has_started(suite, record):
        _write_json(Path(suite)/'execution_started.json', dict(protocol=PROTOCOL,
            runner_identity=_record_digest(record), first_child_started_unix=time.time()))


def child_command(suite):
    return [sys.executable, '-u', str(ROOT/CORE), '--resume', str(suite)]


def main(argv=None):
    import torch
    from fusion_confirmatory_audit import make_plan, create_suite, validate_resume
    from fusion_audit_delivery import finish
    args = arguments().parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no audit, email or shutdown started')
    plan = validate_resume(args.resume) if args.resume else make_plan(args)
    record = validate_runner(args.resume) if args.resume else None
    check_email_ready(args.to)
    if args.fallback_os_shutdown:
        subprocess.run(['sudo', '-n', '-l', '/usr/sbin/shutdown', '-h', 'now'], check=True)
    url = preflight()
    if args.check_only:
        print('Audit preflight OK; no registration, inference, training, email or shutdown.', flush=True)
        return
    if args.resume:
        suite = Path(args.resume).resolve()
    else:
        suite = Path(create_suite(args, plan=plan)).resolve()
        record = runner_record(suite)
        _write_json(suite/'runner.json', record)
    log_path = suite/'run.log'
    armed = execution_has_started(suite, record) if args.resume else False
    code, delivered, failure = 1, False, None
    try:
        with log_path.open('a', encoding='utf-8', buffering=1) as log:
            try:
                print(f'Confirmatory audit suite: {suite}', flush=True)
                print('FIXED OFFICIAL VALIDATION ONLY: locked checkpoints and comparisons; zero training epochs.', flush=True)
                print('One final report email, then managed Workbench stop.', flush=True)
                with subprocess.Popen(child_command(suite), cwd=ROOT,
                        env=child_environment(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        text=True, errors='replace', bufsize=1,
                        **({'start_new_session': True} if os.name == 'posix' else {})) as child:
                    child._resolver_owned_group = os.name == 'posix'
                    armed = True
                    try:
                        mark_execution_started(suite, record)
                        for line in child.stdout:
                            log.write(line); print(line, end='', flush=True)
                        code = child.wait()
                    except BaseException:
                        _stop_child(child)
                        raise
            except BaseException as exc:
                code = 130 if isinstance(exc, KeyboardInterrupt) else 1
                failure = dict(error_type=type(exc).__name__, traceback=traceback.format_exc())
                log.write(failure['traceback'])
            finally:
                log.write(f'\nAudit child exit code: {code}\n')
                log.flush(); os.fsync(log.fileno())
    finally:
        try:
            _write_json(suite/'runner_status.json', dict(protocol=PROTOCOL,
                child_exit_code=code, failure=failure, execution_started=armed,
                evaluation_only=True, training_epochs=0,
                instance_policy='email then managed stop', shutdown_after_completion=True))
        finally:
            if armed:
                delivered = finish(suite, log_path, args.to, code, url, args.fallback_os_shutdown)
    if code == 0 and not delivered:
        code = 2
    raise SystemExit(code)


if __name__ == '__main__':
    main()
