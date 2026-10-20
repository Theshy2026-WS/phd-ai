"""Train, package small analysis files, optionally export, then stop Workbench.

Run in tmux. This does not send email. Shutdown is armed only after preflight.
Archives exclude datasets and checkpoints. No download to an offline laptop is
possible; --export-to can preserve an archive in a user-specified GCS location.
"""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import traceback

from overnight_twostage import preflight, stop_workbench

ROOT = Path(__file__).resolve().parent


def bundle_results(suite, log_path):
    suite = Path(suite).resolve()
    destination = suite.parent/(suite.name+'_analysis.tar.gz')
    temporary = destination.with_suffix('.tmp')
    with tarfile.open(temporary, 'w:gz') as archive:
        for path in sorted(suite.rglob('*')):
            if (path.is_file() and not path.is_symlink() and
                    path.suffix in ('.csv', '.json', '.yaml', '.txt')):
                archive.add(path, arcname=str(Path(suite.name)/path.relative_to(suite)))
        if Path(log_path).is_file():
            archive.add(log_path, arcname=f'{suite.name}/training.log')
        for name in ('models/attention_correction.py', 'models/vit.py',
                     'correction_experiment.py', 'literature_experiment.py',
                     'overnight_correction.py', 'ATTENTION_CORRECTION.md'):
            archive.add(ROOT/name, arcname=f'{suite.name}/source/{name}')
    temporary.replace(destination)
    return destination


def finish(suite, log_path, url, fallback, export_to=None):
    # Packaging/export failure must NOT leave the GPU running overnight.
    try:
        archive = bundle_results(suite, log_path)
        print(f'Analysis archive: {archive}', flush=True)
        if export_to:
            if not export_to.startswith('gs://'):
                raise ValueError('--export-to must be a private gs://bucket/prefix')
            subprocess.run(['gcloud', 'storage', 'cp', str(archive),
                            export_to.rstrip('/')+'/'+archive.name],
                           check=True, timeout=180)
        if hasattr(os, 'sync'):
            os.sync()
    finally:
        stop_workbench(url, fallback)


def main():
    import torch
    from correction_experiment import arguments, create_suite
    parser = arguments()
    parser.description = __doc__
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--fallback-os-shutdown', action='store_true')
    parser.add_argument('--export-to', help='Optional existing private gs://bucket/prefix; not created automatically')
    args = parser.parse_args()
    if min(args.correction_epochs, args.vit_epochs) < 1:
        parser.error('Epochs must be positive')
    if args.export_to and not args.export_to.startswith('gs://'):
        parser.error('--export-to must start with gs://')
    if args.export_to and shutil.which('gcloud') is None:
        parser.error('--export-to requires gcloud installed before training')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no training or shutdown started')
    # Validate parent/checkpoint before arming shutdown. No new suite on check-only.
    if args.check_only:
        from correction_experiment import make_plan
        if not args.resume:
            plan = make_plan(args)
            print(f'Parent verified: {plan["base_checkpoint"]}', flush=True)
    if args.fallback_os_shutdown:
        subprocess.run(['sudo', '-n', '-l', '/usr/sbin/shutdown', '-h', 'now'], check=True)
    url = preflight()
    if args.check_only:
        print('Checks passed. No training or shutdown performed.', flush=True)
        return
    suite = Path(args.resume).resolve() if args.resume else create_suite(args)
    print(f'Suite: {suite}', flush=True)
    log_path = suite/'overnight.log'
    # Separate child makes a durable tee log without global stdout replacement.
    code = 1
    with log_path.open('a', encoding='utf-8', buffering=1) as log:
        try:
            command = [sys.executable, '-u', str(ROOT/'correction_experiment.py'),
                       '--resume', str(suite)]
            with subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
                for line in process.stdout:
                    log.write(line)
                    print(line, end='', flush=True)
                code = process.wait()
        except BaseException:
            log.write(traceback.format_exc())
            raise
        finally:
            log.write(f'\nExit code: {code}\n')
            log.flush()
            os.fsync(log.fileno())
            finish(suite, log_path, url, args.fallback_os_shutdown, args.export_to)
    raise SystemExit(code)


if __name__ == '__main__':
    main()
