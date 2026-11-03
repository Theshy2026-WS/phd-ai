"""Run the spatial CNN pair in tmux, email analysis, then stop this Workbench.

Preflight failure and --check-only never arm shutdown. A completed or failed
child run triggers notification attempts followed by managed instance stop.
"""
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time
import traceback

from overnight_twostage import preflight, stop_workbench
from run_notified import load_email_credentials, send_mail


ROOT = Path(__file__).resolve().parent
ATTACHMENT_LIMIT = 18 * 1024 * 1024
ALLOWED_SUFFIXES = {'.csv', '.json', '.yaml', '.yml', '.md', '.png', '.txt', '.log'}
BLOCKED_DIRECTORIES = {'data', 'datasets', 'checkpoints', '.git', '.venv', 'venv'}
SOURCE_FILES = ('run_spatial_overnight.py', 'spatial_experiment.py',
                'models/spatial_preact.py', 'baseline_diagnostics.py',
                'utils/baseline_diagnostics.py', 'datasets/resnet_tiny.py',
                'datasets/utility_tiny.py', 'utils/literature_training.py',
                'SPATIAL_EXPERIMENTS.md')


def _secret_name(name):
    name = name.lower()
    return name.startswith(('email.txt', '.env'))


def _regular_inside(path, root):
    """Reject symlinks at every level, not just the final file."""
    path, root = Path(path), Path(root)
    if path.is_symlink() or root.is_symlink():
        return False
    try:
        relative = path.relative_to(root)
    except ValueError:
        return False
    ancestor = root
    for part in relative.parts:
        ancestor = ancestor/part
        if ancestor.is_symlink() or _secret_name(part):
            return False
    return path.is_file() and path.resolve().is_relative_to(root.resolve())


def _analysis_files(suite, compact=False):
    files, omitted = [], []
    for directory, subdirectories, names in os.walk(suite, followlinks=False):
        current = Path(directory)
        subdirectories[:] = [name for name in sorted(subdirectories)
            if name.lower() not in BLOCKED_DIRECTORIES and not _secret_name(name)
            and not (current/name).is_symlink()]
        for name in sorted(names):
            path = current/name
            if path.suffix.lower() not in ALLOWED_SUFFIXES or not _regular_inside(path, suite):
                continue
            relative = path.relative_to(suite)
            bulky = (name == 'predictions.csv'
                     or name.startswith('paired_selection') and path.suffix.lower() == '.csv'
                     or 'manifest' in name.lower() or name.endswith('_ids.json'))
            if compact and bulky:
                omitted.append(relative.as_posix())
            else:
                files.append((path, relative))
    return files, omitted


def _write_archive(suite, destination, compact):
    files, omitted = _analysis_files(suite, compact=compact)
    temporary = destination.with_suffix('.tmp')
    with tarfile.open(temporary, 'w:gz') as archive:
        for path, relative in files:
            archive.add(path, arcname=(Path(suite.name)/relative).as_posix(), recursive=False)
        for name in SOURCE_FILES:
            source = ROOT/name
            if _regular_inside(source, ROOT):
                archive.add(source, arcname=(Path(suite.name)/'source'/name).as_posix(), recursive=False)
        manifest = json.dumps(dict(compact=compact, omitted_analysis_files=omitted,
            excluded_types=['model checkpoints', 'full probability NPZ', 'nested archives',
                            'datasets', 'credentials', 'symlinks'],
            note='The full analysis archive and all original outputs remain on the VM.'),
            indent=2).encode('utf-8')
        info = tarfile.TarInfo(f'{suite.name}/analysis_bundle.json')
        info.size = len(manifest)
        archive.addfile(info, io.BytesIO(manifest))
    temporary.replace(destination)
    return omitted


def bundle_results(suite):
    suite = Path(suite).absolute()
    if not suite.is_dir() or suite.is_symlink() or any(p.is_symlink() for p in suite.parents):
        raise ValueError('Analysis suite must be a real directory without symlink ancestors')
    full = suite.parent/(suite.name+'_analysis.tar.gz')
    _write_archive(suite, full, compact=False)
    attachment, omitted, compact = full, [], None
    if full.stat().st_size > ATTACHMENT_LIMIT:
        compact = suite.parent/(suite.name+'_analysis_compact.tar.gz')
        omitted = _write_archive(suite, compact, compact=True)
        attachment = compact if compact.stat().st_size <= ATTACHMENT_LIMIT else None
    return dict(full=full, compact=compact, attachment=attachment, omitted=omitted)


def finish(suite, log_path, url, fallback, recipient, code):
    """Notification errors must not leave a completed GPU job running."""
    status = 'SUCCESS' if code == 0 else 'FAILED / INTERRUPTED'
    subject = f'[Tiny ImageNet spatial] {status}'
    body = (f'Status: {status}\nExit code: {code}\nSuite: {suite}\nLog: {log_path}\n'
            'This run uses fresh CNN training, not an extension of the old 100-epoch checkpoint.\n'
            'Managed Workbench stop will be requested after notification attempts.\n'
            'Outputs, probability arrays, and checkpoints remain on the instance disk.\n')
    attachment = None
    try:
        try:
            bundle = bundle_results(suite)
            attachment = bundle['attachment']
            body += f'Full analysis archive: {bundle["full"]}\n'
            print(f'Analysis archive: {bundle["full"]}', flush=True)
            if bundle['compact'] is not None:
                body += (f'Compact archive: {bundle["compact"]}\n'
                         'The email-sized version omits per-image prediction/pair CSVs and '
                         'large image manifests; summaries, curves, class diagnostics, and galleries remain.\n')
            if attachment is None:
                subject += ' / ATTACHMENT TOO LARGE'
                body += 'Both analysis archives exceed the attachment limit; retrieve them after restarting the VM.\n'
        except Exception as exc:
            subject += ' / ANALYSIS PACKAGE FAILED'
            body += f'Packaging failed ({type(exc).__name__}); original files remain on disk.\n'
            print(f'Packaging failed ({type(exc).__name__}).', flush=True)
        sent = False
        for attempt in range(3):
            try:
                send_mail(recipient, subject, body, attachment=attachment)
                sent = True
                print('Completion email accepted by Gmail; check inbox/spam after reconnecting.', flush=True)
                break
            except Exception as exc:
                print(f'Email attempt {attempt+1} failed ({type(exc).__name__}).', flush=True)
                if attempt < 2:
                    time.sleep(5)
        if not sent:
            print('Notification failed; retrieve results from the instance disk after restart.', flush=True)
        return sent
    finally:
        try:
            if hasattr(os, 'sync'):
                os.sync()
        finally:
            stop_workbench(url, fallback)


def child_environment():
    return {key: value for key, value in os.environ.items()
            if key not in ('GMAIL_USER', 'GMAIL_APP_PASSWORD')}


def child_command(suite):
    return [sys.executable, '-u', str(ROOT/'spatial_experiment.py'), '--resume', str(suite)]


def arguments():
    from spatial_experiment import arguments as training_arguments
    parser = training_arguments()
    parser.description = __doc__
    parser.add_argument('--to', default='shenwenfei@gmail.com')
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--fallback-os-shutdown', action='store_true')
    return parser


def main(argv=None):
    import torch
    from spatial_experiment import make_plan, create_suite, validate_resume
    args = arguments().parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no training or shutdown started')
    load_email_credentials()
    if any(not os.environ.get(key, '').strip() for key in ('GMAIL_USER', 'GMAIL_APP_PASSWORD')):
        raise RuntimeError('Missing Gmail credentials; no training or shutdown started')
    plan = validate_resume(args.resume) if args.resume else make_plan(args)
    if args.resume and (args.smoke or args.epochs is not None and
                        args.epochs != plan['config']['training']['epochs']):
        raise ValueError('Resume keeps the saved smoke/full protocol and total horizon')
    if args.fallback_os_shutdown:
        subprocess.run(['sudo', '-n', '-l', '/usr/sbin/shutdown', '-h', 'now'], check=True)
    url = preflight()
    send_mail(args.to, '[Tiny ImageNet spatial] Preflight OK',
              'GPU, checkpoint/data protocol, notification and managed-stop permission checked. '
              + ('Check only: no training, new suite, or shutdown.' if args.check_only else
                 'The selected spatial suite will run; analysis notification precedes the stop request.'))
    if args.check_only:
        print('Preflight OK; no training, new suite, or shutdown performed.', flush=True)
        return
    suite = Path(args.resume).resolve() if args.resume else create_suite(args, plan=plan)
    log_path = suite/'overnight.log'
    armed, code = False, 1
    try:
        # Opening the log can fail before shutdown is armed.
        with log_path.open('a', encoding='utf-8', buffering=1) as log:
            try:
                print(f'Suite: {suite}', flush=True)
                with subprocess.Popen(child_command(suite), cwd=ROOT, env=child_environment(),
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        text=True, errors='replace', bufsize=1) as child:
                    armed = True
                    try:
                        for line in child.stdout:
                            log.write(line)
                            print(line, end='', flush=True)
                        code = child.wait()
                    except BaseException:
                        child.terminate()
                        try:
                            child.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            child.kill()
                            child.wait()
                        raise
            except BaseException:
                log.write(traceback.format_exc())
                raise
            finally:
                log.write(f'\nChild exit code: {code}\n')
                log.flush()
                os.fsync(log.fileno())
    finally:
        if armed:
            finish(suite, log_path, url, args.fallback_os_shutdown, args.to, code)
    raise SystemExit(code)


if __name__ == '__main__':
    main()
