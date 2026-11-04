"""Run the bounded Tiny ImageNet generalization study, email once, then stop Workbench.

No email is sent during preflight.  SMTP authentication, data/checkpoint checks,
and managed-stop permissions are checked before a child can arm shutdown.
After that child completes or fails, one completion email contains the three
analysis materials, followed by the instance stop request.
"""
import io
import json
import os
from pathlib import Path
import smtplib
import ssl
import subprocess
import sys
import tarfile
import time
import traceback

from overnight_twostage import preflight, stop_workbench
from run_notified import load_email_credentials, send_mail


ROOT = Path(__file__).resolve().parent
ATTACHMENT_LIMIT = 18 * 1024 * 1024
ALLOWED_SUFFIXES = {'.csv', '.json', '.yaml', '.yml', '.md', '.png', '.pdf', '.txt', '.log'}
BLOCKED_DIRECTORIES = {'data', 'datasets', 'checkpoints', '.git', '.venv', 'venv'}
COMPONENTS = ('data_audit', 'previous_last', 'training')
SOURCE_FILES = (
    'run_generalization_overnight.py', 'generalization_suite.py',
    'generalization_experiment.py', 'generalization_audit.py',
    'generalization_last.py',
    'models/generalization_preact.py', 'models/spatial_preact.py',
    'models/spatial_fusion.py', 'fusion30_experiment.py',
    'mechanism_diagnostics.py', 'baseline_diagnostics.py',
    'utils/baseline_diagnostics.py', 'datasets/resnet_tiny.py',
    'datasets/utility_tiny.py', 'utils/literature_training.py',
    'GENERALIZATION_OVERNIGHT.md',
)


def _secret_name(name):
    return name.lower().startswith(('email.txt', '.env'))


def _regular_inside(path, root):
    path, root = Path(path), Path(root)
    if path.is_symlink() or root.is_symlink():
        return False
    try:
        relative = path.relative_to(root)
    except ValueError:
        return False
    ancestor = root
    for part in relative.parts:
        ancestor = ancestor / part
        if ancestor.is_symlink() or _secret_name(part):
            return False
    return path.is_file() and path.resolve().is_relative_to(root.resolve())


def _analysis_files(directory, compact=0):
    """Return whitelisted real files; compact1 drops per-image data, 2 drops PNGs."""
    files, omitted = [], []
    if not directory.is_dir() or directory.is_symlink():
        return files, omitted
    for current_name, subdirectories, names in os.walk(directory, followlinks=False):
        current = Path(current_name)
        subdirectories[:] = [name for name in sorted(subdirectories)
            if name.lower() not in BLOCKED_DIRECTORIES and not _secret_name(name)
            and not (current / name).is_symlink()]
        for name in sorted(names):
            path = current / name
            if path.suffix.lower() not in ALLOWED_SUFFIXES or not _regular_inside(path, directory):
                continue
            relative = path.relative_to(directory)
            lower = name.lower()
            bulky = ((path.suffix.lower() == '.csv' and
                      ('prediction' in lower or lower.startswith('paired_')))
                     or 'manifest' in lower or lower.endswith('_ids.json')
                     or lower in {'image_inventory.csv', 'image_hashes.csv', 'image_metadata.csv'})
            if compact and bulky or compact >= 2 and path.suffix.lower() == '.png':
                omitted.append(relative.as_posix())
            else:
                files.append((path, relative))
    return files, omitted


def _add_json(archive, name, value):
    content = json.dumps(value, indent=2, ensure_ascii=False).encode('utf-8')
    member = tarfile.TarInfo(name)
    member.size = len(content)
    archive.addfile(member, io.BytesIO(content))


def _temporary_archive(destination):
    temporary = destination.with_suffix(destination.suffix + '.tmp')
    if destination.is_symlink() or temporary.is_symlink():
        raise ValueError('Refusing a symlink analysis archive destination')
    return temporary


def _component_archive(suite, component, destination, compact):
    directory = suite / component
    if directory.is_symlink():
        raise ValueError('Analysis component must not be a symlink')
    files, omitted = _analysis_files(directory, compact)
    temporary = _temporary_archive(destination)
    with tarfile.open(temporary, 'w:gz') as archive:
        for path, relative in files:
            archive.add(path, arcname=f'{component}_analysis/{relative.as_posix()}', recursive=False)
        _add_json(archive, f'{component}_analysis/analysis_bundle.json', dict(
            component=component, compact_level=compact,
            component_directory_exists=directory.is_dir(),
            omitted_analysis_files=omitted,
            excluded_types=['checkpoints', 'probability NPZ', 'datasets',
                            'credentials', 'symlinks', 'unrequested nested archives'],
            note='An interrupted or failed study can contain partial results. '
                 'Use the completion status and component report before interpreting results. '
                 'All original outputs remain on the instance disk.'))
    temporary.replace(destination)
    return omitted


def _envelope(suite, components, destination, compact, omissions):
    """Only the three archives produced above may be nested in the email envelope."""
    temporary = _temporary_archive(destination)
    with tarfile.open(temporary, 'w:gz') as archive:
        for component, path in components.items():
            if path.is_symlink() or not path.is_file():
                raise ValueError('Expected a regular generated component archive')
            archive.add(path, arcname=f'{component}_analysis.tar.gz', recursive=False)
        for path in sorted(suite.iterdir()):
            if (path.suffix.lower() in ALLOWED_SUFFIXES and _regular_inside(path, suite)
                    and not ('manifest' in path.name.lower() or path.name.endswith('_ids.json'))):
                archive.add(path, arcname=f'suite/{path.name}', recursive=False)
        for name in SOURCE_FILES:
            path = ROOT / name
            if _regular_inside(path, ROOT):
                archive.add(path, arcname=f'source/{name}', recursive=False)
        _add_json(archive, 'delivery_manifest.json', dict(
            compact_level=compact, materials=list(components), omitted=omissions,
            explanation='One completion email contains three independently extractable reports: '
                        'data_audit_analysis.tar.gz, previous_last_analysis.tar.gz, and training_analysis.tar.gz. '
                        'Budget-limited jobs are explicitly incomplete. Checkpoints and full probabilities stay on the VM.'))
    temporary.replace(destination)


def bundle_results(suite):
    suite = Path(suite).absolute()
    if not suite.is_dir() or suite.is_symlink() or any(p.is_symlink() for p in suite.parents):
        raise ValueError('Analysis suite must be a real directory without symlink ancestors')
    versions, omissions = {}, {}
    for compact, suffix in ((0, ''), (1, '_compact'), (2, '_compact_nogalleries')):
        components = {}
        omitted = {}
        for name in COMPONENTS:
            destination = suite.parent / f'{suite.name}_{name}_analysis{suffix}.tar.gz'
            omitted[name] = _component_archive(suite, name, destination, compact)
            components[name] = destination
        envelope = suite.parent / f'{suite.name}_delivery{suffix}.tar.gz'
        _envelope(suite, components, envelope, compact, omitted)
        versions[compact] = dict(envelope=envelope, components=components)
        omissions[compact] = omitted
        if envelope.stat().st_size <= ATTACHMENT_LIMIT:
            return dict(full=versions[0]['envelope'], components=versions[0]['components'],
                compact=envelope if compact else None, attachment=envelope,
                compact_level=compact, omitted=omitted)
    return dict(full=versions[0]['envelope'], components=versions[0]['components'],
        compact=versions[2]['envelope'], attachment=None, compact_level=2,
        omitted=omissions[2])


def check_email_ready(recipient):
    """Authenticate without transmitting a preflight or starting email."""
    if not recipient.strip() or any(c in recipient for c in '\r\n'):
        raise ValueError('Need a valid recipient without newlines')
    load_email_credentials()
    user = os.environ.get('GMAIL_USER', '').strip()
    password = os.environ.get('GMAIL_APP_PASSWORD', '').replace(' ', '')
    if not user or not password:
        raise RuntimeError('Missing Gmail credentials; no training or shutdown started')
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465, timeout=30,
                              context=ssl.create_default_context()) as smtp:
            smtp.login(user, password)
    except Exception as exc:
        raise RuntimeError(f'Email authentication failed ({type(exc).__name__}); '
                           'no training or shutdown started') from None
    print('Email authentication checked; no email sent.', flush=True)


def _record_delivery(suite, value):
    try:
        path = Path(suite) / 'delivery_status.json'
        if path.is_symlink():
            return
        temporary = path.with_suffix('.tmp')
        if temporary.is_symlink():
            return
        temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
        temporary.replace(path)
    except Exception as exc:
        print(f'Could not record delivery status ({type(exc).__name__}).', flush=True)


def _has_budget_limit(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key in ('budget_exhausted', 'time_limit_reached') and item is True:
                return True
            if key in ('status', 'reason', 'stop_reason') and isinstance(item, str):
                normalized = item.lower().replace('-', '_').replace(' ', '_')
                if 'budget' in normalized or 'time_limit' in normalized:
                    return True
            if key == 'deferred_jobs' and item:
                return True
            if isinstance(item, (dict, list)) and _has_budget_limit(item):
                return True
    if isinstance(value, list):
        return any(_has_budget_limit(item) for item in value)
    return False


def completion_status(suite, code):
    if code != 0:
        return 'FAILED / INTERRUPTED'
    for name in ('status.json', 'completion.json'):
        path = Path(suite) / name
        if path.is_file() and not path.is_symlink():
            try:
                if _has_budget_limit(json.loads(path.read_text(encoding='utf-8'))):
                    return 'BUDGET LIMIT / PARTIAL'
            except (OSError, ValueError):
                pass
    return 'SUCCESS'


def finish(suite, log_path, url, fallback, recipient, code):
    """Always request stop after packaging and the one completion notification."""
    status = completion_status(suite, code)
    subject = f'[Tiny ImageNet generalization] {status}'
    body = (f'Status: {status}\nExit code: {code}\nSuite: {suite}\nLog: {log_path}\n'
            'The attachment contains three analysis materials: data integrity/distribution audit, '
            'the previous actual last-checkpoint diagnosis, and bounded generalization training comparisons.\n'
            'Read the saved plan and per-job completion status before comparing results. '
            'A budget limit preserves completed work but does not mean every scheduled epoch ran.\n'
            'Managed Workbench stop will be requested after notification attempts.\n'
            'All checkpoints, probabilities and full analysis files remain on the instance disk.\n')
    attachment, sent, attempts, package_error = None, False, 0, None
    try:
        try:
            bundle = bundle_results(suite)
            attachment = bundle['attachment']
            body += f'Full delivery archive: {bundle["full"]}\n'
            for name, path in bundle['components'].items():
                body += f'{name} analysis: {path}\n'
            print(f'Full delivery archive: {bundle["full"]}', flush=True)
            if bundle['compact'] is not None:
                body += (f'Email-sized delivery: {bundle["compact"]}\n'
                         f'Compact level: {bundle["compact_level"]}; bulk per-image CSVs and '
                         'manifests omitted; level2 additionally omits galleries. '
                         'Curves, aggregate paired statistics, class diagnostics and reports remain.\n')
            if attachment is None:
                subject += ' / ATTACHMENT TOO LARGE'
                body += 'Even the compact analysis exceeds the email limit; retrieve the paths above after restart.\n'
        except Exception as exc:
            package_error = type(exc).__name__
            subject += ' / ANALYSIS PACKAGE FAILED'
            body += f'Packaging failed ({package_error}); original outputs remain on the instance.\n'
            print(f'Packaging failed ({package_error}).', flush=True)
        for attempts in range(1, 4):
            try:
                send_mail(recipient, subject, body, attachment=attachment)
                sent = True
                print('One completion email accepted by Gmail; check inbox/spam after reconnecting.', flush=True)
                break
            except Exception as exc:
                print(f'Completion email attempt {attempts} failed ({type(exc).__name__}).', flush=True)
                if attempts < 3:
                    time.sleep(5)
        if not sent:
            print('Notification failed; full results remain on the instance disk.', flush=True)
        return sent
    finally:
        _record_delivery(suite, dict(child_exit_code=code, experiment_status=status, email_accepted=sent,
            email_attempts=attempts, package_error=package_error,
            attachment=str(attachment) if attachment else None,
            stop_requested=True, smtp_acceptance_is_not_inbox_delivery=True))
        try:
            if hasattr(os, 'sync'):
                os.sync()
        finally:
            stop_workbench(url, fallback)


def child_environment():
    return {key: value for key, value in os.environ.items()
            if key not in ('GMAIL_USER', 'GMAIL_APP_PASSWORD')}


def child_command(suite):
    return [sys.executable, '-u', str(ROOT / 'generalization_suite.py'), '--resume', str(suite)]


def arguments():
    from generalization_suite import arguments as suite_arguments
    parser = suite_arguments()
    parser.description = __doc__
    parser.add_argument('--to', default='shenwenfei@gmail.com')
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--fallback-os-shutdown', action='store_true')
    return parser


def main(argv=None):
    import torch
    from generalization_suite import make_plan, create_suite, validate_resume
    args = arguments().parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no training or shutdown started')
    plan = validate_resume(args.resume) if args.resume else make_plan(args)
    if args.resume:
        if args.smoke:
            raise ValueError('Resume keeps the saved smoke/full protocol and epoch budgets')
        for name in ('fresh_epochs', 'warm_epochs', 'max_hours'):
            requested = getattr(args, name, None)
            if requested is not None and requested != plan.get(name):
                raise ValueError('Resume keeps the saved smoke/full protocol and epoch/time budgets')
    check_email_ready(args.to)
    if args.fallback_os_shutdown:
        subprocess.run(['sudo', '-n', '-l', '/usr/sbin/shutdown', '-h', 'now'], check=True)
    url = preflight()
    if args.check_only:
        print('Preflight OK; no training, new suite, email or shutdown performed.', flush=True)
        return
    suite = Path(args.resume).resolve() if args.resume else create_suite(args, plan=plan)
    log_path = suite / 'overnight.log'
    armed, code = False, 1
    try:
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
