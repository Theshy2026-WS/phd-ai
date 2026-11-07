"""Run the finite Tiny ImageNet ViT AugReg and fusion study, email one report ZIP, then stop Workbench.

Preflight authenticates email and checks GPU/data/parent/stop access without
sending mail or creating a suite. Only a successfully started training child
arms the final notification and stop path. No archive contains executable
source, nested archives, model weights, probability arrays or credentials.
"""
from email.message import EmailMessage
import csv
import json
import os
from pathlib import Path
import smtplib
import ssl
import subprocess
import sys
import time
import traceback
import zipfile

from overnight_twostage import preflight, stop_workbench
from run_notified import load_email_credentials


ROOT = Path(__file__).resolve().parent
ATTACHMENT_LIMIT = 18 * 1024 * 1024
ALLOWED_SUFFIXES = {'.csv', '.json', '.yaml', '.yml', '.md', '.png', '.pdf', '.txt', '.log'}
BLOCKED_DIRECTORIES = {'data', 'datasets', 'checkpoints', '.git', '.venv', 'venv',
                       'source', 'sources', 'source_snapshot', 'credentials', '.config'}
COMPONENTS = ('training', 'feature_analysis')


def _material_name(component):
    return 'training_analysis' if component == 'training' else component


def _secret_name(name):
    lower = name.lower()
    return (lower.startswith(('email.txt', '.env', 'credentials.', 'client_secret',
                              'application_default_credentials', 'service_account',
                              'service-account', 'gcp_service_account', 'secrets.'))
            or lower in {'credentials', 'secrets'})


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


def _bulk_file(path):
    lower = path.name.lower()
    if lower.endswith('_ids.json') or lower in {
            'image_inventory.csv', 'image_hashes.csv', 'image_metadata.csv'}:
        return True
    if path.suffix.lower() != '.csv':
        return False
    if lower in {'prediction.csv', 'predictions.csv', 'paired_selection.csv',
                 'paired_train.csv', 'paired_training.csv'} or lower.startswith('per_image'):
        return True
    # Preserve aggregate paired/class/confusion CSVs. Per-image tables have
    # explicit image identifiers; do not discard every file named paired_*.
    try:
        with path.open(encoding='utf-8-sig') as stream:
            header = [value.strip().lower() for value in next(csv.reader(stream), [])]
    except (OSError, UnicodeError):
        return False
    return bool(set(header) & {'sample_id', 'sample_index', 'image_id', 'image_path', 'relative_path'})


def _analysis_files(directory, compact=0):
    """List report formats only; compact1 removes per-image tables, 2 removes PNGs."""
    directory = Path(directory)
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
            if (compact and _bulk_file(path)) or (compact >= 2 and path.suffix.lower() == '.png'):
                omitted.append(relative.as_posix())
            else:
                files.append((path, relative))
    return files, omitted


def _temporary_archive(destination):
    temporary = destination.with_suffix(destination.suffix + '.tmp')
    if destination.is_symlink() or temporary.is_symlink():
        raise ValueError('Refusing a symlink report archive destination')
    return temporary


def _atomic_replace(source, destination):
    for attempt in range(5):
        try:
            source.replace(destination)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(.05 * (attempt + 1))


def _report_zip(suite, destination, compact):
    temporary = _temporary_archive(destination)
    omissions, components = {}, {}
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for component in COMPONENTS:
            directory = suite / component
            if directory.is_symlink():
                raise ValueError('Analysis component must not be a symlink')
            files, omitted = _analysis_files(directory, compact)
            omissions[component] = omitted
            material = _material_name(component)
            for path, relative in files:
                archive.write(path, f'{material}/{relative.as_posix()}')
            components[component] = dict(directory=str(directory),
                archive_directory=material, exists=directory.is_dir(), report_file_count=len(files))
            archive.writestr(f'{material}/analysis_bundle.json', json.dumps(dict(
                component=component, component_directory_exists=directory.is_dir(),
                report_file_count=len(files), compact_level=compact,
                omitted_analysis_files=omitted,
                note='Missing or interrupted components are partial evidence, not completed experiments.'), indent=2))
        for path in sorted(suite.iterdir()):
            if path.suffix.lower() not in ALLOWED_SUFFIXES or not _regular_inside(path, suite):
                continue
            if compact and _bulk_file(path):
                omissions.setdefault('suite', []).append(path.name)
                continue
            if compact >= 2 and path.suffix.lower() == '.png':
                omissions.setdefault('suite', []).append(path.name)
                continue
            archive.write(path, f'suite/{path.name}')
        archive.writestr('delivery_manifest.json', json.dumps(dict(
            protocol='tiny-vit-augreg-report-delivery-v1', compact_level=compact,
            components=components, omitted=omissions,
            allowed_suffixes=sorted(ALLOWED_SUFFIXES),
            excluded_types=['executable source', 'nested archives', 'model checkpoints',
                            'probability NPZ', 'datasets', 'credentials', 'symlinks'],
            note='Single-layer report ZIP. Source provenance remains in metadata hashes/Git commit; '
                 'source files and all original outputs remain on the VM. '
                 'SMTP acceptance does not prove inbox delivery or attachment-policy acceptance.'), indent=2))
    with zipfile.ZipFile(temporary) as archive:
        if archive.testzip() is not None:
            raise ValueError('Report ZIP failed its CRC verification')
        if any(Path(name).suffix.lower() not in ALLOWED_SUFFIXES for name in archive.namelist()):
            raise ValueError('Report ZIP contains an unexpected file format')
    _atomic_replace(temporary, destination)
    return omissions, components


def bundle_results(suite):
    suite = Path(suite).absolute()
    if not suite.is_dir() or suite.is_symlink() or any(path.is_symlink() for path in suite.parents):
        raise ValueError('Analysis suite must be a real directory without symlink ancestors')
    full = None
    compact_path = None
    components, omitted = {}, {}
    for level, suffix in ((0, ''), (1, '_compact'), (2, '_compact_nogalleries')):
        destination = suite.parent / f'{suite.name}_reports{suffix}.zip'
        omitted, components = _report_zip(suite, destination, level)
        if level == 0:
            full = destination
        else:
            compact_path = destination
        if destination.stat().st_size <= ATTACHMENT_LIMIT:
            return dict(full=full, compact=compact_path, attachment=destination,
                        components=components, compact_level=level, omitted=omitted)
    return dict(full=full, compact=compact_path, attachment=None,
                components=components, compact_level=2, omitted=omitted)


def _email_credentials():
    load_email_credentials()
    user = os.environ.get('GMAIL_USER', '').strip()
    password = os.environ.get('GMAIL_APP_PASSWORD', '').replace(' ', '')
    if not user or not password:
        raise RuntimeError('Missing Gmail credentials; no training or shutdown started')
    return user, password


def _recipient(recipient):
    if not recipient.strip() or any(character in recipient for character in '\r\n'):
        raise ValueError('Need a valid recipient without newlines')


def check_email_ready(recipient):
    """SMTP login only: never transmit a preflight/start email."""
    _recipient(recipient)
    user, password = _email_credentials()
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465, timeout=30,
                              context=ssl.create_default_context()) as smtp:
            smtp.login(user, password)
    except Exception as exc:
        raise RuntimeError(f'Email authentication failed ({type(exc).__name__}); '
                           'no training or shutdown started') from None
    print('Email authentication checked; no email sent.', flush=True)


def send_mail(recipient, subject, body, attachment=None):
    """Transmit the one completion report using its actual application/zip MIME."""
    _recipient(recipient)
    user, password = _email_credentials()
    message = EmailMessage()
    message['From'], message['To'], message['Subject'] = user, recipient, subject
    message.set_content(body)
    if attachment is not None:
        attachment = Path(attachment)
        if attachment.suffix.lower() != '.zip' or attachment.is_symlink() or not attachment.is_file():
            raise ValueError('Completion attachment must be a regular report ZIP')
        if attachment.stat().st_size > ATTACHMENT_LIMIT:
            raise ValueError('Report ZIP exceeds the email attachment limit')
        message.add_attachment(attachment.read_bytes(), maintype='application',
                               subtype='zip', filename=attachment.name)
    with smtplib.SMTP_SSL('smtp.gmail.com', 465, timeout=30,
                          context=ssl.create_default_context()) as smtp:
        smtp.login(user, password)
        smtp.send_message(message)


def _record_delivery(suite, value):
    try:
        path = Path(suite) / 'delivery_status.json'
        temporary = path.with_suffix('.tmp')
        if path.is_symlink() or temporary.is_symlink():
            return
        temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
        _atomic_replace(temporary, path)
    except Exception as exc:
        print(f'Could not record delivery status ({type(exc).__name__}).', flush=True)


def _incomplete(value):
    if isinstance(value, dict):
        if value.get('status') not in (None, 'complete', 'completed', 'success'):
            return True
        if value.get('failures') or value.get('skipped_jobs') or value.get('deferred_jobs'):
            return True
        return any(_incomplete(item) for item in value.values() if isinstance(item, (dict, list)))
    if isinstance(value, list):
        return any(_incomplete(item) for item in value)
    return False


def completion_status(suite, code):
    if code != 0:
        return 'FAILED / INTERRUPTED'
    for name in ('completion.json', 'status.json'):
        path = Path(suite) / name
        if path.is_file() and not path.is_symlink():
            try:
                value = json.loads(path.read_text(encoding='utf-8'))
            except (OSError, ValueError):
                return 'PARTIAL / STATUS UNAVAILABLE'
            if not isinstance(value, dict) or value.get('status') not in ('complete', 'completed', 'success'):
                return 'PARTIAL / INCOMPLETE'
            if _incomplete(value):
                return 'PARTIAL / INCOMPLETE'
            if not all((Path(suite) / component).is_dir() and
                       not (Path(suite) / component).is_symlink() for component in COMPONENTS):
                return 'PARTIAL / MISSING ANALYSIS'
            return 'SUCCESS'
    return 'PARTIAL / STATUS UNAVAILABLE'


def finish(suite, log_path, url, fallback, recipient, code):
    """One completion notification (retrying failed sends), then request stop."""
    status = completion_status(suite, code)
    subject = f'[Tiny ImageNet ViT AugReg fusion] {status}'
    body = (f'Status: {status}\nExit code: {code}\nSuite: {suite}\nLog: {log_path}\n'
            'The single-layer ZIP contains training_analysis and feature_analysis reports.\n'
            'The queue trains independent ViTs with a predeclared AugReg search, then tests fixed-CNN fusion. '
            'Frozen CNN-only head controls, frozen/joint fusion and lightweight fusion comparisons are described in the saved plan.\n'
            'The slow native64 global-attention experiment is not scheduled.\n'
            'Executable source, nested archives, checkpoints, probability arrays and credentials '
            'are excluded. Source hashes and model/recipe provenance remain in metadata.\n'
            'Read the saved plan and per-job completion status before comparing results.\n'
            'Full checkpoints and analysis files remain on the instance disk.\n'
            'Managed Workbench stop will be requested after the completion email attempts.\n')
    attachment, sent, attempts, package_error = None, False, 0, None
    try:
        try:
            bundle = bundle_results(suite)
            attachment = bundle['attachment']
            body += f'Full report ZIP: {bundle["full"]}\n'
            print(f'Full report ZIP: {bundle["full"]}', flush=True)
            if bundle['compact'] is not None:
                body += (f'Email-sized report ZIP: {bundle["compact"]}\n'
                         f'Compact level: {bundle["compact_level"]}; level1 omits per-image tables; '
                         'level2 also omits PNGs. Aggregates, class diagnostics and epochs remain.\n')
            if attachment is None:
                subject += ' / ATTACHMENT TOO LARGE'
                body += 'Even the compact reports exceed the email limit; retrieve the local ZIP after restart.\n'
        except Exception as exc:
            package_error = type(exc).__name__
            subject += ' / ANALYSIS PACKAGE FAILED'
            body += f'Packaging failed ({package_error}); original outputs remain on the instance.\n'
            print(f'Packaging failed ({package_error}).', flush=True)
        for attempts in range(1, 4):
            try:
                send_mail(recipient, subject, body, attachment=attachment)
                sent = True
                print('Completion email accepted by SMTP; inbox delivery is not confirmed.', flush=True)
                break
            except Exception as exc:
                print(f'Completion email attempt {attempts} failed ({type(exc).__name__}).', flush=True)
                if attempts < 3:
                    time.sleep(5)
        if not sent:
            print('Notification failed; report ZIPs remain on the instance disk.', flush=True)
        return bool(sent and attachment is not None and package_error is None and status == 'SUCCESS')
    finally:
        _record_delivery(suite, dict(child_exit_code=code, experiment_status=status,
            email_accepted=sent, email_attempts=attempts, package_error=package_error,
            attachment=str(attachment) if attachment else None, stop_requested=True,
            analysis_package_succeeded=package_error is None,
            analysis_attachment_available=attachment is not None,
            delivery_complete=bool(sent and attachment is not None
                                   and package_error is None and status == 'SUCCESS'),
            smtp_acceptance_is_not_inbox_delivery=True))
        try:
            if hasattr(os, 'sync'):
                os.sync()
        finally:
            stop_workbench(url, fallback)


def child_environment():
    return {key: value for key, value in os.environ.items()
            if key not in ('GMAIL_USER', 'GMAIL_APP_PASSWORD')}


def child_command(suite):
    return [sys.executable, '-u', str(ROOT / 'vit_augreg_suite.py'), '--resume', str(suite)]


def arguments():
    from vit_augreg_suite import arguments as suite_arguments
    parser = suite_arguments()
    parser.description = __doc__
    parser.add_argument('--to', default='shenwenfei@gmail.com')
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--fallback-os-shutdown', action='store_true')
    return parser


def main(argv=None):
    import torch
    from vit_augreg_suite import make_plan, create_suite, validate_resume
    args = arguments().parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no training or shutdown started')
    plan = validate_resume(args.resume) if args.resume else make_plan(args)
    if args.resume:
        if args.smoke:
            raise ValueError('Resume keeps the saved smoke/full protocol and epoch budgets')
        if getattr(args, 'profiles', None) is not None and list(args.profiles) != list(plan.get('profiles', [])):
            raise ValueError('Resume keeps the saved profile search and source selection policy')
        for name in ('vit_epochs', 'fusion_epochs'):
            requested = getattr(args, name, None)
            if requested is not None and requested != plan.get(name):
                raise ValueError('Resume keeps the saved smoke/full protocol and epoch budgets')
    check_email_ready(args.to)
    if args.fallback_os_shutdown:
        subprocess.run(['sudo', '-n', '-l', '/usr/sbin/shutdown', '-h', 'now'], check=True)
    url = preflight()
    if args.check_only:
        print('Preflight OK; no training, new suite, email or shutdown performed.', flush=True)
        return
    suite = Path(args.resume).resolve() if args.resume else create_suite(args, plan=plan)
    log_path = suite / 'overnight.log'
    armed, code, delivered = False, 1, False
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
            delivered = finish(suite, log_path, url, args.fallback_os_shutdown, args.to, code)
    if code == 0 and not delivered:
        code = 2
    raise SystemExit(code)


if __name__ == '__main__':
    main()
