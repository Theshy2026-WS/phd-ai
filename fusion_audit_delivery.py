"""Send one fixed official-validation report ZIP, then request Workbench stop."""
import json
import os
from pathlib import Path
import time
import zipfile

from overnight_twostage import stop_workbench
from run_hcsi_tiny import _write_json
from run_vit_augreg_overnight import (
    ALLOWED_SUFFIXES, ATTACHMENT_LIMIT, _analysis_files, _atomic_replace,
    _bulk_file, _regular_inside, _temporary_archive, send_mail)


PROTOCOL = 'tiny-confirmatory-fusion-audit-delivery-v1'


def _directory(path):
    path = Path(path).absolute()
    if not path.is_dir() or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('Audit reports require a real directory without symlink ancestors')
    return path


def _report_zip(suite, target, compact):
    directory = suite/'feature_analysis'
    if directory.is_symlink():
        raise ValueError('Audit report component must not be a symlink')
    files, omitted = _analysis_files(directory, compact)
    temporary = _temporary_archive(target)
    root_omitted = []
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path, relative in files:
            archive.write(path, 'feature_analysis/'+relative.as_posix())
        for path in sorted(suite.iterdir()):
            if path.suffix.lower() not in ALLOWED_SUFFIXES or not _regular_inside(path, suite):
                continue
            if (compact and _bulk_file(path)) or (compact >= 2 and path.suffix.lower() == '.png'):
                root_omitted.append(path.name)
            else:
                archive.write(path, 'suite/'+path.name)
        archive.writestr('delivery_manifest.json', json.dumps(dict(protocol=PROTOCOL,
            evaluation_only=True, training_epochs=0, target_partition='official_validation', compact_level=compact,
            components=dict(feature_analysis=dict(exists=directory.is_dir(), report_file_count=len(files))),
            omitted=dict(feature_analysis=omitted, suite=root_omitted),
            allowed_suffixes=sorted(ALLOWED_SUFFIXES),
            excluded_types=['executable source', 'nested archives', 'model checkpoints',
                'probability arrays', 'datasets', 'credentials', 'symlinks'],
            note='Fixed official-validation assessment only; there is no training component. The legacy '
                 'internal audit partition was previously evaluated. No model is selected or tuned '
                 'using these final results. Original checkpoints and prediction caches stay on disk. '
                 'Email acceptance does not confirm inbox delivery or managed stop.'), indent=2))
    with zipfile.ZipFile(temporary) as archive:
        names = archive.namelist()
        if (archive.testzip() is not None or len(names) != len(set(names))
                or any(Path(n).suffix.lower() not in ALLOWED_SUFFIXES for n in names)):
            raise ValueError('Audit report ZIP failed format or CRC verification')
    _atomic_replace(temporary, target)
    return dict(feature_analysis=omitted, suite=root_omitted)


def bundle_results(suite):
    suite = _directory(suite)
    full, compact_path = None, None
    for level, suffix in ((0, ''), (1, '_compact'), (2, '_compact_nogalleries')):
        target = suite.parent/f'{suite.name}_reports{suffix}.zip'
        omitted = _report_zip(suite, target, level)
        if level == 0:
            full = target
        else:
            compact_path = target
        if target.stat().st_size <= ATTACHMENT_LIMIT:
            return dict(full=full, compact=compact_path, attachment=target,
                compact_level=level, omitted=omitted)
    return dict(full=full, compact=compact_path, attachment=None,
        compact_level=level, omitted=omitted)


def completion_status(suite, code):
    if code != 0:
        return 'FAILED / INTERRUPTED'
    path = Path(suite)/'completion.json'
    try:
        if path.is_symlink() or not path.is_file():
            return 'PARTIAL / STATUS UNAVAILABLE'
        value = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return 'PARTIAL / STATUS UNAVAILABLE'
    if (not isinstance(value, dict) or value.get('status') != 'complete'
            or value.get('failure') or value.get('failures')):
        return 'PARTIAL / INCOMPLETE'
    analysis = Path(suite)/'feature_analysis'
    if not analysis.is_dir() or analysis.is_symlink():
        return 'PARTIAL / MISSING ANALYSIS'
    if value.get('official_validation_evaluated') is not True:
        return 'FAILED / AUDIT SCOPE CHANGED'
    return 'SUCCESS'


def _record_delivery(suite, value):
    try:
        _write_json(Path(suite)/'delivery_status.json', value)
    except Exception as exc:
        print(f'Could not record audit delivery ({type(exc).__name__}); managed stop still follows.', flush=True)


def finish(suite, log_path, recipient, code, url, fallback=False):
    status = 'FAILED / STATUS UNAVAILABLE'
    attachment, sent, attempts, package_error, record = None, False, 0, None, None
    try:
        status = completion_status(suite, code)
        subject = f'[Tiny ImageNet confirmatory fusion audit] {status}'
        body = (f'Status: {status}\nSuite: {suite}\nLog: {log_path}\nExit code: {code}\n'
            'FIXED OFFICIAL VALIDATION ONLY: zero training epochs. Checkpoints, the 90/10 probability '
            'mixture, primary comparison and secondary comparisons were locked before inference. '
            'The legacy internal audit partition was previously evaluated and is not used as fresh evidence.\n'
            'The primary comparison is the existing joint model versus the source CNN. '
            'The probability ensemble and initialization/readout controls are fixed secondary analyses. '
            'These final scores must not be used to choose another model or mixture weight.\n'
            'Read the registered plan, confidence intervals, paired corrected/harmed counts and '
            'actual completion status. The assessed partition is the official Tiny ImageNet validation set.\n'
            'One final reports-only ZIP is sent after evaluation and analysis. Checkpoints '
            'and probability caches remain on disk. Managed Workbench stop follows the email '
            'attempts, including failure paths. SMTP acceptance does not prove inbox delivery.\n')
        try:
            bundle = bundle_results(suite)
            attachment = bundle['attachment']
            body += f'Full report ZIP: {bundle["full"]}\n'
            if bundle['compact'] is not None:
                body += f'Email report ZIP: {bundle["compact"]}\n'
            if attachment is None:
                subject += ' / ATTACHMENT TOO LARGE'
                body += 'Compact reports exceed the mail limit; retrieve the local ZIP after restart.\n'
        except Exception as exc:
            package_error = type(exc).__name__
            subject += ' / PACKAGE FAILED'
            body += f'Packaging failed ({package_error}); original reports remain on disk.\n'
        for attempt in range(1, 4):
            attempts = attempt
            try:
                send_mail(recipient, subject, body, attachment=attachment)
                sent = True
                print('Audit report email accepted by SMTP; managed stop follows.', flush=True)
                break
            except Exception as exc:
                print(f'Audit email attempt {attempt} failed ({type(exc).__name__}).', flush=True)
                if attempt < 3:
                    time.sleep(5)
        return bool(sent and attachment is not None and package_error is None and status == 'SUCCESS')
    finally:
        try:
            record = dict(protocol=PROTOCOL, child_exit_code=code, experiment_status=status,
                evaluation_only=True, training_epochs=0, email_accepted=sent,
                email_attempts=attempts, package_error=package_error,
                attachment=str(attachment) if attachment else None,
                delivery_complete=bool(sent and attachment is not None and package_error is None and status == 'SUCCESS'),
                stop_requested=True, instance_policy='email then managed stop',
                managed_stop_confirmed=False, smtp_acceptance_is_not_inbox_delivery=True)
            _record_delivery(suite, record)
        finally:
            try:
                if hasattr(os, 'sync'):
                    os.sync()
            finally:
                try:
                    stop_workbench(url, fallback)
                except BaseException as exc:
                    if record is not None:
                        record['stop_error_type'] = type(exc).__name__
                        _record_delivery(suite, record)
                    raise
