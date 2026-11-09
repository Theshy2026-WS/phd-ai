"""One reports-only email for pairwise and HCSI, followed by managed VM stop.

Only the combined parent runner calls ``finish``, after a child has actually
started. The original child protocols stay email-free and do not stop the VM.
No network or instance operation is performed merely by importing this module.
"""
import json
import os
from pathlib import Path
import time
import zipfile

from overnight_twostage import stop_workbench
from run_vit_augreg_overnight import (
    ALLOWED_SUFFIXES, ATTACHMENT_LIMIT, _analysis_files, _atomic_replace,
    _bulk_file, _regular_inside, _temporary_archive, check_email_ready,
    child_environment, completion_status, send_mail,
)


CHILDREN = ('pairwise', 'hcsi')
COMPONENTS = ('training', 'feature_analysis')
PROTOCOL = 'tiny-combined-fusion-report-delivery-v1'


def _directory(path, *, required):
    path = Path(path).absolute()
    if path.is_symlink() or any(parent.is_symlink() for parent in path.parents):
        raise ValueError('Report directories must not have symlink ancestors')
    if path.exists() and not path.is_dir():
        raise ValueError('Report source must be a directory')
    if required and not path.is_dir():
        raise ValueError('Combined queue directory is missing')
    return path


def child_suites(queue, children=None):
    """Resolve the two declared sources, rejecting duplicate or unsafe names."""
    queue = _directory(queue, required=True)
    if children is None:
        marker = queue/'queue.json'
        if not _regular_inside(marker, queue):
            raise ValueError('Need a regular combined queue manifest')
        records = json.loads(marker.read_text(encoding='utf-8')).get('children')
        if not isinstance(records, list) or len(records) != len(CHILDREN):
            raise ValueError('Combined manifest must declare both child suites')
        if len({item.get('name') for item in records if isinstance(item, dict)}) != len(CHILDREN):
            raise ValueError('Duplicate or invalid combined child names')
        children = {item['name']: item['suite'] for item in records}
    if not isinstance(children, dict) or set(children) != set(CHILDREN):
        raise ValueError('Only pairwise and hcsi report roots are supported')
    result = {name: _directory(children[name], required=False) for name in CHILDREN}
    if result['pairwise'].resolve() == result['hcsi'].resolve():
        raise ValueError('The two studies must have separate suite roots')
    return result


def _root_reports(root, compact):
    """Root-level metadata only; components are collected separately."""
    files, omitted = [], []
    if not root.is_dir():
        return files, omitted
    for path in sorted(root.iterdir()):
        if path.suffix.lower() not in ALLOWED_SUFFIXES or not _regular_inside(path, root):
            continue
        if (compact and _bulk_file(path)) or (compact >= 2 and path.suffix.lower() == '.png'):
            omitted.append(path.name)
        else:
            files.append(path)
    return files, omitted


def _report_zip(queue, children, destination, compact):
    temporary = _temporary_archive(destination)
    components, omissions = {}, {}
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, suite in children.items():
            components[name] = {}
            for component in COMPONENTS:
                directory = suite/component
                if directory.is_symlink():
                    raise ValueError('Child report component must not be a symlink')
                files, omitted = _analysis_files(directory, compact)
                material = 'training_analysis' if component == 'training' else component
                prefix = f'{name}/{material}'
                for path, relative in files:
                    archive.write(path, f'{prefix}/{relative.as_posix()}')
                evidence = dict(component=component, directory_exists=directory.is_dir(),
                    report_file_count=len(files), compact_level=compact,
                    omitted_analysis_files=omitted,
                    note='Missing or interrupted components are partial evidence, not completed experiments.')
                archive.writestr(prefix+'/analysis_bundle.json', json.dumps(evidence, indent=2))
                components[name][component] = evidence
                omissions[prefix] = omitted
            files, omitted = _root_reports(suite, compact)
            for path in files:
                archive.write(path, f'{name}/suite/{path.name}')
            omissions[name+'/suite'] = omitted
        files, omitted = _root_reports(queue, compact)
        for path in files:
            archive.write(path, 'queue/'+path.name)
        omissions['queue'] = omitted
        archive.writestr('delivery_manifest.json', json.dumps(dict(
            protocol=PROTOCOL, compact_level=compact, components=components,
            omitted=omissions, allowed_suffixes=sorted(ALLOWED_SUFFIXES),
            excluded_types=['executable source', 'nested archives', 'model checkpoints',
                'probability arrays', 'datasets', 'credentials', 'symlinks'],
            note='One ordinary ZIP for both studies. Weights and original outputs remain on disk. '
                 'SMTP acceptance does not prove inbox delivery. Managed instance stop follows email attempts.'),
            indent=2))
    with zipfile.ZipFile(temporary) as archive:
        if archive.testzip() is not None:
            raise ValueError('Combined report ZIP failed CRC verification')
        names = archive.namelist()
        if len(names) != len(set(names)) or any(Path(name).suffix.lower() not in ALLOWED_SUFFIXES for name in names):
            raise ValueError('Combined ZIP contains duplicate or unexpected formats')
    _atomic_replace(temporary, destination)
    return components, omissions


def bundle_results(queue, children=None):
    """Merge report files directly; never attach the two ZIPs inside another ZIP."""
    queue = _directory(queue, required=True)
    children = child_suites(queue, children)
    full = compact_path = None
    for level, suffix in ((0, ''), (1, '_compact'), (2, '_compact_nogalleries')):
        target = queue.parent/f'{queue.name}_reports{suffix}.zip'
        components, omitted = _report_zip(queue, children, target, level)
        if level == 0:
            full = target
        else:
            compact_path = target
        if target.stat().st_size <= ATTACHMENT_LIMIT:
            return dict(full=full, compact=compact_path, attachment=target,
                components=components, compact_level=level, omitted=omitted)
    return dict(full=full, compact=compact_path, attachment=None,
        components=components, compact_level=level, omitted=omitted)


def queue_status(queue, code, *, timed_out=False):
    if timed_out:
        return 'PARTIAL / TIME LIMIT'
    queue = Path(queue)
    marker = queue/'completion.json'
    try:
        if not _regular_inside(marker, queue):
            return 'FAILED / INTERRUPTED' if code else 'PARTIAL / STATUS UNAVAILABLE'
        record = json.loads(marker.read_text(encoding='utf-8'))
        if record.get('status') == 'budget_exhausted':
            return 'PARTIAL / TIME LIMIT'
        if code:
            return 'FAILED / INTERRUPTED'
        if record.get('status') not in ('complete', 'completed', 'success'):
            return 'PARTIAL / INCOMPLETE'
        children = child_suites(queue)
        if any(completion_status(suite, 0) != 'SUCCESS' for suite in children.values()):
            return 'PARTIAL / CHILD INCOMPLETE'
        return 'SUCCESS'
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return 'PARTIAL / STATUS UNAVAILABLE'


def _record_delivery(queue, value):
    try:
        target = Path(queue)/'delivery_status.json'
        temporary = target.with_suffix('.tmp')
        if target.is_symlink() or temporary.is_symlink():
            raise ValueError('Delivery record cannot be a symlink')
        temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
        _atomic_replace(temporary, target)
    except Exception as exc:
        print(f'Could not record combined delivery ({type(exc).__name__}); managed stop still follows.', flush=True)


def finish(queue, log_path, recipient, code, url, fallback=False, *,
           timed_out=False, budget=None, before_stop=None):
    """Attempt one unified notification, then stop even if packaging/mail fails.

    Caller authorization: invoke only after this queue has spawned a child.
    ``before_stop`` persists the caller's final wall budget before VM termination.
    """
    status = 'FAILED / STATUS UNAVAILABLE'
    attachment, sent, attempts, package_error = None, False, 0, None
    callback_error = None
    record = None
    try:
        status = queue_status(queue, code, timed_out=timed_out)
        subject = f'[Tiny ImageNet pairwise + HCSI] {status}'
        body = (f'Status: {status}\nQueue: {queue}\nLog: {log_path}\nExit code: {code}\n'
            'The queue includes four frozen-anchor pairwise resolver tests and five full-update '
            'HCSI/late-fusion controls. Each original study retains its own training and analysis identity.\n'
            'This is one combined completion/partial-result message, sent after the queue ends.\n'
            'Incomplete jobs are partial evidence, not completed comparisons. Full weights and '
            'prediction arrays remain on the instance disk.\n'
            'Managed Workbench stop is requested after the email attempts, including on training, '
            'packaging or email failure. SMTP acceptance does not prove inbox delivery.\n')
        if budget:
            body += (f'Total configured wall budget: {budget["total_seconds"]/3600:g}h; '
                f'previously spent: {budget.get("prior_spent_seconds", 0):g}s.\n')
        try:
            bundle = bundle_results(queue)
            attachment = bundle['attachment']
            body += f'Full combined reports: {bundle["full"]}\n'
            if bundle['compact'] is not None:
                body += f'Email reports: {bundle["compact"]}\n'
            if attachment is None:
                subject += ' / ATTACHMENT TOO LARGE'
                body += 'Compact reports exceed the mail limit; retrieve the local ZIP after restart.\n'
        except Exception as exc:
            package_error = type(exc).__name__
            subject += ' / PACKAGE FAILED'
            body += f'Packaging failed ({package_error}); original reports remain on disk.\n'
        for attempt in range(1, 4):
            if budget and time.monotonic()+35 > budget['total_deadline_monotonic']:
                print('Delivery time exhausted; reports remain local and stop is still requested.', flush=True)
                break
            attempts = attempt
            try:
                send_mail(recipient, subject, body, attachment=attachment)
                sent = True
                print('Combined completion email accepted by SMTP; managed stop follows.', flush=True)
                break
            except Exception as exc:
                print(f'Combined email attempt {attempt} failed ({type(exc).__name__}).', flush=True)
                if attempt < 3:
                    time.sleep(5)
        return bool(sent and attachment is not None and package_error is None and status == 'SUCCESS')
    finally:
        # The instance can disappear during stop_workbench, so accounting must
        # be finalized before calling it. No callback or record error may block stop.
        try:
            if before_stop is not None:
                try:
                    before_stop()
                except BaseException as exc:
                    callback_error = type(exc).__name__
                    print(f'Final budget recording failed ({callback_error}); managed stop still follows.', flush=True)
            record = dict(child_exit_code=code, experiment_status=status,
                email_accepted=sent, email_attempts=attempts, package_error=package_error,
                attachment=str(attachment) if attachment else None,
                delivery_complete=bool(sent and attachment is not None and package_error is None and status == 'SUCCESS'),
                before_stop_error=callback_error, stop_requested=True,
                instance_policy='email then managed stop', managed_stop_confirmed=False,
                smtp_acceptance_is_not_inbox_delivery=True)
            _record_delivery(queue, record)
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
                        try:
                            _record_delivery(queue, record)
                        except Exception:
                            pass
                    raise
