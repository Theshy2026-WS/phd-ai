"""Deliver the three targeted fusion controls once, then stop this Workbench.

Only the parent runner calls ``finish`` after its experiment child has started.
Importing this module performs no email or instance operation. Report packaging
reuses the existing suffix allowlist, secret exclusions and compact ZIP policy.
"""
import json
import os
from pathlib import Path
import time

from overnight_twostage import stop_workbench
from run_hcsi_tiny import _write_json
from run_vit_augreg_overnight import bundle_results, completion_status, send_mail


PROTOCOL = 'tiny-fusion-mechanism-delivery-v1'


def _record_delivery(suite, value):
    try:
        _write_json(Path(suite)/'delivery_status.json', value)
    except Exception as exc:
        print(f'Could not record delivery ({type(exc).__name__}); managed stop still follows.', flush=True)


def finish(suite, log_path, recipient, code, url, fallback=False):
    """Attempt one final report message (retry failures), then managed stop.

    The safe ZIP contains all arms' reports directly, not nested archives. If
    training, packaging or SMTP fails, partial evidence is still delivered when
    possible and the authorized stop path still runs. No startup mail is sent.
    """
    status = 'FAILED / STATUS UNAVAILABLE'
    attachment, sent, attempts, package_error = None, False, 0, None
    record = None
    try:
        status = completion_status(suite, code)
        subject = f'[Tiny ImageNet fusion mechanism controls] {status}'
        body = (f'Status: {status}\nSuite: {suite}\nLog: {log_path}\nExit code: {code}\n'
            'Three targeted controls retain the winning late-fusion model family: '
            'random ViT with CLS/Linear, pretrained ViT with reset CNN classifier columns '
            'and bias, and pretrained ViT with patch-mean/Linear readout. Both encoders '
            'update in every arm. The existing winning model is reused, not retrained.\n'
            'There is no default total runtime cutoff. Read the recorded horizon, source '
            'identities and actual metrics in the attached reports. Development-set '
            'comparisons are exploratory; audit and official validation stay unused.\n'
            'This is one final message after training and analysis, including partial '
            'results on failure. Checkpoints and probability arrays remain on disk.\n'
            'Managed Workbench stop follows the email attempts, even on packaging or '
            'email failure. SMTP acceptance does not prove inbox delivery.\n')
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
                print('Fusion mechanism completion email accepted by SMTP; managed stop follows.', flush=True)
                break
            except Exception as exc:
                print(f'Fusion mechanism email attempt {attempt} failed ({type(exc).__name__}).', flush=True)
                if attempt < 3:
                    time.sleep(5)
        return bool(sent and attachment is not None and package_error is None and status == 'SUCCESS')
    finally:
        # The VM may disappear while its stop request is being processed.
        # Persist the notification outcome before issuing managed stop.
        try:
            record = dict(protocol=PROTOCOL, child_exit_code=code, experiment_status=status,
                email_accepted=sent, email_attempts=attempts, package_error=package_error,
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
