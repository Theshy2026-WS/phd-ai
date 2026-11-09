"""Train the strongest independent CNN/ViT pair, email reports, keep the VM running.

No Workbench stop, OS shutdown, sudo, or cloud-control command is part of this
runner. Training and analysis finish before the single completion notification.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from run_vit_augreg_overnight import (bundle_results, check_email_ready,
    child_environment, completion_status, send_mail)


ROOT = Path(__file__).resolve().parent


def arguments():
    from best_pair_joint import arguments as training_arguments
    parser = training_arguments()
    parser.description = __doc__
    parser.add_argument('--to', default='shenwenfei@gmail.com')
    parser.add_argument('--check-only', action='store_true')
    return parser


def child_command(suite):
    return [sys.executable, '-u', str(ROOT/'best_pair_joint.py'), '--resume', str(suite)]


def _record_delivery(suite, value):
    try:
        target = Path(suite)/'delivery_status.json'
        temporary = target.with_suffix('.tmp')
        if target.is_symlink() or temporary.is_symlink():
            return
        temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
        for attempt in range(5):
            try:
                temporary.replace(target)
                break
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(.05*(attempt+1))
    except Exception as exc:
        print(f'Could not record delivery ({type(exc).__name__}); instance remains running.', flush=True)


def finish(suite, log_path, recipient, code):
    """Email once after completion; retry failed sends, never stop the VM."""
    status = completion_status(suite, code)
    subject = f'[Tiny ImageNet strongest CNN + ViT joint30] {status}'
    body = (f'Status: {status}\nExit code: {code}\nSuite: {suite}\nLog: {log_path}\n'
        'The report ZIP includes training_analysis and feature_analysis.\n'
        'This is one new 30-epoch joint late-feature-fusion experiment initialized '
        'from the strongest independent CNN and ViT checkpoints. Neither backbone '
        'is independently retrained by this queue. Both encoders and the fusion '
        'classifier update jointly. Read source identities and actual metrics in the reports.\n'
        'Feature mismatch uses a global, label-blind ID permutation rather than '
        'a roll within class-ordered batches.\n'
        'The instance will remain running. No managed stop or OS shutdown is requested.\n'
        'Full checkpoints and probability NPZs remain on the instance disk.\n')
    attachment, sent, attempts, package_error = None, False, 0, None
    try:
        try:
            bundle = bundle_results(suite)
            attachment = bundle['attachment']
            body += f'Full report ZIP: {bundle["full"]}\n'
            if bundle['compact'] is not None:
                body += f'Email report ZIP: {bundle["compact"]}\n'
            if attachment is None:
                subject += ' / ATTACHMENT TOO LARGE'
                body += 'Compact reports exceed the mail limit; retrieve the local ZIP.\n'
        except Exception as exc:
            package_error = type(exc).__name__
            subject += ' / ANALYSIS PACKAGE FAILED'
            body += f'Packaging failed ({package_error}); original outputs remain on disk.\n'
        for attempts in range(1, 4):
            try:
                send_mail(recipient, subject, body, attachment=attachment)
                sent = True
                print('Completion email accepted by SMTP; instance remains running.', flush=True)
                break
            except Exception as exc:
                print(f'Completion email attempt {attempts} failed ({type(exc).__name__}).', flush=True)
                if attempts < 3:
                    time.sleep(5)
        return bool(sent and attachment is not None and package_error is None and status == 'SUCCESS')
    finally:
        _record_delivery(suite, dict(child_exit_code=code, experiment_status=status,
            email_accepted=sent, email_attempts=attempts, package_error=package_error,
            attachment=str(attachment) if attachment else None,
            analysis_package_succeeded=package_error is None,
            delivery_complete=bool(sent and attachment is not None and package_error is None and status == 'SUCCESS'),
            stop_requested=False, instance_policy='keep running',
            smtp_acceptance_is_not_inbox_delivery=True))


def main(argv=None):
    import torch
    from best_pair_joint import make_plan, create_suite, validate_resume
    args = arguments().parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no training started')
    plan = validate_resume(args.resume) if args.resume else make_plan(args)
    if args.resume:
        if args.smoke or (args.epochs is not None and args.epochs != plan['fusion_epochs']):
            raise ValueError('Resume retains its saved full/smoke protocol and30-epoch horizon')
    check_email_ready(args.to)
    if args.check_only:
        print('Preflight OK; no training, suite creation or email; instance remains running.', flush=True)
        return
    suite = Path(args.resume).resolve() if args.resume else create_suite(args, plan=plan)
    log_path = suite/'run.log'
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
            delivered = finish(suite, log_path, args.to, code)
    if code == 0 and not delivered:
        code = 2
    raise SystemExit(code)


if __name__ == '__main__':
    main()
