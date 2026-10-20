"""Four full dual-image jobs, analysis email, then managed Workbench stop.

Use existing GMAIL_USER/GMAIL_APP_PASSWORD environment variables inside tmux.
No credential storage. Preflight failure never arms shutdown.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

from overnight_twostage import preflight, stop_workbench
from overnight_correction import bundle_results
from run_notified import send_mail, load_email_credentials

ROOT = Path(__file__).resolve().parent


def finish(suite, log, url, fallback, recipient, code):
    try:
        archive = None
        problem = ''
        try:
            archive = bundle_results(suite, log)
            print(f'Analysis archive: {archive}', flush=True)
        except Exception as exc:
            problem = f'\nPackaging failed: {type(exc).__name__}. Files remain on disk.'
        status = 'SUCCESS' if code == 0 else 'FAILED / INTERRUPTED'
        body = (f'Status: {status}\nExit code: {code}\nSuite: {suite}\n'
                'Workbench shutdown will be requested after this notification attempt.'
                '\nResults and checkpoints remain on the instance disk.' + problem)
        for attempt in range(3):
            try:
                send_mail(recipient, f'[Tiny ImageNet] {status}', body, attachment=archive)
                print('Completion email accepted by Gmail.', flush=True)
                break
            except Exception as exc:
                print(f'Email attempt {attempt+1} failed ({type(exc).__name__}).', flush=True)
                if attempt < 2:
                    time.sleep(5)
        else:
            print('WARNING: email failed; retrieve results from disk after restart.', flush=True)
    finally:
        try:
            if hasattr(os, 'sync'):
                os.sync()
        finally:
            stop_workbench(url, fallback)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-run')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience; 0 disables')
    parser.add_argument('--to', default='shenwenfei@gmail.com')
    parser.add_argument('--fallback-os-shutdown', action='store_true')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    if args.patience < 0:
        parser.error('--patience must be nonnegative')
    import torch
    from correction_experiment import find_parent
    from dual_image_experiment import create_suite, JOBS
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no training or shutdown started')
    load_email_credentials()
    for key in ('GMAIL_USER', 'GMAIL_APP_PASSWORD'):
        if not os.environ.get(key, '').strip():
            raise RuntimeError(f'Missing {key}; no training or shutdown started')
    find_parent(args.base_run)
    if args.fallback_os_shutdown:
        subprocess.run(['sudo', '-n', '-l', '/usr/sbin/shutdown', '-h', 'now'], check=True)
    url = preflight()
    send_mail(args.to, '[Tiny ImageNet] Overnight preflight OK',
              'Email and stop permissions checked. ' +
              ('Check only: no training or shutdown.' if args.check_only else
               'Starting seed42 jobs: prediction100, feature100, prediction25, aacnn100. '
               'Analysis email will be attempted before automatic shutdown.'))
    if args.check_only:
        print('Preflight OK; no training or shutdown performed.')
        return
    suite = create_suite(argparse.Namespace(base_run=args.base_run, smoke=False,
                                          variants=list(JOBS), patience=args.patience))
    print(f'Suite: {suite}', flush=True)
    log_path = suite/'training.log'
    code = 1
    try:
        with log_path.open('a', encoding='utf-8', buffering=1) as log:
            env = {k: v for k, v in os.environ.items()
                   if k not in ('GMAIL_USER', 'GMAIL_APP_PASSWORD')}
            with subprocess.Popen([sys.executable, '-u', str(ROOT/'dual_image_experiment.py'),
                                   '--resume', str(suite)], cwd=ROOT, env=env,
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, errors='replace', bufsize=1) as child:
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
            log.write(f'\nExit code: {code}\n')
    finally:
        finish(suite, log_path, url, args.fallback_os_shutdown, args.to, code)
    raise SystemExit(code)


if __name__ == '__main__':
    main()
