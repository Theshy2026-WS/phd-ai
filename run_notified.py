"""Run a command, email its exit status via Gmail, and NEVER shut down the VM.

Uses GMAIL_USER/GMAIL_APP_PASSWORD in the environment or private email.txt. Use an app
password, not your Google login password. Run inside tmux to survive disconnects.
"""
import argparse
from collections import deque
from email.message import EmailMessage
import os
from pathlib import Path
import smtplib
import ssl
import subprocess
import sys
import time
import stat


def load_email_credentials(path=None):
    """Parse two literal keys, never execute file content or print secrets.

    Explicit environment values take precedence. The optional local file must
    be owner-only on POSIX. It is intentionally not a shell source file.
    """
    keys = ('GMAIL_USER', 'GMAIL_APP_PASSWORD')
    if all(os.environ.get(k, '').strip() for k in keys):
        return
    path = Path(path) if path is not None else Path(__file__).resolve().parent/'email.txt'
    if not path.exists():
        return
    if path.is_symlink() or not path.is_file():
        raise RuntimeError('email.txt must be a regular private file')
    if os.name == 'posix' and stat.S_IMODE(path.stat().st_mode) & 0o077:
        raise RuntimeError('Protect credentials first: chmod 600 email.txt')
    values = {}
    for line in path.read_text(encoding='utf-8-sig').splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        # Support KEY: value (user format) and KEY=value, without shell eval.
        parts = line.replace('=', ':', 1).split(':', 1) if ':' not in line else line.split(':', 1)
        if len(parts) != 2 or parts[0].strip() not in keys:
            raise RuntimeError('Invalid email.txt format; expected two credential keys')
        key, value = parts[0].strip(), parts[1].strip()
        if key in values or not value or set(value) == {'*'}:
            raise RuntimeError('Missing, duplicate or placeholder credential in email.txt')
        values[key] = value
    if set(values) != set(keys):
        raise RuntimeError('email.txt must contain both credential keys')
    for key in keys:
        if not os.environ.get(key, '').strip():
            os.environ[key] = values[key]


def send_mail(recipient, subject, body, attachment=None):
    load_email_credentials()
    user = os.environ.get('GMAIL_USER', '').strip()
    password = os.environ.get('GMAIL_APP_PASSWORD', '').replace(' ', '')
    if not user or not password:
        raise RuntimeError('Set GMAIL_USER and GMAIL_APP_PASSWORD first.')
    message = EmailMessage()
    message['From'] = user
    message['To'] = recipient
    message['Subject'] = subject
    message.set_content(body)
    if attachment is not None:
        attachment = Path(attachment)
        # Leave room for MIME/base64 expansion within typical SMTP limits.
        if attachment.stat().st_size > 18 * 1024 * 1024:
            raise ValueError('Analysis archive exceeds the 18 MiB attachment limit.')
        message.add_attachment(attachment.read_bytes(), maintype='application',
                               subtype='gzip', filename=attachment.name)
    with smtplib.SMTP_SSL('smtp.gmail.com', 465, timeout=30,
                          context=ssl.create_default_context()) as smtp:
        smtp.login(user, password)
        smtp.send_message(message)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--to', required=True)
    parser.add_argument('--log', default='mixers_100ep_overnight.log')
    parser.add_argument('--test-email', action='store_true')
    parser.add_argument('--analysis-suite', help='Package this correction suite and attach it to the completion email')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command
    if command and command[0] == '--':
        command = command[1:]
    if not args.test_email and not command:
        parser.error('Provide a training command after --')
    # Fail before training if email configuration is broken. No secrets in logs.
    try:
        send_mail(args.to, '[Tiny ImageNet] Notification test' if args.test_email else
                  '[Tiny ImageNet] Training starting',
                  'Email notification works. The instance will NOT be stopped automatically.'
                  if args.test_email else 'Training is starting. A second email will report '
                  'success or failure. The instance will remain running.')
    except Exception as exc:
        raise SystemExit(f'Email preflight failed ({type(exc).__name__}); no training started.')
    if args.test_email:
        print('Test email accepted by Gmail; check your inbox/spam folder.')
        return
    started = time.monotonic()
    tail = deque(maxlen=25)
    code = 1
    log_path = Path(args.log).resolve()
    try:
        with log_path.open('a', encoding='utf-8', buffering=1) as log:
            # Do not pass email credentials to the training child process.
            child_env = {k: v for k, v in os.environ.items()
                         if k not in ('GMAIL_USER', 'GMAIL_APP_PASSWORD')}
            with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, errors='replace', bufsize=1, env=child_env) as process:
                try:
                    for line in process.stdout:
                        log.write(line)
                        print(line, end='', flush=True)
                        tail.append(line)
                    code = process.wait()
                except BaseException:
                    process.terminate()
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    raise
            log.write(f'\nTraining exit code: {code}\n')
    finally:
        status = 'SUCCESS' if code == 0 else 'FAILED / INTERRUPTED'
        body = (f'Status: {status}\nExit code: {code}\n'
                f'Elapsed: {(time.monotonic()-started)/3600:.2f} hours\n'
                f'Log: {log_path}\nInstance is NOT automatically stopped.\n\n'
                'Last output lines:\n' + ''.join(tail))
        archive = None
        if args.analysis_suite:
            try:
                from overnight_correction import bundle_results
                # This helper ONLY packages files; never invokes shutdown.
                archive = bundle_results(args.analysis_suite, log_path)
                body += f'\n\nAnalysis archive: {archive}\n'
                if archive.stat().st_size > 18 * 1024 * 1024:
                    raise ValueError('Attachment too large')
            except Exception as exc:
                body += (f'\nAttachment unavailable ({type(exc).__name__}); '
                         'analysis files remain on the instance.\n')
                archive = None
                if code == 0:
                    code = 2
        sent = False
        for attempt in range(3):
            try:
                subject = f'[Tiny ImageNet] {status}'
                if args.analysis_suite and archive is None:
                    subject += ' / ATTACHMENT FAILED'
                if archive is None:
                    send_mail(args.to, subject, body)
                else:
                    send_mail(args.to, subject, body, attachment=archive)
                sent = True
                print('Completion email accepted by Gmail.', flush=True)
                break
            except Exception as exc:
                print(f'Email attempt {attempt+1} failed ({type(exc).__name__}).', flush=True)
                if attempt < 2:
                    time.sleep(5)
        if not sent:
            print('WARNING: notification failed; inspect the log manually.', flush=True)
            if code == 0:
                code = 2
    raise SystemExit(code)


if __name__ == '__main__':
    main()
