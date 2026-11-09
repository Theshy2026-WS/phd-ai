"""Run bounded HCSI mechanism and matched-control tests; email reports and keep the VM running.

The wall budget starts before preflight. A cooperative child deadline and an
independent watchdog reserve time for packaging and SMTP. No machine stop or
shutdown action is part of this runner, including failure and timeout paths.
"""
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import traceback

from run_vit_augreg_overnight import (bundle_results, check_email_ready,
    child_environment, completion_status, send_mail)


ROOT = Path(__file__).resolve().parent


def hours_argument(value):
    import argparse
    try:
        hours = float(value)
    except (ValueError, TypeError):
        raise argparse.ArgumentTypeError('max-hours must be a number') from None
    if not math.isfinite(hours) or not .1 <= hours <= 10:
        raise argparse.ArgumentTypeError('max-hours must be between0.1 and10; longer runs are disabled')
    return hours


def arguments():
    from hcsi_tiny_experiment import arguments as training_arguments
    parser = training_arguments()
    parser.description = __doc__
    parser.add_argument('--max-hours', type=hours_argument, default=10.,
        help='Total wall budget in hours,0.1..10; includes a packaging/email reserve')
    parser.add_argument('--to', default='shenwenfei@gmail.com')
    parser.add_argument('--check-only', action='store_true')
    return parser


def budget_deadlines(start_monotonic, hours, spent_seconds=0.):
    """Use at most10h; preserve a15min reserve for the default10h queue."""
    total = hours_argument(hours)*3600.
    spent = float(spent_seconds)
    if not math.isfinite(spent) or spent < 0:
        raise ValueError('Recorded suite wall time must be finite and nonnegative')
    remaining = max(0., total-spent)
    reserve = min(900., max(120., remaining*.025), remaining*.5)
    return dict(start_monotonic=float(start_monotonic),
        total_deadline_monotonic=float(start_monotonic)+remaining,
        child_deadline_monotonic=float(start_monotonic)+remaining-reserve,
        total_seconds=total, prior_spent_seconds=spent, remaining_seconds=remaining,
        delivery_reserve_seconds=reserve)


def saved_budget(suite, requested_hours, *, now_unix=None):
    """Resume can lower a saved cap, never reset or enlarge a suite's budget."""
    cap, spent = hours_argument(requested_hours)*3600., 0.
    path = Path(suite)/'wall_budget.json'
    if path.is_symlink():
        raise ValueError('Wall-budget record must not be a symlink')
    if path.is_file():
        value = json.loads(path.read_text(encoding='utf-8'))
        previous_cap = float(value['cap_seconds'])
        spent = float(value['elapsed_active_seconds'])
        if (not math.isfinite(previous_cap) or not 360 <= previous_cap <= 36000
                or not math.isfinite(spent) or spent < 0):
            raise ValueError('Invalid saved cumulative wall budget')
        cap = min(cap, previous_cap)
        if not value.get('invocation_finished', False):
            # If the wrapper was killed, charge all time since its last start.
            # This deliberately cannot turn a crash into a fresh10-hour budget.
            now = time.time() if now_unix is None else float(now_unix)
            prior_start = float(value['invocation_start_unix'])
            if not math.isfinite(prior_start) or prior_start > now+5:
                raise ValueError('Invalid unfinished wall-budget invocation timestamp')
            spent += max(0., now-prior_start)
    return cap/3600., spent


def record_budget(suite, budget, *, invocation_start_unix, finished):
    elapsed = max(0., time.monotonic()-budget['start_monotonic'])
    spent = budget['prior_spent_seconds']+(elapsed if finished else 0.)
    _write_json(Path(suite)/'wall_budget.json', dict(cap_seconds=budget['total_seconds'],
        max_hours=budget['total_seconds']/3600., elapsed_active_seconds=spent,
        invocation_start_unix=float(invocation_start_unix), invocation_finished=bool(finished),
        delivery_reserve_seconds=budget['delivery_reserve_seconds'],
        child_deadline_unix=time.time()+budget['child_deadline_monotonic']-time.monotonic(),
        scope='Cumulative suite wall time, including preflight and delivery. Clean idle gaps between resumes are excluded; an unfinalized invocation is conservatively charged until resume.',
        instance_policy='keep running'))


def child_command(suite, deadline_unix):
    return [sys.executable, '-u', str(ROOT/'hcsi_tiny_experiment.py'),
        '--resume', str(suite), '--deadline-unix', str(float(deadline_unix))]


def _write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    if path.is_symlink() or temporary.is_symlink():
        raise ValueError('Refusing a symlink delivery record')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    for attempt in range(5):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(.05*(attempt+1))


class ChildWatchdog:
    """Stop only the child, even when its stdout or data loader is stalled."""
    def __init__(self, child, deadline_monotonic, *, grace_seconds=30.):
        self.child, self.deadline = child, float(deadline_monotonic)
        self.grace_seconds = float(grace_seconds)
        self.cancelled = threading.Event()
        self.timed_out = threading.Event()
        self.error = None
        self.thread = threading.Thread(target=self._run, daemon=True,
            name='hcsi-tiny-walltime-watchdog')

    def _run(self):
        remaining = max(0., self.deadline-time.monotonic())
        if self.cancelled.wait(remaining):
            return
        if self.child.poll() is not None and not _owned_posix_group(self.child):
            return
        self.timed_out.set()
        print('Compute deadline reached; stopping the experiment child and retaining partial reports.', flush=True)
        try:
            _stop_child(self.child, grace_seconds=self.grace_seconds)
        except Exception as exc:
            self.error = type(exc).__name__
            print(f'Child deadline handling failed ({self.error}); no VM control requested.', flush=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *unused):
        self.cancelled.set()
        self.thread.join(timeout=max(1., 2*self.grace_seconds+1))
        return False


def _owned_posix_group(child):
    return (os.name == 'posix' and getattr(child, '_resolver_owned_group', False)
        and isinstance(getattr(child, 'pid', None), int) and child.pid > 0)


def _signal_owned_group(child, signum):
    if not _owned_posix_group(child):
        raise ValueError('Refusing to signal a process group not created by this runner')
    try:
        os.killpg(child.pid, signum)
    except ProcessLookupError:
        pass


def _stop_child(child, grace_seconds=30.):
    group = _owned_posix_group(child)
    if child.poll() is not None and not group:
        return
    if group:
        _signal_owned_group(child, signal.SIGTERM)
    else:
        child.terminate()
    try:
        child.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        if group:
            _signal_owned_group(child, getattr(signal, 'SIGKILL', 9))
        else:
            child.kill()
        child.wait(timeout=grace_seconds)
    finally:
        if group:
            # DataLoader descendants can keep stdout open after main exits.
            # Kill only this newly created process group, including orphans.
            _signal_owned_group(child, getattr(signal, 'SIGKILL', 9))


def finish(suite, log_path, recipient, code, *, timed_out=False, budget=None):
    """Package completed or partial evidence and attempt one completion email."""
    # Cooperative deadline handling can finish before the watchdog fires.
    # Its nonzero child exit still represents partial time-limited evidence.
    if not timed_out:
        marker = Path(suite)/'completion.json'
        if marker.is_file() and not marker.is_symlink():
            try:
                timed_out = json.loads(marker.read_text(encoding='utf-8')).get('status') == 'budget_exhausted'
            except (OSError, ValueError, AttributeError):
                pass
    status = completion_status(suite, code)
    if timed_out:
        status = 'PARTIAL / TIME LIMIT'
    subject = f'[Tiny ImageNet HCSI adaptation] {status}'
    body = (f'Status: {status}\nExit code: {code}\nSuite: {suite}\nLog: {log_path}\n'
        'Five matched tests use the same independent CNN checkpoint: '
        'CNN continuation, CNN with an MLP head, late feature fusion with an MLP, '
        'HCSI cross-stream gating with the same fusion head, and late fusion '
        'with an unpretrained ViT as a matched capacity control. Participating '
        'encoders update jointly; there is no frozen-backbone fusion policy.\n'
        'This is a Tiny ImageNet adaptation of the HCSI mechanism, not an exact '
        'reproduction of the original SBI/deepfake benchmark.\n'
        'Each arm has at most30 epochs. Incomplete arms are partial evidence, not final model scores.\n'
        'The default10-hour wall budget reserves15 minutes for packaging and email; '
        'GPU work is stopped before this reserve. Completed epoch checkpoints remain resumable.\n'
        'The instance remains running on success, failure, timeout and mail failure.\n'
        'Full checkpoints and prediction NPZs remain on the instance disk. '
        'SMTP acceptance does not prove inbox delivery.\n')
    if budget:
        body += (f'Cumulative suite wall budget: {budget["total_seconds"]/3600:g}h; '
            f'previously spent: {budget["prior_spent_seconds"]:g}s; '
            f'delivery reserve this invocation: {budget["delivery_reserve_seconds"]:g}s.\n')
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
                body += 'Retrieve the local compact ZIP; attachment exceeds the mail limit.\n'
        except Exception as exc:
            package_error = type(exc).__name__
            subject += ' / PACKAGE FAILED'
            body += f'Packaging failed ({package_error}); original evidence remains on disk.\n'
        for attempts in range(1, 4):
            # A blocked SMTP call is bounded by the reused helper's30s timeout.
            # Do not begin retries once the reserved overall deadline is spent.
            if budget and time.monotonic()+35 > budget['total_deadline_monotonic']:
                print('Delivery wall budget exhausted; ZIP retained locally and mail remains unconfirmed.', flush=True)
                attempts -= 1
                break
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
        try:
            _write_json(Path(suite)/'delivery_status.json', dict(child_exit_code=code,
                experiment_status=status, time_limit_reached=bool(timed_out),
                email_accepted=sent, email_attempts=attempts, package_error=package_error,
                attachment=str(attachment) if attachment else None,
                delivery_complete=bool(sent and attachment is not None and package_error is None and status == 'SUCCESS'),
                stop_requested=False, instance_policy='keep running',
                smtp_acceptance_is_not_inbox_delivery=True))
        except Exception as exc:
            print(f'Could not record delivery ({type(exc).__name__}); instance remains running.', flush=True)


def main(argv=None):
    import torch
    from hcsi_tiny_experiment import make_plan, create_suite, validate_resume
    started, started_unix = time.monotonic(), time.time()
    args = arguments().parse_args(argv)
    cap_hours, prior_spent = (saved_budget(args.resume, args.max_hours) if args.resume else (args.max_hours, 0.))
    budget = budget_deadlines(started, cap_hours, prior_spent)
    if not args.check_only and budget['remaining_seconds'] <= 0:
        raise RuntimeError('The cumulative suite wall budget is exhausted; no additional training is allowed')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no HCSI training started')
    plan = validate_resume(args.resume) if args.resume else make_plan(args)
    if args.resume:
        recorded = plan.get('epochs', plan.get('fusion_epochs', 30))
        if args.smoke or (args.epochs is not None and args.epochs != recorded):
            raise ValueError('Resume retains its recorded smoke/full protocol and training horizon')
    check_email_ready(args.to)
    if args.check_only:
        print('Preflight OK; no suite, training or email. Instance remains running.', flush=True)
        return
    if time.monotonic() >= budget['child_deadline_monotonic']:
        raise RuntimeError('Preflight exhausted the compute budget; no training started')
    suite = Path(args.resume).resolve() if args.resume else create_suite(args, plan=plan)
    record_budget(suite, budget, invocation_start_unix=started_unix, finished=False)
    log_path = suite/'run.log'
    armed, code, delivered, timed_out = False, 1, False, False
    try:
        with log_path.open('a', encoding='utf-8', buffering=1) as log:
            try:
                deadline_unix = time.time()+budget['child_deadline_monotonic']-time.monotonic()
                print(f'Suite: {suite}\nCumulative wall budget: {cap_hours:g}h; remaining{budget["remaining_seconds"]/3600:.2f}h. VM remains running.', flush=True)
                with subprocess.Popen(child_command(suite, deadline_unix), cwd=ROOT,
                        env=child_environment(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        text=True, errors='replace', bufsize=1,
                        **({'start_new_session':True} if os.name == 'posix' else {})) as child:
                    child._resolver_owned_group = os.name == 'posix'
                    armed = True
                    with ChildWatchdog(child, budget['child_deadline_monotonic']) as watchdog:
                        try:
                            for line in child.stdout:
                                log.write(line)
                                print(line, end='', flush=True)
                            code = child.wait()
                        except BaseException:
                            _stop_child(child)
                            raise
                    timed_out = watchdog.timed_out.is_set()
                    if timed_out:
                        code = 124
            except BaseException:
                log.write(traceback.format_exc())
                raise
            finally:
                log.write(f'\nChild exit code: {code}\n')
                log.flush(); os.fsync(log.fileno())
    finally:
        try:
            if armed:
                delivered = finish(suite, log_path, args.to, code, timed_out=timed_out, budget=budget)
        finally:
            record_budget(suite, budget, invocation_start_unix=started_unix, finished=True)
    if code == 0 and not delivered:
        code = 2
    raise SystemExit(code)


if __name__ == '__main__':
    main()
