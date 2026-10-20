"""Daytime correction-only run: 25 epochs, email analysis archive, NEVER stop VM."""
import argparse
import json
from pathlib import Path
import sys

from run_notified import main as notify_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--to', default='shenwenfei@gmail.com')
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--base-run')
    parser.add_argument('--resume', help='Resume a correction-only suite, preserving its epoch schedule')
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error('--epochs must be positive')
    import torch
    from correction_experiment import ROOT, create_suite
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no training started')
    if args.resume:
        suite = Path(args.resume).resolve()
        plan = json.loads((suite/'suite.json').read_text(encoding='utf-8'))
        if plan['variants'] != ['attention_correction']:
            raise ValueError('This runner resumes correction-only suites, never a ViT job')
        if plan['correction_epochs'] != args.epochs:
            raise ValueError('Resume requires --epochs matching the recorded schedule')
    else:
        suite = create_suite(argparse.Namespace(
            base_run=args.base_run, correction_epochs=args.epochs,
            vit_epochs=100, variants=['attention_correction']))
    print(f'Suite: {suite}\nCorrection only; instance will remain running.', flush=True)
    notify_run(['--to', args.to, '--log', str(suite/'training.log'),
                '--analysis-suite', str(suite), '--', sys.executable, '-u',
                str(ROOT/'correction_experiment.py'), '--resume', str(suite)])


if __name__ == '__main__':
    main()
