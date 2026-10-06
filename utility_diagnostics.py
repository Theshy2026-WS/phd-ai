"""Read-only diagnostics of saved utility predictions and trained checkpoints.

No training or checkpoint modification. Interventions measure frozen-model
sensitivity, NOT the performance of a retrained CNN or a deployable router.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import uuid

import numpy as np
import pandas as pd


PAIRS = (('cnn', 'attention'), ('cnn', 'conv'), ('conv', 'attention'))


def auc_rank(labels, scores):
    """Tie-aware ROC AUC; unavailable when a class is absent."""
    y = np.asarray(labels, dtype=bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if not n1 or not n0:
        return float('nan')
    ranks = pd.Series(np.asarray(scores)).rank(method='average').to_numpy()
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def validate_paired(frame):
    if frame.empty or frame.isna().any().any():
        raise ValueError('Empty or incomplete paired predictions')
    if frame.duplicated(['sample_id', 'occlusion']).any():
        raise ValueError('Duplicate sample/condition')
    if frame.groupby('sample_id').target.nunique().max() != 1:
        raise ValueError('Labels differ across conditions')
    expected = set(frame.sample_id)
    for _, group in frame.groupby('occlusion'):
        if set(group.sample_id) != expected:
            raise ValueError('Conditions contain different samples')
    for variant in ('cnn', 'conv', 'attention'):
        col = frame[f'{variant}_correct']
        if not col.isin([True, False]).all():
            raise ValueError('Correctness must be boolean')
        if not ((frame[f'{variant}_pred'] == frame.target) == col).all():
            raise ValueError('Predictions disagree with correctness')


def pair_stats(group, reference, other):
    a = group[f'{reference}_correct'].astype(bool)
    b = group[f'{other}_correct'].astype(bool)
    corrected, harmed = int((~a & b).sum()), int((a & ~b).sum())
    return dict(count=len(group), reference_accuracy=float(a.mean()),
                other_accuracy=float(b.mean()), corrected=corrected, harmed=harmed,
                accuracy_gain=float(b.mean()-a.mean()),
                mean_delta_nll=float((group[f'{reference}_nll']-group[f'{other}_nll']).mean()),
                oracle_accuracy=float((a | b).mean()))


def paired_tables(frame):
    validate_paired(frame)
    comparisons, bins, signals = [], [], []
    for area, group in frame.groupby('occlusion'):
        # Same fixed CNN entropy bins as the original audit; no learned thresholds.
        entropy_bins = pd.cut(group.cnn_entropy, np.linspace(0, np.log(200)+1e-6, 6),
                              include_lowest=True)
        for reference, other in PAIRS:
            key = dict(occlusion=area, reference=reference, other=other)
            comparisons.append(dict(**key, **pair_stats(group, reference, other)))
            for interval, subset in group.groupby(entropy_bins, observed=True):
                bins.append(dict(**key, cnn_entropy_bin=str(interval),
                                 **pair_stats(subset, reference, other)))
            discordant = group[group[f'{reference}_correct'] != group[f'{other}_correct']]
            for signal, sign in (('entropy', 1), ('confidence', -1), ('uncertainty_margin', -1)):
                signals.append(dict(**key, signal=f'{reference}_{signal}',
                    direction='higher_uncertainty_predicts_other_correct',
                    discordant_count=len(discordant),
                    corrected=int(discordant[f'{other}_correct'].sum()),
                    auc_on_discordant_only=auc_rank(discordant[f'{other}_correct'],
                                                  sign*discordant[f'{reference}_{signal}'])))
    return pd.DataFrame(comparisons), pd.DataFrame(bins), pd.DataFrame(signals)


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def paired_analysis(args):
    tables = paired_tables(pd.read_csv(args.paired))
    out = Path(args.output) / f'paired_diagnostic_{uuid.uuid4().hex[:8]}'
    out.mkdir(parents=True, exist_ok=False)
    for name, table in zip(('pairwise', 'entropy_bins', 'signals'), tables):
        table.to_csv(out/f'{name}.csv', index=False)
    (out/'provenance.json').write_text(json.dumps(dict(
        input=str(Path(args.paired).resolve()), sha256=file_hash(args.paired),
        interpretation='Exploratory single-run associations. AUC uses only label-defined '
        'corrected/harmed samples; not a deployable router score. Entropy associations '
        'can reflect reference-model errors or calibration, not special attention utility. '
        'Compare CNN-conv and repeat seeds. Oracle uses labels.'), indent=2), encoding='utf-8')
    print(tables[0].to_string(index=False))
    print(tables[2].to_string(index=False))
    print(f'Saved: {out}')
    return out


@contextmanager
def intervention(model, mode):
    """Bypass just the spatial mixer or the whole addon, preserving all weights."""
    import torch
    if mode == 'full':
        yield
        return
    if mode == 'mixer_off':
        def hook(module, inputs, output):
            if isinstance(output, tuple):  # MultiheadAttention returns (values, weights).
                return torch.zeros_like(output[0]), output[1]
            return torch.zeros_like(output)
        handle = model.addon.mixer.register_forward_hook(hook)
    elif mode == 'addon_off':
        handle = model.addon.register_forward_hook(lambda module, inputs, output: inputs[0])
    else:
        raise ValueError(mode)
    try:
        yield
    finally:
        handle.remove()


def checkpoint_diagnostic(args):
    import torch
    import timm
    from tqdm import tqdm
    from models.audit_repvit import AuditRepViT
    from datasets.utility_tiny import make_datasets, loader
    from utility_experiment import runtime, occlude, sample_metrics, dump

    audit_dir = Path(args.audit_dir)
    provenance = json.loads((audit_dir/'provenance.json').read_text(encoding='utf-8'))
    config = provenance['config']
    seed = provenance['metadata']['seed']
    if timm.__version__ != provenance['metadata']['timm']:
        raise ValueError('Use the same timm version as training')
    # Never download data during diagnostics.
    config['dataset']['download'] = False
    device, amp, dtype, _ = runtime(config, seed)
    datasets, _, fingerprint = make_datasets(config['dataset'], seed)
    if fingerprint != provenance['metadata']['split_hash']:
        raise ValueError('Dataset split changed')
    original = pd.read_csv(audit_dir/'paired_utility.csv')
    validate_paired(original)
    out = Path(args.output)/f'checkpoint_diagnostic_seed{seed}_{uuid.uuid4().hex[:8]}'
    out.mkdir(parents=True, exist_ok=False)
    summary, replay, updates = [], [], []
    for index, variant in ((1, 'conv'), (2, 'attention')):
        checkpoint_path = Path(provenance['runs'][index])/'best.pt'
        if file_hash(checkpoint_path) != provenance['checkpoint_sha256'][index]:
            raise ValueError('Checkpoint changed since original audit')
        ck = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if ck['metadata']['variant'] != variant or ck['metadata']['split_hash'] != fingerprint:
            raise ValueError('Checkpoint variant/split mismatch')
        model = AuditRepViT(variant, seed)
        for name, param in model.named_parameters():
            if name.startswith('addon.'):
                delta = ck['model'][name].float()-param.detach().float()
                updates.append(dict(variant=variant, parameter=name,
                    initial_l2=float(param.detach().float().norm()),
                    trained_l2=float(ck['model'][name].float().norm()),
                    change_l2=float(delta.norm())))
        model.load_state_dict(ck['model'], strict=True)
        model = model.to(device).eval()
        rows = []
        with torch.inference_mode():
            for area in config['audit']['occlusion']:
                for x, y, ids in tqdm(loader(datasets['audit'], config['dataset'], seed),
                                     desc=f'Diagnostic {variant} area={area}'):
                    x, y = occlude(x, area).to(device), y.to(device)
                    labels = y.cpu().tolist()
                    full = None
                    for mode in ('full', 'mixer_off', 'addon_off'):
                        with intervention(model, mode), torch.autocast(device, dtype=dtype, enabled=amp):
                            logits = model(x)
                        if not torch.isfinite(logits).all():
                            raise ValueError('Non-finite diagnostic logits')
                        if mode == 'full':
                            full = logits.float()
                        m = {k:v.cpu().tolist() for k,v in sample_metrics(logits,y).items()}
                        change = (logits.float().softmax(1)-full.softmax(1)).abs().sum(1).cpu().tolist()
                        for i, sample_id in enumerate(ids):
                            rows.append(dict(sample_id=sample_id, target=labels[i],
                                occlusion=area, mode=mode, probability_l1_vs_full=change[i],
                                **{k:v[i] for k,v in m.items()}))
        frame = pd.DataFrame(rows)
        frame.to_csv(out/f'{variant}_interventions.csv', index=False)
        keys = ['sample_id','target','occlusion']
        full_frame = frame[frame['mode']=='full']
        matched = full_frame.merge(original[keys+[f'{variant}_pred',f'{variant}_nll']],
                                   on=keys, how='outer', indicator=True, validate='one_to_one')
        if not matched['_merge'].eq('both').all():
            raise ValueError('Original audit and replay sample keys differ')
        replay.append(dict(variant=variant, count=len(matched),
            prediction_mismatches=int((matched.pred!=matched[f'{variant}_pred']).sum()),
            max_abs_nll_change=float((matched.nll-matched[f'{variant}_nll']).abs().max())))
        for area, condition in frame.groupby('occlusion'):
            base = condition[condition['mode']=='full'].set_index('sample_id')
            for mode, subset in condition.groupby('mode'):
                s = subset.set_index('sample_id').loc[base.index]
                summary.append(dict(variant=variant, occlusion=area, mode=mode, count=len(s),
                    accuracy=float(s.correct.mean()), nll=float(s.nll.mean()),
                    accuracy_change_vs_full=float(s.correct.mean()-base.correct.mean()),
                    nll_change_vs_full=float(s.nll.mean()-base.nll.mean()),
                    prediction_flip_rate=float((s.pred!=base.pred).mean()),
                    corrected_vs_full=int((s.correct & ~base.correct).sum()),
                    harmed_vs_full=int((~s.correct & base.correct).sum()),
                    mean_probability_l1_vs_full=float(s.probability_l1_vs_full.mean())))
        del model, ck
    pd.DataFrame(summary).to_csv(out/'intervention_summary.csv',index=False)
    pd.DataFrame(replay).to_csv(out/'replay_check.csv',index=False)
    pd.DataFrame(updates).to_csv(out/'parameter_updates.csv',index=False)
    dump(out/'provenance.json', dict(source_audit=str(audit_dir.resolve()),
        checkpoint_sha256=provenance['checkpoint_sha256'], seed=seed, split_hash=fingerprint,
        torch=str(torch.__version__), timm=timm.__version__, device=device, amp=amp,
        interpretation='Frozen-checkpoint interventions, no retraining. mixer_off preserves '
        'positional convolution and FFN. addon_off bypasses the entire addon. Hooks still '
        'compute discarded outputs: do NOT use these modes for speedup measurements. '
        'Performance loss may reflect co-adaptation/distribution shift and is not proof '
        'of special global information or superiority to a retrained conv model. '
        'Inspect replay_check before interpreting; old environment replay can differ.'))
    print(pd.DataFrame(replay).to_string(index=False))
    print(f'Saved: {out}')
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('paired', help='CPU analysis of saved predictions; no training')
    p.add_argument('--paired', required=True)
    p.add_argument('--output', default='runs/tiny_utility')
    c = sub.add_parser('checkpoint', help='Evaluate saved best.pt files; no training')
    c.add_argument('--audit-dir', required=True)
    c.add_argument('--output', default='runs/tiny_utility')
    args = parser.parse_args()
    (paired_analysis if args.command == 'paired' else checkpoint_diagnostic)(args)


if __name__ == '__main__':
    main()
