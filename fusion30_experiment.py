"""Matched 30-epoch warm-start screens of 8->16->32 spatial feature fusion.

Every arm starts at the same completed CNN8 checkpoint and gets a new optimizer
and a short cosine schedule. This is not scratch training or a resumed optimizer
from the original 100 epochs. Only selection Top-1 chooses checkpoints, including
the unmodified epoch-zero state. Audit and official validation are never read.
"""
import copy
import gc
import json
from pathlib import Path
import re
import time
import traceback

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset
import yaml

from baseline_diagnostics import class_labels, clean_datasets
from correction_experiment import file_hash
from datasets.resnet_tiny import prepare_datasets
from datasets.utility_tiny import loader
from literature_experiment import ROOT, environment, evaluate, new_name, recipe_hash, save_json
from main import build_scheduler
from models.attention_correction import state_digest
from models.spatial_fusion import FUSION_VARIANTS, build_fusion_model, fusion_description
from resnet_experiment import atomic_save
from spatial_experiment import (_diagnose, _probe, _save_history, execution_environment,
                                restore_rng, rng_state, source_hashes as spatial_sources)
from utility_experiment import runtime, sync
from utils.baseline_diagnostics import evaluate_clean
from utils.literature_profiling import estimate_flops
from utils.literature_training import train_epoch


PROTOCOL = 'tiny-cnn8-fusion30-v1'
NOISE = dict(probability=.5, std_rgb=.02, distribution='Gaussian', clip_rgb=[0., 1.],
             scope='training only, after inherited per-image augmentation and before batch mixing')
NEW_SOURCES = ('fusion30_experiment.py', 'models/spatial_fusion.py', 'mechanism_diagnostics.py')


def source_hashes():
    return {**spatial_sources(), **{name: file_hash(ROOT/name) for name in NEW_SOURCES}}


class NoisyTrainingDataset(Dataset):
    """Add mild image-space noise without changing paired augmentation/RNG streams."""
    def __init__(self, base, probability=.5, std_rgb=.02):
        if base.split != 'train':
            raise ValueError('Noise augmentation is restricted to training images')
        if not 0. <= probability <= 1. or not 0. <= std_rgb <= .1:
            raise ValueError('Invalid mild-noise settings')
        self.dataset = base
        self.folder, self.indices, self.seed, self.split = base.folder, base.indices, base.seed, 'train'
        self.probability, self.std_rgb = probability, std_rgb

    @property
    def epoch(self):
        return self.dataset.epoch

    @epoch.setter
    def epoch(self, value):
        self.dataset.epoch = value

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, position):
        x, label, identifier = self.dataset[position]
        generator = torch.Generator().manual_seed(
            self.seed + self.epoch*1000003 + self.indices[position] + 7000000001)
        if torch.rand((), generator=generator).item() < self.probability and self.std_rgb:
            mean = x.new_tensor([.485, .456, .406])[:, None, None]
            std = x.new_tensor([.229, .224, .225])[:, None, None]
            rgb = x*std + mean
            noise = torch.randn(x.shape, generator=generator, dtype=x.dtype)*self.std_rgb
            x = ((rgb+noise).clamp(0., 1.)-mean)/std
        return x, label, identifier


def training_config(parent_config, config, epochs):
    result = copy.deepcopy(parent_config if config is None else config)
    result['training'].update(epochs=epochs, early_stopping_patience=0,
        probe_per_class=20, learning_rate=5e-5)
    result['training']['scheduler'].update(name='cosine', warmup_epochs=min(1, epochs-1), min_lr=1e-6)
    return result


def _new_diagnostic_dir(run, stem):
    destination = run/stem
    return run/new_name(stem+'_retry') if destination.exists() else destination


def _correction_norms(model):
    head = getattr(model, 'correction', None)
    return dict(correction_weight_l2=0. if head is None else float(head.weight.detach().float().norm()),
                correction_bias_l2=0. if head is None else float(head.bias.detach().float().norm()))


def _save_best(run, state):
    # Keep a versioned candidate until last.pt commits the epoch. If power is
    # lost after writing a new best but before last.pt, the previous committed
    # version still exists and resume can restore it exactly.
    atomic_save(run/f'best_epoch{state["epoch"]}.pt', state)
    atomic_save(run/'best.pt', state)


def _clean_best_versions(run, keep_epoch):
    for candidate in run.glob('best_epoch*.pt'):
        match = re.fullmatch(r'best_epoch(\d+)\.pt', candidate.name)
        if match and int(match[1]) != keep_epoch:
            candidate.unlink()


def _committed_best(run, plan, variant, history, best):
    expected_epoch = next(row['epoch'] for row in history if row['selection_accuracy'] == best)
    selected = None
    if (run/'best.pt').exists():
        selected = torch.load(run/'best.pt', map_location='cpu', weights_only=False)
        _checkpoint_matches(selected, plan, variant)
    if selected is None or selected['epoch'] != expected_epoch or selected['selection']['accuracy'] != best:
        version = run/f'best_epoch{expected_epoch}.pt'
        if not version.is_file():
            raise ValueError('Fusion committed best version is missing after an interrupted checkpoint write')
        selected = torch.load(version, map_location='cpu', weights_only=False)
        _checkpoint_matches(selected, plan, variant)
        if selected['epoch'] != expected_epoch or selected['selection']['accuracy'] != best:
            raise ValueError('Fusion best checkpoint does not match committed history')
        atomic_save(run/'best.pt', selected)
    _clean_best_versions(run, expected_epoch)
    return selected


def _checkpoint_matches(state, plan, variant):
    meta = state['metadata']
    expected = dict(protocol=PROTOCOL, variant=variant, seed=plan['seed'],
        source_hashes=plan['source_hashes'], split_hash=plan['split_hash'],
        augmentation_epoch_offset=plan.get('augmentation_epoch_offset', 0),
        parent_checkpoint_sha256=plan['parent_checkpoint_sha256'],
        recipe_hash=recipe_hash(plan['config']), environment=plan['execution_environment'])
    if any(meta.get(key) != value for key, value in expected.items()):
        raise ValueError('Fusion checkpoint differs from saved source, parent, split, recipe or environment')
    if recipe_hash(state['config']) != expected['recipe_hash']:
        raise ValueError('Fusion checkpoint configuration hash is inconsistent')
    if not 0 <= state['epoch'] <= plan['config']['training']['epochs']:
        raise ValueError('Fusion checkpoint epoch is outside the recorded horizon')


def _validate_history(state):
    history = state['history']
    if ([row['epoch'] for row in history] != list(range(state['epoch']+1))
            or state['best'] != max(row['selection_accuracy'] for row in history)):
        raise ValueError('Fusion checkpoint history/epoch/best chronology is inconsistent')


def _metadata(plan, variant, model):
    return dict(protocol=PROTOCOL, variant=variant, seed=plan['seed'],
        source_hashes=plan['source_hashes'], split_hash=plan['split_hash'],
        augmentation_epoch_offset=plan.get('augmentation_epoch_offset', 0),
        parent_checkpoint_sha256=plan['parent_checkpoint_sha256'],
        parent_run=plan['parent_run'], parent_best_epoch=plan['parent_best_epoch'],
        recipe_hash=recipe_hash(plan['config']), environment=plan['execution_environment'],
        architecture=fusion_description(variant),
        params=sum(p.numel() for p in model.parameters()),
        trainable_params=sum(p.numel() for p in model.parameters() if p.requires_grad),
        initial_state_sha256=state_digest(model),
        encoder_initial_state_sha256=state_digest(getattr(model, 'encoder', model)),
        fresh_optimizer=True, parent_optimizer_resumed=False,
        parent_checkpoint_used_for_initialization=True, encoder_frozen=False,
        noise=NOISE if variant == 'unet32_noise' else None)


def _train_job(plan, variant, run, parent_state):
    """CPU-testable inner job; public suite entry point requires CUDA."""
    run = Path(run)
    config, seed = copy.deepcopy(plan['config']), plan['seed']
    began = time.perf_counter()
    device, amp, dtype, scaler = runtime(config, seed)
    if environment(device, amp, dtype) != plan['execution_environment']:
        raise ValueError('Fusion jobs require the recorded GPU/software/precision environment')
    supplied, manifest, fingerprint = prepare_datasets(config, seed)
    if fingerprint != plan['split_hash']:
        raise ValueError('Fusion split differs from the CNN8 reference')
    if variant == 'unet32_noise':
        supplied['train'] = NoisyTrainingDataset(supplied['train'], NOISE['probability'], NOISE['std_rgb'])
    probe = supplied['train_probe']
    probe_ids = [Path(probe.folder.samples[i][0]).relative_to(probe.folder.root).as_posix()
                 for i in probe.indices]
    model = build_fusion_model(variant, parent_state, seed).to(device)
    metadata = _metadata(plan, variant, model)
    if (run/'summary.json').exists():
        summary = json.loads((run/'summary.json').read_text(encoding='utf-8'))
        if any(summary.get(key) != value for key, value in metadata.items()):
            raise ValueError('Completed fusion run differs from the requested plan')
        return summary
    decay, no_decay = [], []
    for parameter in model.parameters():
        (no_decay if parameter.ndim <= 1 else decay).append(parameter)
    optimizer = torch.optim.AdamW([
        dict(params=decay, weight_decay=config['training']['weight_decay']),
        dict(params=no_decay, weight_decay=0.)], lr=config['training']['learning_rate'])
    scheduler = build_scheduler(optimizer, config['training'])
    if (run/'last.pt').exists():
        saved = torch.load(run/'last.pt', map_location='cpu', weights_only=False)
        _checkpoint_matches(saved, plan, variant)
        _validate_history(saved)
        if saved['metadata'] != metadata:
            raise ValueError('Fusion initialized state differs from saved checkpoint')
        model.load_state_dict(saved['model'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        scaler.load_state_dict(saved['scaler'])
        restore_rng(saved['rng'], device)
        history, best, elapsed = saved['history'], saved['best'], saved['elapsed_wall_seconds']
        del saved
        selected_state = _committed_best(run, plan, variant, history, best)
        del selected_state
    else:
        if run.exists():
            names = {path.name for path in run.iterdir()}
            allowed = {'metadata.json', 'config.yaml', 'split_manifest.json', 'train_probe_ids.json',
                       'best.pt', 'best.tmp', 'last.tmp', 'epochs.csv.tmp'}
            if (not (run/'metadata.json').exists() or
                    any(name not in allowed and not name.startswith('initial_selection')
                        and not re.fullmatch(r'best_epoch0\.(?:pt|tmp)', name) for name in names)
                    or json.loads((run/'metadata.json').read_text(encoding='utf-8')) != metadata):
                raise ValueError('Refusing to overwrite an unrelated/inconsistent unstarted fusion directory')
            if (run/'best.pt').exists():
                partial = torch.load(run/'best.pt', map_location='cpu', weights_only=False)
                _checkpoint_matches(partial, plan, variant)
                if partial['epoch'] != 0 or partial['metadata'] != metadata:
                    raise ValueError('Unstarted fusion checkpoint is not the declared epoch-zero model')
        else:
            run.mkdir(parents=True)
        save_json(run/'metadata.json', metadata)
        save_json(run/'split_manifest.json', manifest)
        save_json(run/'train_probe_ids.json', probe_ids)
        (run/'config.yaml').write_text(yaml.safe_dump(config), encoding='utf-8')
        # The initial model is a legitimate checkpoint candidate. This avoids
        # disguising degradation by forcing selection of a fine-tuned epoch.
        initial_dir = _new_diagnostic_dir(run, 'initial_selection')
        before = rng_state(device)
        try:
            selected = evaluate_clean(model.float().eval(), supplied['selection'], device=device,
                batch_size=config['dataset']['batch_size'], num_workers=config['dataset']['num_workers'],
                output_dir=initial_dir, class_names=class_labels(supplied['train_probe']), example_count=0)
        finally:
            restore_rng(before, device)
        selected = {key: selected[key] for key in ('accuracy', 'top5_accuracy', 'nll')}
        probed = _probe(model, probe, config, seed, device, dtype)
        row = dict(epoch=0, augmentation_epoch=plan.get('augmentation_epoch_offset', 0),
            lr=optimizer.param_groups[0]['lr'], train_seconds=0.,
            peak_training_memory_bytes=0, initial_selection_directory=initial_dir.relative_to(run).as_posix(),
            **{f'selection_{key}': value for key, value in selected.items()},
            **{f'train_probe_{key}': value for key, value in probed.items()},
            **_correction_norms(model),
            clean_accuracy_gap_pp=100*(probed['accuracy']-selected['accuracy']))
        history, best, elapsed = [row], selected['accuracy'], 0.
        _save_best(run, dict(model=model.state_dict(), epoch=0, selection=selected,
                            metadata=metadata, config=config))
        atomic_save(run/'last.pt', dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(), scaler=scaler.state_dict(), history=history, best=best,
            config=config, metadata=metadata, epoch=0, elapsed_wall_seconds=0., rng=rng_state(device)))
        _save_history(run, history)
    print(f'Fusion run: {run}; {variant}; additional epochs={config["training"]["epochs"]}; '
          f'next={history[-1]["epoch"]+1}', flush=True)
    for epoch in range(history[-1]['epoch']+1, config['training']['epochs']+1):
        sync(device)
        if device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        lr = optimizer.param_groups[0]['lr']
        data_epoch = plan.get('augmentation_epoch_offset', 0) + epoch
        loss = train_epoch(model, loader(supplied['train'], config['dataset'], seed, data_epoch),
            config, seed, data_epoch, device, amp, dtype, optimizer, scaler)
        sync(device)
        train_seconds = time.perf_counter()-started
        peak_memory = torch.cuda.max_memory_allocated() if device == 'cuda' else 0
        selected = evaluate(model, loader(supplied['selection'], config['dataset'], seed), device, False, dtype)
        row = dict(epoch=epoch, augmentation_epoch=data_epoch,
            lr=lr, train_soft_target_loss=loss, train_seconds=train_seconds,
            peak_training_memory_bytes=peak_memory,
            **_correction_norms(model),
            **{f'selection_{key}': value for key, value in selected.items()})
        if epoch == 1 or epoch % plan['probe_every'] == 0 or epoch == config['training']['epochs']:
            probed = _probe(model, probe, config, seed, device, dtype)
            row.update({f'train_probe_{key}': value for key, value in probed.items()})
            row['clean_accuracy_gap_pp'] = 100*(probed['accuracy']-selected['accuracy'])
        if selected['accuracy'] > best:
            best = selected['accuracy']
            _save_best(run, dict(model=model.state_dict(), epoch=epoch, selection=selected,
                                metadata=metadata, config=config))
        history.append(row)
        scheduler.step()
        atomic_save(run/'last.pt', dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(), scaler=scaler.state_dict(), history=history, best=best,
            config=config, metadata=metadata, epoch=epoch,
            elapsed_wall_seconds=elapsed+time.perf_counter()-began, rng=rng_state(device)))
        _clean_best_versions(run, next(item['epoch'] for item in history if item['selection_accuracy'] == best))
        _save_history(run, history)
        print(f'{variant} epoch {epoch}/{config["training"]["epochs"]}: '
            f'selection={100*selected["accuracy"]:.2f}% NLL={selected["nll"]:.5f}'
            + (f' clean-probe={100*row["train_probe_accuracy"]:.2f}%'
               if 'train_probe_accuracy' in row else ''), flush=True)
    _save_history(run, history)
    model.zero_grad(set_to_none=True)
    del optimizer, scheduler, scaler
    best_state = torch.load(run/'best.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(best_state['model'], strict=True)
    best_epoch, best_selection = best_state['epoch'], best_state['selection']
    del best_state
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()
    diagnostic_dir, scores = _diagnose(model, supplied, config, seed, run, plan['examples'])
    flops = estimate_flops(model, device)
    summary = dict(**metadata, completed_epochs=history[-1]['epoch'], best_epoch=best_epoch,
        best_selection=best_selection, clean_scores=scores,
        last_epoch_selection={key: history[-1][f'selection_{key}']
                              for key in ('accuracy', 'top5_accuracy', 'nll')},
        initial_selection=history[0], diagnostic_directory=diagnostic_dir.relative_to(run).as_posix(),
        forward_flops_estimate=flops,
        total_train_seconds=sum(row['train_seconds'] for row in history),
        peak_training_memory_bytes=max(row['peak_training_memory_bytes'] for row in history),
        total_experiment_seconds=elapsed+time.perf_counter()-began,
        official_validation_evaluated=False, audit_partition_evaluated=False)
    save_json(run/'summary.json', summary)
    return summary


def paired_predictions(left_path, right_path, output):
    with np.load(left_path, allow_pickle=False) as left, np.load(right_path, allow_pickle=False) as right:
        for key in ('image_ids', 'targets', 'class_names'):
            if not np.array_equal(left[key], right[key]):
                raise ValueError('Fusion pairing requires identical ordered IDs, labels and class mapping')
        labels, ids = left['targets'], left['image_ids']
        a, b = left['predicted_labels'], right['predicted_labels']
        p, q = left['probabilities'], right['probabilities']
        if (labels.ndim != 1 or not len(labels) or len(set(ids.tolist())) != len(ids)
                or a.shape != labels.shape or b.shape != labels.shape or p.shape != q.shape
                or p.ndim != 2 or p.shape[0] != len(labels)
                or not np.isfinite(p).all() or not np.isfinite(q).all()
                or np.any(p < 0) or np.any(q < 0)
                or not np.allclose(p.sum(1), 1., atol=1e-5)
                or not np.allclose(q.sum(1), 1., atol=1e-5)):
            raise ValueError('Invalid fusion paired prediction arrays')
        ca, cb = a == labels, b == labels
        corrected, harmed = ~ca & cb, ca & ~cb
        ix = np.arange(len(labels))
        tiny = float(np.nextafter(np.float32(0), np.float32(1)))
        loss_a = -np.log(np.maximum(p[ix, labels].astype(np.float64), tiny))
        loss_b = -np.log(np.maximum(q[ix, labels].astype(np.float64), tiny))
        frame = pd.DataFrame(dict(image_id=ids, target=labels, prediction_reference=a,
            prediction_candidate=b, correct_reference=ca, correct_candidate=cb,
            corrected=corrected, harmed=harmed, nll_reference=loss_a, nll_candidate=loss_b))
        frame.to_csv(output, index=False)
        return dict(n=len(labels), corrected=int(corrected.sum()), harmed=int(harmed.sum()),
            both_correct=int((ca & cb).sum()), both_wrong=int((~ca & ~cb).sum()),
            delta_accuracy_pp=100*float((cb.astype(float)-ca.astype(float)).mean()),
            mean_nll_improvement=float((loss_a-loss_b).mean()))


def _curve_png(output, histories):
    image = Image.new('RGB', (1120, 650), 'white')
    draw = ImageDraw.Draw(image)
    draw.text((35, 15), 'CNN8 warm-start screen: selection Top-1 (solid), clean train probe (dashed)', fill='black')
    x0, x1, y0, y1 = 75, 1080, 70, 575
    max_epoch = max(int(frame['epoch'].max()) for frame in histories.values())
    for percent in range(0, 101, 10):
        y = y1-(y1-y0)*percent/100
        draw.line((x0, y, x1, y), fill='#dddddd')
        draw.text((30, y-7), str(percent), fill='black')
    for i, (variant, frame) in enumerate(histories.items()):
        color = ('#2266aa', '#cc5522', '#338844')[i]
        draw.text((80+i*335, 615), variant, fill=color)
        for column, dashed in (('selection_accuracy', False), ('train_probe_accuracy', True)):
            valid = frame[['epoch', column]].dropna()
            points = [(x0+(x1-x0)*float(e)/max(1, max_epoch), y1-(y1-y0)*float(v))
                      for e, v in valid.itertuples(index=False, name=None)]
            for a, b in zip(points, points[1:]):
                if dashed:
                    for j in range(0, 10, 2):
                        p = tuple(a[k]+(b[k]-a[k])*j/10 for k in (0, 1))
                        q = tuple(a[k]+(b[k]-a[k])*(j+1)/10 for k in (0, 1))
                        draw.line((*p, *q), fill=color, width=2)
                else:
                    draw.line((*a, *b), fill=color, width=3)
    draw.text((500, 590), 'Additional fine-tuning epoch (0 = original parent)', fill='black')
    image.save(output/'learning_curves.png')


def _selection_curve_png(output, histories, column, filename, title):
    image = Image.new('RGB', (1120, 650), 'white')
    draw = ImageDraw.Draw(image)
    draw.text((35, 15), title, fill='black')
    scale = 100. if column.endswith('accuracy') else 1.
    values = [float(value)*scale for frame in histories.values() for value in frame[column].dropna()]
    padding = max((max(values)-min(values))*.12, .1 if scale == 100. else .005)
    lower, upper = min(values)-padding, max(values)+padding
    x0, x1, y0, y1 = 90, 1080, 70, 575
    max_epoch = max(int(frame['epoch'].max()) for frame in histories.values())
    for i in range(6):
        value = lower+(upper-lower)*i/5
        y = y1-(y1-y0)*i/5
        draw.line((x0, y, x1, y), fill='#dddddd')
        draw.text((15, y-7), f'{value:.3f}', fill='black')
    for i, (variant, frame) in enumerate(histories.items()):
        color = ('#2266aa', '#cc5522', '#338844')[i]
        draw.text((80+i*335, 615), variant, fill=color)
        points = [(x0+(x1-x0)*float(e)/max(1, max_epoch),
                   y1-(y1-y0)*(float(v)*scale-lower)/(upper-lower))
                  for e, v in frame[['epoch', column]].dropna().itertuples(index=False, name=None)]
        if len(points) > 1:
            draw.line(points, fill=color, width=3)
        else:
            x, y = points[0]
            draw.ellipse((x-3, y-3, x+3, y+3), fill=color)
    for epoch in sorted(set((0, max_epoch, *range(5, max_epoch, 5)))):
        x = x0+(x1-x0)*epoch/max(1, max_epoch)
        draw.text((x-5, 578), str(epoch), fill='black')
    draw.text((430, 595), 'Additional fine-tuning epoch (0 = original parent)', fill='black')
    image.save(output/filename)


def comparison(output, plan):
    output = Path(output)
    summaries, rows, histories = {}, [], {}
    for variant in FUSION_VARIANTS:
        run = output/variant
        if not (run/'summary.json').exists():
            continue
        summary = json.loads((run/'summary.json').read_text(encoding='utf-8'))
        summaries[variant] = summary
        histories[variant] = pd.read_csv(run/'epochs.csv')
        train, selection = summary['clean_scores']['train_clean'], summary['clean_scores']['selection']
        rows.append(dict(variant=variant, additional_epochs=summary['completed_epochs'],
            best_additional_epoch=summary['best_epoch'], train_clean_top1_pct=100*train['accuracy'],
            selection_top1_pct=100*selection['accuracy'], selection_top5_pct=100*selection['top5_accuracy'],
            gap_pp=100*(train['accuracy']-selection['accuracy']), selection_nll=selection['nll'],
            last_epoch_selection_top1_pct=100*summary['last_epoch_selection']['accuracy'],
            last_epoch_selection_nll=summary['last_epoch_selection']['nll'],
            params=summary['params'], trainable_params=summary['trainable_params'],
            gflops=summary['forward_flops_estimate']/1e9,
            train_minutes=summary['total_train_seconds']/60,
            peak_training_memory_bytes=summary['peak_training_memory_bytes']))
    if not rows:
        return {}
    frame = pd.DataFrame(rows)
    frame.to_csv(output/'comparison.csv', index=False)
    _curve_png(output, histories)
    _selection_curve_png(output, histories, 'selection_accuracy', 'selection_top1_zoom.png',
                         'Selection Top-1 (%): zoomed vertical axis, all epochs')
    _selection_curve_png(output, histories, 'selection_nll', 'selection_nll.png',
                         'Selection NLL: natural-log cross entropy, all epochs (lower is better)')
    paired = {}
    if 'cnn8_control' in summaries:
        control = summaries['cnn8_control']
        parent_predictions = output/'cnn8_control'/control['initial_selection']['initial_selection_directory']/'predictions.npz'
        control_predictions = output/'cnn8_control'/control['diagnostic_directory']/'selection/predictions.npz'
        for variant, summary in summaries.items():
            predictions = output/variant/summary['diagnostic_directory']/'selection/predictions.npz'
            paired[f'{variant}_vs_parent'] = paired_predictions(
                parent_predictions, predictions, output/f'paired_{variant}_vs_parent.csv')
            if variant != 'cnn8_control':
                paired[f'{variant}_vs_control'] = paired_predictions(
                    control_predictions, predictions, output/f'paired_{variant}_vs_control.csv')
    if 'unet32' in summaries and 'unet32_noise' in summaries:
        clean_summary, noise_summary = summaries['unet32'], summaries['unet32_noise']
        paired['unet32_noise_vs_unet32'] = paired_predictions(
            output/'unet32'/clean_summary['diagnostic_directory']/'selection/predictions.npz',
            output/'unet32_noise'/noise_summary['diagnostic_directory']/'selection/predictions.npz',
            output/'paired_unet32_noise_vs_unet32.csv')
    save_json(output/'paired_summary.json', paired)
    text = [
        f'# CNN8: {plan["config"]["training"]["epochs"]}-epoch spatial fusion and noise screen', '',
        'All arms warm-start the same original CNN8 best checkpoint. All encoder parameters are trainable. '
        'Each arm uses a fresh AdamW optimizer, the same recorded epoch budget and the inherited augmentation recipe. '
        'Epoch 0 is eligible for selection; strict selection Top-1 improvements retain the earlier epoch on ties.', '',
        f'Training augmentation and batch-mixing streams use epoch offset '
        f'{plan.get("augmentation_epoch_offset", 0)} plus the additional epoch, providing fresh common draws '
        'without changing the seed. The optimizer/scheduler count additional epochs only.', '',
        '| Arm | Best added epoch | Clean train Top-1 | Best selection Top-1 | Last selection Top-1 | Gap pp | Best selection NLL |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        text.append(f'| {row["variant"]} | {row["best_additional_epoch"]} | '
            f'{row["train_clean_top1_pct"]:.3f}% | {row["selection_top1_pct"]:.3f}% | '
            f'{row["last_epoch_selection_top1_pct"]:.3f}% | '
            f'{row["gap_pp"]:.3f} | {row["selection_nll"]:.6f} |')
    text += ['', '## Interpretation limits', '',
        '- Compare fusion against the continued-training control, then noise against otherwise identical fusion.',
        '- Noise is an augmentation hypothesis, not an assumption that more noise improves generalization. '
        'The fixed setting is probability 0.5 and RGB standard deviation 0.02; clean evaluation is unchanged.',
        '- A selected epoch of zero means no fine-tuned epoch exceeded the parent selection Top-1.',
        '- A smaller gap alone is insufficient: a useful result must preserve or improve unseen-image performance.',
        '- This bounded single-seed screen does not isolate skips from decoder capacity, establish novelty, '
        'or give a final test estimate. The audit and official validation sets were not accessed.', '',
        '## Paired selection changes', '',
        '| Comparison | Corrected | Harmed | Delta Top-1 pp | Mean NLL improvement |',
        '|---|---:|---:|---:|---:|']
    for name, item in paired.items():
        text.append(f'| {name} | {item["corrected"]} | {item["harmed"]} | '
                    f'{item["delta_accuracy_pp"]:+.3f} | {item["mean_nll_improvement"]:+.6f} |')
    text += ['', 'See learning_curves.png, selection_top1_zoom.png, selection_nll.png, per-arm epochs.csv, '
        'diagnostics/class_comparison.csv and '
        'diagnostics/selection/galleries for detailed results. Full probability NPZ and checkpoints stay on the VM.', '']
    (output/'report.md').write_text('\n'.join(text), encoding='utf-8')
    return dict(rows=rows, paired=paired)


def train_fusion_suite(parent_run, output_dir, config=None, epochs=30, probe_every=5,
                       examples=12, smoke=False, resume=True):
    from mechanism_diagnostics import load_spatial_parent
    if not 1 <= epochs <= 30 or probe_every < 1 or examples < 0:
        raise ValueError('Need 1..30 epochs, positive probe interval and nonnegative examples')
    parent_run, parent_summary, parent_state, parent_hash = load_spatial_parent(parent_run)
    output = Path(output_dir).resolve()
    if output == parent_run or parent_run in output.parents:
        raise ValueError('Fusion output must be outside the original training run')
    seed = parent_summary['seed']
    config = training_config(parent_state['config'], config, 1 if smoke else epochs)
    _, _, _, fingerprint, _ = clean_datasets(config, seed, parent_summary['split_hash'], 0)
    plan = dict(protocol=PROTOCOL, config=config, source_hashes=source_hashes(),
        execution_environment=execution_environment(config, seed), seed=seed,
        parent_run=str(parent_run), parent_checkpoint_sha256=parent_hash,
        parent_best_epoch=parent_summary['best_epoch'], split_hash=fingerprint,
        augmentation_epoch_offset=parent_summary['completed_epochs'],
        variants=list(FUSION_VARIANTS), probe_every=probe_every, examples=examples,
        noise=NOISE, smoke=bool(smoke), warm_start=True, fresh_optimizer=True,
        audit_partition_evaluated=False, official_validation_evaluated=False)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; refusing fusion training on CPU')
    if (output/'suite.json').exists():
        recorded = json.loads((output/'suite.json').read_text(encoding='utf-8'))
        if not resume or recorded != plan:
            raise ValueError('Fusion resume requires identical source, parent, split, config and environment')
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError('Refusing to overwrite an unrelated fusion output directory')
        output.mkdir(parents=True, exist_ok=True)
        save_json(output/'suite.json', plan)
    initial_digests, initialized_models, failures = [], {}, {}
    for variant in FUSION_VARIANTS:
        try:
            summary = _train_job(plan, variant, output/variant, parent_state['model'])
            initial_digests.append(summary['encoder_initial_state_sha256'])
            initialized_models[variant] = summary['initial_state_sha256']
            if len(set(initial_digests)) != 1:
                raise RuntimeError('Fusion arms did not start at identical encoder weights/buffers')
            if ('unet32' in initialized_models and 'unet32_noise' in initialized_models
                    and initialized_models['unet32'] != initialized_models['unet32_noise']):
                raise RuntimeError('Noise and standard U-shaped arms had different added initialization')
        except Exception as exc:
            failures[variant] = dict(status='failed', error_type=type(exc).__name__, traceback=traceback.format_exc())
            save_json(output/'job_failures.json', failures)
            print(f'Fusion arm {variant} failed ({type(exc).__name__}); attempting remaining arms.', flush=True)
        try:
            comparison(output, plan)
            failures.pop('comparison', None)
        except Exception as exc:
            failures['comparison'] = dict(status='failed', error_type=type(exc).__name__, traceback=traceback.format_exc())
            save_json(output/'job_failures.json', failures)
        gc.collect()
        torch.cuda.empty_cache()
    if file_hash(parent_run/'best.pt') != parent_hash:
        raise RuntimeError('Original CNN8 checkpoint changed during the fusion screen')
    save_json(output/'job_failures.json', failures)
    result = dict(status='failed' if failures else 'complete', protocol=PROTOCOL,
        runs=len(initialized_models), failures=failures,
        additional_epochs_per_arm=config['training']['epochs'], smoke=bool(smoke),
        parent_checkpoint_sha256=parent_hash, audit_partition_evaluated=False,
        official_validation_evaluated=False)
    save_json(output/'completion.json', result)
    if failures:
        raise RuntimeError('One or more fusion arms/analyses failed; partial outputs and job_failures.json are preserved')
    return result
