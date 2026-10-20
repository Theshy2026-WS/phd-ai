"""Multi-head author loss, while leaving existing single-head training intact."""
import torch
from torch.nn import functional as F
from tqdm import tqdm
from resnet_experiment import mix_batch, train_epoch as single_head_train_epoch


def soft_target_loss(outputs, targets):
    heads = outputs if isinstance(outputs, (list, tuple)) else [outputs]
    return sum(-(targets * F.log_softmax(head.float(), dim=1)).sum(1).mean()
               for head in heads) / len(heads)


def train_epoch(model, batches, config, seed, epoch, device, amp, dtype, optimizer, scaler):
    from models.literature_conformer import TinyConformer
    if not isinstance(model, TinyConformer) and not hasattr(model, 'training_loss'):
        return single_head_train_epoch(model, batches, config, seed, epoch, device, amp,
                                       dtype, optimizer, scaler)
    model.train()
    total, count = 0., 0
    for batch_index, (x, y, _) in enumerate(tqdm(batches, desc=f'Train {epoch}', leave=False)):
        x, y = x.to(device), y.to(device)
        x, targets = mix_batch(x, y, config['training'], seed, epoch, batch_index)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device, enabled=amp, dtype=dtype):
            loss = (model.training_loss(x, targets) if hasattr(model, 'training_loss')
                    else soft_target_loss(model(x), targets))
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite training loss')
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['gradient_clip_norm'],
                                      error_if_nonfinite=not amp)
        scaler.step(optimizer)
        scaler.update()
        total += loss.item() * len(y)
        count += len(y)
    return total / count
