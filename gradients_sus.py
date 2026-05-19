"""
gradients_sus.py — brew_poison pipeline with per-sample gradient analysis at every training iteration.

For each training batch this script computes the per-sample gradient matrix G (shape [N, D]),
where row i is the flattened gradient of loss(f(x_i), y_i) w.r.t. selected model parameters,
then computes the SVD of G and saves both results.

Run:
    python gradients_sus.py                         # CIFAR10, ResNet18, default settings
    python gradients_sus.py --dryrun                # fast smoke test
    python gradients_sus.py --grad_layers last      # last linear layer only (default, manageable size)
    python gradients_sus.py --grad_layers all       # full gradient matrix (very large, use with care)
    python gradients_sus.py --grad_dtype float32    # full-precision storage (default is float16)
    python gradients_sus.py --no_vmap               # force loop-based per-sample grads

All other flags are identical to brew_poison.py (see forest/options.py).

Output files (written to --grad_save_dir, default "gradient_logs/"):
    gradients.pt   — torch.save'd list of dicts, one entry per batch across all epochs:
                     {'epoch': int, 'batch': int, 'is_poisoned': bool,
                      'poisoned_count': int, 'G': Tensor[N, D]}
    SVD_matrix.pt  — torch.save'd list of dicts, one entry per batch across all epochs:
                     {'epoch': int, 'batch': int, 'is_poisoned': bool,
                      'poisoned_count': int, 'singular_values': Tensor[min(N, D)]}

Load example:
    import torch
    grads   = torch.load('gradient_logs/gradients.pt')
    svd     = torch.load('gradient_logs/SVD_matrix.pt')
    G_first = grads[0]['G']                 # shape [N, D]
    S_first = svd[0]['singular_values']     # shape [min(N, D)]
    print(grads[0]['phase'], grads[0]['is_poisoned'], grads[0]['epoch'], grads[0]['batch'])

    # Select only poisoned batches from the validation phase:
    poisoned = [r for r in svd if r['phase'] == 'validate' and r['is_poisoned']]
    clean    = [r for r in svd if r['phase'] == 'validate' and not r['is_poisoned']]

svd = torch.load('gradient_logs/SVD_matrix.pt')
poisoned = [r for r in svd if r['phase'] == 'validate' and r['is_poisoned']]
clean    = [r for r in svd if r['phase'] == 'validate' and not r['is_poisoned']]

Notes on storage size:
    With --grad_layers last (default), D equals the parameter count of the final linear
    layer (e.g., 640 for ResNet18 on CIFAR10: 512*10 weight + 10 bias).
    With --grad_layers all, D equals the total number of trainable parameters (~11 M for
    ResNet18), which produces very large files (~hundreds of GB for a full run).
    Use --grad_dtype float16 (default) to halve storage at a small precision cost.
"""

import argparse
import os
import datetime
import time
import types
from collections import defaultdict

import torch
import numpy as np

import forest
from forest.victims.training import run_validation, check_targets
from forest.victims.utils import print_and_save_stats, pgd_step
from forest.consts import NON_BLOCKING, BENCHMARK

torch.backends.cudnn.benchmark = BENCHMARK
torch.multiprocessing.set_sharing_strategy(forest.consts.SHARING_STRATEGY)


try:
    from torch.func import functional_call, vmap, grad as func_grad
    _HAS_VMAP = True
except ImportError:
    _HAS_VMAP = False



def _unwrap_model(model):
    """Strip DataParallel wrapper if present so named_parameters are consistent."""
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def _get_target_params(model, layers_mode):
    """Return [(name, param)] for the parameters to include in the gradient matrix G.

    layers_mode='last' : parameters of the last top-level child module that owns
                         at least one trainable parameter (e.g. the final linear layer).
    layers_mode='all'  : every trainable parameter in the model.
    """
    raw = _unwrap_model(model)
    all_params = [(n, p) for n, p in raw.named_parameters() if p.requires_grad]
    if not all_params:
        return []

    if layers_mode == 'all':
        return all_params

    # 'last': walk top-level children in reverse and pick the first with parameters
    children = list(raw.named_children())
    for child_name, child_module in reversed(children):
        child_params = [
            (f'{child_name}.{n}' if n else child_name, p)
            for n, p in child_module.named_parameters()
            if p.requires_grad
        ]
        if child_params:
            return child_params

    # Fallback: just the very last parameter
    return [all_params[-1]]


# ────────────────────────────────────────────────────────────────────────────
# Per-sample gradient computation (vmap path)
# ────────────────────────────────────────────────────────────────────────────

def _per_sample_grads_vmap(model, inputs, labels, criterion, target_param_names):
    """Compute per-sample gradients using torch.func.vmap (requires PyTorch >= 2.0).

    The model must be in eval mode before calling this function so that stateful
    operations (BatchNorm) use running estimates rather than batch statistics,
    which are undefined for the effective batch-size-1 used internally by vmap.

    Returns G of shape [N, D] where D = total number of selected parameter elements.
    """
    raw = _unwrap_model(model)
    all_params = {n: p.detach() for n, p in raw.named_parameters()}
    buffers    = {n: b          for n, b in raw.named_buffers()}

    tgt_set   = set(target_param_names)
    tgt_params = {n: all_params[n] for n in target_param_names}
    oth_params = {n: p for n, p in all_params.items() if n not in tgt_set}

    def _loss_single(tgt, oth, buf, x, y):
        merged = {**tgt, **oth}
        # x is [C,H,W]; unsqueeze to [1,C,H,W] for functional_call
        out = functional_call(raw, (merged, buf), x.unsqueeze(0))
        return criterion(out, y.unsqueeze(0))

    grad_fn  = func_grad(_loss_single, argnums=0)
    vmap_fn  = vmap(grad_fn, in_dims=(None, None, None, 0, 0))

    per_sample = vmap_fn(tgt_params, oth_params, buffers, inputs, labels)
    # per_sample: dict  param_name -> Tensor[N, *param_shape]
    G = torch.cat([v.flatten(start_dim=1) for v in per_sample.values()], dim=1)
    return G


# ────────────────────────────────────────────────────────────────────────────
# Per-sample gradient computation (loop fallback)
# ────────────────────────────────────────────────────────────────────────────

def _per_sample_grads_loop(model, inputs, labels, criterion, target_params):
    """Compute per-sample gradients by looping over each sample individually.

    Uses torch.autograd.grad so that .grad attributes on model parameters are
    never modified and the training backward pass is unaffected.
    The model must be in eval mode before calling this function (same reason as
    the vmap path: BatchNorm is ill-conditioned at batch-size 1 in train mode).

    Returns G of shape [N, D].
    """
    params_list = [p for _, p in target_params]
    grads = []
    for i in range(inputs.size(0)):
        out_i  = model(inputs[i : i + 1])
        loss_i = criterion(out_i, labels[i : i + 1])
        g_i    = torch.autograd.grad(loss_i, params_list, retain_graph=False)
        grads.append(torch.cat([g.detach().flatten() for g in g_i]))
    return torch.stack(grads)  # [N, D]


# ────────────────────────────────────────────────────────────────────────────
# Dispatcher
# ────────────────────────────────────────────────────────────────────────────

def compute_per_sample_grads(model, inputs, labels, criterion, grad_config):
    """Compute the per-sample gradient matrix G for the given batch.

    Temporarily sets the model to eval mode so that BatchNorm is stable,
    then restores the original training flag.  The model's .grad buffers
    are never modified.

    Args:
        model       : the nn.Module (may be DataParallel-wrapped)
        inputs      : Tensor[N, ...], already on the correct device, detached
        labels      : LongTensor[N], already on the correct device
        criterion   : loss function (e.g. CrossEntropyLoss)
        grad_config : dict with keys 'layers' ('last'|'all'),
                      'use_vmap' (bool), 'dtype' ('float16'|'float32')

    Returns:
        G : Tensor[N, D] on CPU, or None on failure
    """
    target_params = _get_target_params(model, grad_config['layers'])
    if not target_params:
        return None

    N = inputs.size(0)
    was_training = model.training
    model.eval()  # use running stats to avoid ill-conditioned batch norm

    try:
        with torch.enable_grad():
            use_vmap = grad_config['use_vmap'] and _HAS_VMAP
            if use_vmap:
                target_names = [n for n, _ in target_params]
                G = _per_sample_grads_vmap(model, inputs, labels, criterion, target_names)
            else:
                G = _per_sample_grads_loop(model, inputs, labels, criterion, target_params)
    finally:
        if was_training:
            model.train()

    if G.dim() != 2 or G.size(0) != N:
        raise RuntimeError(f'Expected G shape [{N}, D], got {tuple(G.shape)}')

    return G


# ────────────────────────────────────────────────────────────────────────────
# Modified training step
# ────────────────────────────────────────────────────────────────────────────

def run_step_with_grad_analysis(kettle, poison_delta, loss_fn, epoch, stats,
                                model, defs, criterion, optimizer, scheduler,
                                grad_config, grad_records, svd_records,
                                phase='pretrain'):
    """Mirror of forest.victims.training.run_step with added per-sample gradient logging.

    The gradient analysis block runs after augmentation/poisoning but BEFORE the
    training backward pass, capturing the model's gradient landscape on the exact
    data it will train on.  torch.autograd.grad is used so that .grad buffers are
    never modified by the analysis step.

    phase : 'pretrain'  — clean training before brew (poison_delta is None here,
                          so is_poisoned will always be False)
            'retrain'   — model.retrain() call after brew (poison_delta applied)
            'validate'  — model.validate() call after brew (poison_delta applied)

    Appends one entry to grad_records and svd_records per batch.
    """
    epoch_loss, total_preds, correct_preds = 0, 0, 0
    storage_dtype = torch.float16 if grad_config['dtype'] == 'float16' else torch.float32

    loader = kettle.partialloader if kettle.args.ablation < 1.0 else kettle.trainloader

    for batch, (inputs, labels, ids) in enumerate(loader):
        model.train()
        optimizer.zero_grad()

        # ── Transfer to device ───────────────────────────────────────────────
        inputs = inputs.to(**kettle.setup)
        labels = labels.to(dtype=torch.long, device=kettle.setup['device'],
                           non_blocking=NON_BLOCKING)

        # ── Add adversarial (poison) perturbation ────────────────────────────
        poison_slices, batch_positions = [], []
        if poison_delta is not None:
            for batch_id, image_id in enumerate(ids.tolist()):
                lookup = kettle.poison_lookup.get(image_id)
                if lookup is not None:
                    poison_slices.append(lookup)
                    batch_positions.append(batch_id)
            if batch_positions:
                inputs[batch_positions] += poison_delta[poison_slices].to(**kettle.setup)

        # ── Data augmentation ────────────────────────────────────────────────
        if defs.augmentations:
            inputs = kettle.augment(inputs)

        # ── Adversarial training (PGD inner loop, usually 0 steps) ───────────
        for _ in range(defs.adversarial_steps):
            inputs = pgd_step(inputs, labels, model, loss_fn, kettle.dm, kettle.ds,
                              eps=kettle.args.eps, tau=kettle.args.tau)

        # ── GRADIENT ANALYSIS ────────────────────────────────────────────────
        # Pure observation: does not touch optimizer state or .grad buffers.
        is_poisoned   = bool(batch_positions)
        poisoned_count = len(batch_positions)
        N              = inputs.size(0)

        try:
            G = compute_per_sample_grads(
                model, inputs.detach(), labels, criterion, grad_config
            )
        except Exception as exc:
            print(f'[GradAnalysis] WARNING  epoch={epoch} batch={batch}: {exc}')
            G = None

        if G is not None:
            if N == 1:
                # Batch size 1: SVD reduces to the L2 norm of the single-row vector
                S = G.norm(dim=1)  # shape [1]
            else:
                _, S, _ = torch.linalg.svd(G, full_matrices=False)  # S: [min(N, D)]

            grad_records.append({
                'phase':          phase,
                'epoch':          epoch,
                'batch':          batch,
                'is_poisoned':    is_poisoned,
                'poisoned_count': poisoned_count,
                'G':              G.cpu().to(storage_dtype),
            })
            svd_records.append({
                'phase':           phase,
                'epoch':           epoch,
                'batch':           batch,
                'is_poisoned':     is_poisoned,
                'poisoned_count':  poisoned_count,
                'singular_values': S.cpu().to(storage_dtype),
            })
        # ── End gradient analysis ────────────────────────────────────────────

        # ── Normal training forward / backward / step ────────────────────────
        outputs = model(inputs)
        loss    = loss_fn(model, outputs, labels)
        loss.backward()

        with torch.no_grad():
            if defs.privacy['clip'] is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), defs.privacy['clip'])
            if defs.privacy['noise'] is not None:
                from forest.victims.laplace_mechanism import add_laplace_noise
                noise_type = defs.privacy.get(
                    'noise_type', getattr(kettle.args, 'noise_type', 'gaussian'))
                if noise_type == 'laplace':
                    add_laplace_noise(model, defs.privacy['clip'], defs.privacy['noise'])
                else:
                    for param in model.parameters():
                        noise_sample = (torch.randn_like(param)
                                        * defs.privacy['clip'] * defs.privacy['noise'])
                        param.grad += noise_sample

        optimizer.step()

        predictions  = torch.argmax(outputs.data, dim=1)
        total_preds  += labels.size(0)
        correct_preds += (predictions == labels).sum().item()
        epoch_loss   += loss.item()

        if defs.scheduler == 'cyclic':
            scheduler.step()
        if kettle.args.dryrun:
            break

    if defs.scheduler == 'linear':
        scheduler.step()

    if epoch % defs.validate == 0 or epoch == (defs.epochs - 1):
        valid_acc, valid_loss = run_validation(
            model, criterion, kettle.validloader, kettle.setup, kettle.args.dryrun)
        target_acc, target_loss, target_clean_acc, target_clean_loss = check_targets(
            model, criterion, kettle.targetset,
            kettle.poison_setup['intended_class'],
            kettle.poison_setup['target_class'],
            kettle.setup)
    else:
        valid_acc, valid_loss = None, None
        target_acc, target_loss, target_clean_acc, target_clean_loss = [None] * 4

    current_lr = optimizer.param_groups[0]['lr']
    print_and_save_stats(
        epoch, stats, current_lr,
        epoch_loss / (batch + 1), correct_preds / total_preds,
        valid_acc, valid_loss,
        target_acc, target_loss, target_clean_acc, target_clean_loss)


# ────────────────────────────────────────────────────────────────────────────
# Training loop wrapper (mirrors _VictimSingle._iterate for clean training)
# ────────────────────────────────────────────────────────────────────────────

def train_with_grad_analysis(victim, kettle, max_epoch, grad_config,
                              grad_records, svd_records):
    """Run the clean training phase with per-batch gradient logging.

    Mirrors _VictimSingle._iterate(kettle, poison_delta=None, max_epoch=...) but
    delegates each epoch to run_step_with_grad_analysis instead of run_step.
    """
    stats = defaultdict(list)
    if max_epoch is None:
        max_epoch = victim.defs.epochs

    def loss_fn(model, outputs, labels):
        return victim.criterion(outputs, labels)

    for epoch in range(max_epoch):
        run_step_with_grad_analysis(
            kettle=kettle,
            poison_delta=None,  # clean pre-training phase; no poison applied yet
            loss_fn=loss_fn,
            epoch=epoch,
            stats=stats,
            model=victim.model,
            defs=victim.defs,
            criterion=victim.criterion,
            optimizer=victim.optimizer,
            scheduler=victim.scheduler,
            grad_config=grad_config,
            grad_records=grad_records,
            svd_records=svd_records,
        )
        if victim.args.dryrun:
            break

    return stats


# ────────────────────────────────────────────────────────────────────────────
# Victim._step monkey-patch helper
# ────────────────────────────────────────────────────────────────────────────

def _with_logging(victim, grad_config, grad_records, svd_records, phase, flush_dir=None):
    """Replace victim._step with a version that logs per-sample gradients.

    victim._step is the single hook that both _iterate (and therefore train,
    retrain, validate) call for every epoch.  Replacing it lets us capture
    gradient records during the retrain and validate phases without touching
    any forest source file.

    If flush_dir is set, records are written to per-epoch files after each
    epoch and cleared from memory — keeping at most one epoch in RAM at a time.
    Files are named: gradients_{phase}_epoch{N:04d}.pt / SVD_{phase}_epoch{N:04d}.pt

    The replacement is a bound method so it receives `self` as the first arg
    and has the exact same signature as the original _step.
    """
    flush_stats = {'total_batches': 0, 'poisoned_batches': 0, 'G_shape': None}

    def _step(self, kettle, poison_delta, loss_fn, epoch, stats,
              model, defs, criterion, optimizer, scheduler):
        run_step_with_grad_analysis(
            kettle, poison_delta, loss_fn, epoch, stats,
            model, defs, criterion, optimizer, scheduler,
            grad_config, grad_records, svd_records, phase=phase,
        )
        if flush_dir is not None and grad_records:
            os.makedirs(flush_dir, exist_ok=True)
            g_path = os.path.join(flush_dir, f'gradients_{phase}_epoch{epoch:04d}.pt')
            s_path = os.path.join(flush_dir, f'SVD_{phase}_epoch{epoch:04d}.pt')
            torch.save(list(grad_records), g_path)
            torch.save(list(svd_records),  s_path)
            flush_stats['total_batches']   += len(grad_records)
            flush_stats['poisoned_batches'] += sum(1 for r in grad_records if r['is_poisoned'])
            if flush_stats['G_shape'] is None and grad_records:
                flush_stats['G_shape'] = tuple(grad_records[0]['G'].shape)
            grad_records.clear()
            svd_records.clear()
            print(f'[GradAnalysis] Flushed epoch {epoch} → {g_path}')

    victim._step = types.MethodType(_step, victim)
    return flush_stats


# ────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ────────────────────────────────────────────────────────────────────────────

def _build_parser():
    """Extend the forest options parser with gradient-analysis-specific flags."""
    # parse_known_args lets the forest parser consume its own flags; we handle the rest
    base_parser = forest.options()
    base_args, remaining = base_parser.parse_known_args()

    grad_parser = argparse.ArgumentParser(
        description='Gradient analysis options (appended to brew_poison args)')
    grad_parser.add_argument(
        '--grad_save_dir', default='gradient_logs', type=str,
        help='Directory in which gradients.pt and SVD_matrix.pt are written (default: gradient_logs/).')
    grad_parser.add_argument(
        '--grad_layers', default='last', choices=['last', 'all'],
        help='"last" logs only the final linear layer (default); "all" logs every parameter.')
    grad_parser.add_argument(
        '--grad_dtype', default='float16', choices=['float32', 'float16'],
        help='Storage dtype for saved tensors (default: float16 to save space).')
    grad_parser.add_argument(
        '--no_vmap', action='store_true',
        help='Disable vmap and fall back to a per-sample loop (slower but always available).')

    grad_args = grad_parser.parse_args(remaining)
    return base_args, grad_args


# ────────────────────────────────────────────────────────────────────────────
# Entry point
# ────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    args, grad_args = _build_parser()

    if args.deterministic:
        forest.utils.set_deterministic()

    grad_config = {
        'layers':   grad_args.grad_layers,
        'dtype':    grad_args.grad_dtype,
        'use_vmap': not grad_args.no_vmap,
    }

    vmap_status = 'available' if _HAS_VMAP else 'unavailable (falling back to loop)'
    print(f'[GradAnalysis] torch.func.vmap: {vmap_status}')
    print(f'[GradAnalysis] grad_layers={grad_config["layers"]}, '
          f'grad_dtype={grad_config["dtype"]}, use_vmap={grad_config["use_vmap"]}')

    # ── Forest setup (identical to brew_poison.py) ───────────────────────────
    setup = forest.utils.system_startup(args)

    model = forest.Victim(args, setup=setup)
    data  = forest.Kettle(args, model.defs.batch_size, model.defs.augmentations, setup=setup)
    witch = forest.Witch(args, setup=setup)

    # ── Storage for gradient records (validate phase only) ───────────────────
    grad_records: list = []
    svd_records:  list = []

    # ── Clean training: plain train(), no gradient analysis ───────────────────
    start_time = time.time()
    if args.pretrained:
        print('[GradAnalysis] Pretrained model: skipping training phase.')
        stats_clean = None
    else:
        print('[GradAnalysis] Clean training (no gradient logging) ...')
        stats_clean = model.train(data, max_epoch=args.max_epoch)
    train_time = time.time()

    # ── Poison brewing (unchanged from brew_poison.py) ───────────────────────
    poison_delta = witch.brew(model, data)
    brew_time = time.time()

    # ── Retrain phase: plain retrain(), no gradient analysis ─────────────────
    if not args.pretrained and args.retrain_from_init:
        stats_rerun = model.retrain(data, poison_delta)
    else:
        stats_rerun = None

    # ── Validate phase: fresh victim trained on poisoned data, gradient logging enabled ─
    flush_stats = {'total_batches': 0, 'poisoned_batches': 0, 'G_shape': None}
    if args.vnet is not None:
        train_net = args.net
        args.net  = args.vnet
        if args.vruns > 0:
            model = forest.Victim(args, setup=setup)
            print('[GradAnalysis] Validate phase — per-sample gradient logging enabled (flush per epoch).')
            flush_stats = _with_logging(model, grad_config, grad_records, svd_records,
                                        phase='validate', flush_dir=grad_args.grad_save_dir)
            stats_results = model.validate(data, poison_delta)
        else:
            stats_results = None
        args.net = train_net
    else:
        if args.vruns > 0:
            print('[GradAnalysis] Validate phase — per-sample gradient logging enabled (flush per epoch).')
            flush_stats = _with_logging(model, grad_config, grad_records, svd_records,
                                        phase='validate', flush_dir=grad_args.grad_save_dir)
            stats_results = model.validate(data, poison_delta)
        else:
            stats_results = None

    test_time = time.time()

    # ── Save summary of gradient records (data already flushed per epoch) ───────
    total_batches    = flush_stats['total_batches']
    poisoned_batches = flush_stats['poisoned_batches']
    G_shape          = flush_stats['G_shape']
    print(f'[GradAnalysis] Flushed {total_batches} batch records '
          f'({poisoned_batches} with poisoned samples) to per-epoch files in: {grad_args.grad_save_dir}/')
    print(f'[GradAnalysis]   gradients_validate_epoch{{N:04d}}.pt')
    print(f'[GradAnalysis]   SVD_validate_epoch{{N:04d}}.pt')
    if G_shape is not None:
        print(f'[GradAnalysis] G shape per batch: {G_shape}  '
              f'(N={G_shape[0]} samples, D={G_shape[1]} param elements)')

    # ── Record results (identical to brew_poison.py) ─────────────────────────
    timestamps = dict(
        train_time=str(datetime.timedelta(seconds=train_time - start_time)).replace(',', ''),
        brew_time =str(datetime.timedelta(seconds=brew_time  - train_time)).replace(',', ''),
        test_time =str(datetime.timedelta(seconds=test_time  - brew_time )).replace(',', ''))

    results = (stats_clean, stats_rerun, stats_results)
    forest.utils.record_results(data, witch.stat_optimal_loss, results,
                                args, model.defs, model.model_init_seed,
                                extra_stats=timestamps)

    if args.save is not None:
        data.export_poison(poison_delta, path=args.poison_path, mode=args.save)

    print(datetime.datetime.now().strftime('%A, %d. %B %Y %I:%M%p'))
    print('---------------------------------------------------')
    print(f'Finished computations with train time: {str(datetime.timedelta(seconds=train_time - start_time))}')
    print(f'--------------------------- brew time: {str(datetime.timedelta(seconds=brew_time - train_time))}')
    print(f'--------------------------- test time: {str(datetime.timedelta(seconds=test_time - brew_time))}')
    print('-------------Job finished.-------------------------')
