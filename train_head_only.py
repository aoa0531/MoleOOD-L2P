"""Frozen MoleOOD encoders, no prompts, incremental prediction-head baseline."""
import argparse
import json
import random

import numpy as np
import torch
from torch.nn import functional as F

from train_iil import (build_model, evaluate_loader, loader, resolve,
                       save_checkpoint, validate_split)


def restore_rng(checkpoint):
    random.setstate(checkpoint['python_rng'])
    np.random.set_state(checkpoint['numpy_rng'])
    torch.set_rng_state(checkpoint['torch_rng'].cpu())
    if torch.cuda.is_available() and checkpoint['cuda_rng'] is not None:
        torch.cuda.set_rng_state_all([s.cpu() for s in checkpoint['cuda_rng']])


def head_model(checkpoint, cfg, device, baseline='head_only', resume=False):
    if baseline == 'shared_prompt' and resume:
        model = build_model(cfg, prompted=True).to(device)
        model.load_state_dict(checkpoint['main'], strict=True)
        return model
    model = build_model(cfg, prompted=False).to(device)
    # Source stage_0 contains unused random prompts; retain exactly its backbone/head.
    state = {k: v for k, v in checkpoint['main'].items()
             if not k.startswith('prompt_pool.')}
    model.load_state_dict(state, strict=True)
    model.freeze_encoders()
    if baseline == 'shared_prompt':
        model.enable_shared_prompt(cfg['prompt_init_scale'])
        # Fixed, predeclared choice, never selected by validation/test performance.
        with torch.no_grad():
            model.prompt_pool.prompts.copy_(checkpoint['main']['prompt_pool.prompts'][:1])
    return model


def main(baseline='head_only'):
    if baseline not in ('head_only', 'shared_prompt'):
        raise ValueError('Unknown baseline')
    parser = argparse.ArgumentParser(f'Frozen-encoder IIL baseline: {baseline}')
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--source', help='Prompt experiment stage_0.pt (before D1)')
    source.add_argument('--resume', help='Completed head-only stage checkpoint')
    parser.add_argument('--output-dir', help='New, separate output directory (required with --source)')
    args = parser.parse_args()
    checkpoint = torch.load(args.source or args.resume, map_location='cpu', weights_only=False)
    if checkpoint.get('format') != 'moleood_iil_v1':
        raise ValueError('Expected a trusted IIL checkpoint')
    cfg = checkpoint['config'].copy()
    manifest = checkpoint['manifest']
    validate_split(manifest, resolve(cfg['data_path']))
    if args.source:
        if (checkpoint['stage'] != 0 or not checkpoint['prompted']
                or checkpoint['config'].get('baseline') is not None):
            raise ValueError('Use the original prompt experiment stage_0.pt, not a later stage')
        if not args.output_dir:
            parser.error('--output-dir is required with --source')
        cfg['output_dir'] = args.output_dir
        cfg['baseline'] = baseline
        if baseline == 'shared_prompt':
            cfg['pool_size'] = 1
            cfg['key_loss_weight'] = 0.0
            cfg['shared_prompt_init_index'] = 0
        history = []
    else:
        if cfg.get('baseline') != baseline or checkpoint['prompted'] != (baseline == 'shared_prompt'):
            raise ValueError(f'Resume requires a {baseline} checkpoint')
        if args.output_dir and resolve(args.output_dir) != resolve(cfg['output_dir']):
            raise ValueError('Resume output directory must remain unchanged')
        history = checkpoint['history']
    output_dir = resolve(cfg['output_dir'])
    if args.source and output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError('Choose a new output directory; original results are never overwritten')
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(cfg['device'])
    model = head_model(checkpoint, cfg, device, baseline, resume=bool(args.resume))
    groups = [{'params': model.predictor.parameters(), 'lr': cfg['head_lr']}]
    if baseline == 'shared_prompt':
        groups.append({'params': model.prompt_pool.parameters(), 'lr': cfg['prompt_lr']})
    optimizer = torch.optim.Adam(groups, weight_decay=0)
    weights = checkpoint['class_weights']
    if weights is not None:
        weights = weights.to(device)
    evaluations = {name: loader(cfg, name) for name in cfg['eval_splits']}
    if args.resume:
        optimizer.load_state_dict(checkpoint['optimizer'])
        restore_rng(checkpoint)
    else:
        metrics = {name: evaluate_loader(model, batches, device)
                   for name, batches in evaluations.items()}
        history.append({'stage': 0, 'mode': baseline, 'metrics': metrics})
        # Evaluation/model construction consumes RNG. Restore the original D1 start
        # AFTER both so shuffled batches and predictor dropout align with that run.
        restore_rng(checkpoint)
        save_checkpoint(output_dir/'stage_0.pt', model, optimizer, cfg, manifest, 0, weights, history)
    (output_dir/'metrics.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
    for stage in range(checkpoint['stage'] + 1, len(manifest['stages'])):
        batches = loader(cfg, 'train', manifest['stages'][stage], shuffle=True)
        for epoch in range(cfg['incremental_epochs']):
            model.train()
            total, seen = 0., 0
            for data in batches:
                data = data.to(device)
                loss = F.cross_entropy(model(data), data.y.view(-1), weight=weights)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total += loss.item() * len(data.y)
                seen += len(data.y)
            print(f'{baseline} D{stage} epoch {epoch+1}: cls_loss={total/seen:.6f}', flush=True)
        metrics = {name: evaluate_loader(model, batches, device)
                   for name, batches in evaluations.items()}
        history.append({'stage': stage, 'mode': baseline, 'metrics': metrics})
        save_checkpoint(output_dir/f'stage_{stage}.pt', model, optimizer,
                        cfg, manifest, stage, weights, history)
        (output_dir/'metrics.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
        print(json.dumps(history[-1], indent=2), flush=True)


if __name__ == '__main__':
    main()
