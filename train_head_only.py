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


def head_model(checkpoint, cfg, device):
    model = build_model(cfg, prompted=False).to(device)
    # Source stage_0 contains unused random prompts; retain exactly its backbone/head.
    state = {k: v for k, v in checkpoint['main'].items()
             if not k.startswith('prompt_pool.')}
    model.load_state_dict(state, strict=True)
    model.freeze_encoders()
    return model


def main():
    parser = argparse.ArgumentParser(__doc__)
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
        if checkpoint['stage'] != 0 or not checkpoint['prompted']:
            raise ValueError('Use the original prompt experiment stage_0.pt, not a later stage')
        if not args.output_dir:
            parser.error('--output-dir is required with --source')
        cfg['output_dir'] = args.output_dir
        cfg['baseline'] = 'head_only'
        history = []
    else:
        if cfg.get('baseline') != 'head_only' or checkpoint['prompted']:
            raise ValueError('Resume requires a head-only checkpoint')
        if args.output_dir and resolve(args.output_dir) != resolve(cfg['output_dir']):
            raise ValueError('Resume output directory must remain unchanged')
        history = checkpoint['history']
    output_dir = resolve(cfg['output_dir'])
    if args.source and output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError('Choose a new output directory; original results are never overwritten')
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(cfg['device'])
    model = head_model(checkpoint, cfg, device)
    optimizer = torch.optim.Adam(model.predictor.parameters(), lr=cfg['head_lr'], weight_decay=0)
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
        history.append({'stage': 0, 'mode': 'head_only', 'metrics': metrics})
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
            print(f'Head-only D{stage} epoch {epoch+1}: cls_loss={total/seen:.6f}', flush=True)
        metrics = {name: evaluate_loader(model, batches, device)
                   for name, batches in evaluations.items()}
        history.append({'stage': stage, 'mode': 'head_only', 'metrics': metrics})
        save_checkpoint(output_dir/f'stage_{stage}.pt', model, optimizer,
                        cfg, manifest, stage, weights, history)
        (output_dir/'metrics.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
        print(json.dumps(history[-1], indent=2), flush=True)


if __name__ == '__main__':
    main()
