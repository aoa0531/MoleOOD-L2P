"""Supervised instance-incremental MoleOOD with additive, length-one prompts."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             balanced_accuracy_score, accuracy_score)

from models import Framework, ConditionalGnn, DomainClassifier, KLDist, MeanLoss, DeviationLoss
from models.dataset import LBAPDatasetWithSub, collate_molecules
from models.prompt_pool import step_selected_rows
from prepare_iil_split import validate_split


ROOT = Path(__file__).resolve().parent


def resolve(path):
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def loader(cfg, split, indices=None, shuffle=False):
    dataset = LBAPDatasetWithSub(str(resolve(cfg['data_path'])), split, indices)
    return DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=shuffle,
                      num_workers=cfg.get('num_workers', 0), collate_fn=collate_molecules)


def build_model(cfg, prompted=False):
    shared = prompted and cfg.get('baseline') == 'shared_prompt'
    model = Framework(cfg['emb_dim'], cfg['emb_dim'], 2, cfg['dropout'],
                     prompt_pool_size=cfg['pool_size'] if prompted and not shared else None,
                     prompt_init_scale=cfg['prompt_init_scale'])
    if shared:
        model.enable_shared_prompt(cfg['prompt_init_scale'])
    return model


@torch.no_grad()
def evaluate_loader(model, batches, device):
    model.eval()
    labels, scores = [], []
    histogram = torch.zeros(len(model.prompt_pool.prompts), dtype=torch.long) \
        if model.prompt_pool is not None else None
    for data in batches:
        data = data.to(device)
        output = model(data, return_aux=True)
        labels.extend(data.y.view(-1).cpu().tolist())
        scores.extend(output['logits'].softmax(-1)[:, 1].cpu().tolist())
        if histogram is not None:
            histogram += torch.bincount(output['prompt_indices'].cpu(), minlength=len(histogram))
    if not labels:
        raise ValueError('Empty evaluation split')
    both = len(set(labels)) == 2
    predictions = np.asarray(scores) >= .5
    return {'roc_auc': float(roc_auc_score(labels, scores)) if both else None,
            'pr_auc_ap': float(average_precision_score(labels, scores)) if both else None,
            'balanced_accuracy': float(balanced_accuracy_score(labels, predictions)) if both else None,
            'accuracy': float(accuracy_score(labels, predictions)),
            'count': len(labels),
            'prompt_counts': histogram.tolist() if histogram is not None else None}


def train_initial(model, batches, cfg, device):
    """Original two-stage objective, using D0 only; final fixed-epoch weights."""
    dim, domains = cfg['emb_dim'], cfg['num_domain']
    conditional = ConditionalGnn(dim, dim, domains, 2).to(device)
    classifier = DomainClassifier(dim, domains, 1).to(device)
    opt_con = torch.optim.AdamW(conditional.parameters(), lr=cfg['initial_lr'])
    opt_dom = torch.optim.AdamW(classifier.parameters(), lr=cfg['initial_lr'])
    prior = torch.ones(domains, device=device) / domains
    best_loss, best = float('inf'), None
    for epoch in range(cfg['assistant_epochs']):
        conditional.train(); classifier.train()
        total, seen = 0., 0
        for data in batches:
            data = data.to(device)
            y = data.y.view(-1)
            q = classifier(data).softmax(-1)
            losses = torch.stack([
                F.cross_entropy(conditional(data, torch.full_like(y, e)), y, reduction='none')
                for e in range(domains)], dim=1)
            loss = (q * losses).sum(-1).mean() + KLDist(q, prior)
            opt_con.zero_grad(set_to_none=True); opt_dom.zero_grad(set_to_none=True)
            loss.backward(); opt_con.step(); opt_dom.step()
            total += loss.item()*len(y); seen += len(y)
        print(f'D0 assistant epoch {epoch+1}: {total/seen:.6f}', flush=True)
        if total/seen < best_loss:
            best_loss = total/seen
            best = (deepcopy(conditional.state_dict()), deepcopy(classifier.state_dict()))
    conditional.load_state_dict(best[0]); classifier.load_state_dict(best[1])
    conditional.eval(); classifier.eval()
    conditional.requires_grad_(False); classifier.requires_grad_(False)
    mean_loss = MeanLoss(torch.nn.CrossEntropyLoss())
    deviation = DeviationLoss('abs', 'mean')
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['initial_lr'])
    for epoch in range(cfg['initial_epochs']):
        model.train()
        total, seen = 0., 0
        for data in batches:
            data = data.to(device)
            y = data.y.view(-1)
            with torch.no_grad():
                reference = torch.stack([
                    F.cross_entropy(conditional(data, torch.full_like(y, e)), y, reduction='none')
                    for e in range(domains)], dim=0).mean(0)
                groups = classifier(data).argmax(-1)
            logits = model(data)
            loss = cfg['invariance_weight'] * mean_loss(logits, y, groups) + deviation(
                F.cross_entropy(logits, y, reduction='none'), reference)
            optimizer.zero_grad(set_to_none=True)
            loss.backward(); optimizer.step()
            total += loss.item()*len(y); seen += len(y)
        print(f'D0 main epoch {epoch+1}: {total/seen:.6f}', flush=True)


def make_optimizer(model, cfg):
    return torch.optim.Adam([
        {'params': model.prompt_pool.parameters(), 'lr': cfg['prompt_lr']},
        {'params': model.predictor.parameters(), 'lr': cfg['head_lr']}], weight_decay=0)


def train_increment(model, batches, optimizer, cfg, device, weights, stage):
    for epoch in range(cfg['incremental_epochs']):
        model.train()
        total, seen = 0., 0
        counts = torch.zeros(cfg['pool_size'], dtype=torch.long)
        for data in batches:
            data = data.to(device)
            result = model(data, return_aux=True)
            loss = F.cross_entropy(result['logits'], data.y.view(-1), weight=weights)
            loss = loss + cfg['key_loss_weight'] * result['key_loss']
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            step_selected_rows(optimizer, model.prompt_pool, result['prompt_indices'])
            counts += torch.bincount(result['prompt_indices'].detach().cpu(), minlength=len(counts))
            total += loss.item()*len(data.y); seen += len(data.y)
        print(f'D{stage} epoch {epoch+1}: loss={total/seen:.6f}, prompts={counts.tolist()}', flush=True)


def save_checkpoint(path, model, optimizer, cfg, manifest, stage, weights, history):
    state = {'format': 'moleood_iil_v1', 'main': model.state_dict(),
             'prompted': model.prompt_pool is not None, 'stage': stage,
             'optimizer': optimizer.state_dict() if optimizer is not None else None,
             'config': cfg, 'manifest': manifest,
             'class_weights': weights.cpu() if weights is not None else None,
             'history': history, 'python_rng': random.getstate(),
             'numpy_rng': np.random.get_state(), 'torch_rng': torch.get_rng_state(),
             'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
    tmp = path.with_suffix('.tmp')
    torch.save(state, tmp)
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', default=str(ROOT/'configs/iil_scaffold.json'))
    parser.add_argument('--resume', help='Trusted stage-boundary checkpoint')
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding='utf-8'))
    for name in ('assistant_epochs', 'initial_epochs', 'incremental_epochs', 'batch_size', 'num_domain'):
        if cfg[name] < 1:
            raise ValueError(f'{name} must be positive')
    seed_all(cfg['seed'])
    device = torch.device(cfg['device'])
    manifest = json.loads(resolve(cfg['split_path']).read_text(encoding='utf-8'))
    validate_split(manifest, resolve(cfg['data_path']))
    output_dir = resolve(cfg['output_dir'])
    if not args.resume and output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError('Output directory is nonempty; use --resume or a new output_dir')
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluations = {name: loader(cfg, name) for name in cfg['eval_splits']}
    d0 = loader(cfg, 'train', manifest['stages'][0], shuffle=True)
    counts = torch.bincount(torch.tensor([int(r['cls_label']) for r in d0.dataset.data_infos]), minlength=2)
    weights = (counts.sum()/(2*counts.float())).to(device) if cfg['class_weighted'] else None
    history = []
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        if ckpt['format'] != 'moleood_iil_v1' or ckpt['manifest'] != manifest or ckpt['config'] != cfg:
            raise ValueError('Resume requires identical configuration and split manifest')
        model = build_model(cfg, prompted=ckpt['prompted']).to(device)
        model.load_state_dict(ckpt['main'])
        if model.prompt_pool is None:
            model.enable_prompts(cfg['pool_size'], cfg['prompt_init_scale'])
        optimizer = make_optimizer(model, cfg)
        if ckpt['optimizer'] is not None:
            optimizer.load_state_dict(ckpt['optimizer'])
        history = ckpt['history']
        random.setstate(ckpt['python_rng']); np.random.set_state(ckpt['numpy_rng'])
        torch.set_rng_state(ckpt['torch_rng'].cpu())
        if torch.cuda.is_available() and ckpt['cuda_rng'] is not None:
            torch.cuda.set_rng_state_all([s.cpu() for s in ckpt['cuda_rng']])
        start = ckpt['stage'] + 1
    else:
        model = build_model(cfg).to(device)
        train_initial(model, d0, cfg, device)
        metrics = {name: evaluate_loader(model, batch, device) for name, batch in evaluations.items()}
        history.append({'stage': 0, 'mode': 'unprompted', 'metrics': metrics})
        save_checkpoint(output_dir/'initial_unprompted.pt', model, None, cfg, manifest, 0, weights, history)
        model.enable_prompts(cfg['pool_size'], cfg['prompt_init_scale'])
        optimizer = make_optimizer(model, cfg)
        metrics = {name: evaluate_loader(model, batch, device) for name, batch in evaluations.items()}
        history.append({'stage': 0, 'mode': 'random_prompts', 'metrics': metrics})
        save_checkpoint(output_dir/'stage_0.pt', model, optimizer, cfg, manifest, 0, weights, history)
        start = 1
    (output_dir/'metrics.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
    for stage in range(start, len(manifest['stages'])):
        batches = loader(cfg, 'train', manifest['stages'][stage], shuffle=True)
        train_increment(model, batches, optimizer, cfg, device, weights, stage)
        metrics = {name: evaluate_loader(model, batch, device) for name, batch in evaluations.items()}
        history.append({'stage': stage, 'metrics': metrics})
        save_checkpoint(output_dir/f'stage_{stage}.pt', model, optimizer, cfg, manifest, stage, weights, history)
        (output_dir/'metrics.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
        print(json.dumps(history[-1], indent=2), flush=True)


def evaluation_main():
    parser = argparse.ArgumentParser('Evaluate an IIL checkpoint (no adaptation)')
    parser.add_argument('--iil-checkpoint', required=True)
    parser.add_argument('--data', help='Optional relocated dataset with identical SHA256')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--splits', nargs='+', default=['iid_test', 'ood_test'])
    args = parser.parse_args()
    checkpoint = torch.load(args.iil_checkpoint, map_location='cpu', weights_only=False)
    if checkpoint.get('format') != 'moleood_iil_v1':
        raise ValueError('Not an IIL checkpoint')
    cfg = checkpoint['config'].copy()
    if args.data:
        cfg['data_path'] = args.data
    validate_split(checkpoint['manifest'], resolve(cfg['data_path']))
    device = torch.device(args.device)
    model = build_model(cfg, prompted=checkpoint['prompted']).to(device)
    model.load_state_dict(checkpoint['main'])
    print(json.dumps({s: evaluate_loader(model, loader(cfg, s), device) for s in args.splits}, indent=2))


if __name__ == '__main__':
    main()
