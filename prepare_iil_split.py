"""Create deterministic, approximately stratified instance increments."""
import argparse
import hashlib
import json
import random
from collections import defaultdict, Counter
from pathlib import Path


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_split(data_path, seed=42, fractions=(.4, .15, .15, .15, .15)):
    if min(fractions) <= 0 or abs(sum(fractions) - 1) > 1e-8:
        raise ValueError('Positive fractions must sum to one')
    data = json.loads(Path(data_path).read_text(encoding='utf-8'))
    rows = data['split']['train']
    # Exact SMILES grouping: no chemistry toolkit required. See README limitation.
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[row['smiles']].append(i)
    rng = random.Random(seed)
    grouped = list(groups.values())
    rng.shuffle(grouped)
    grouped.sort(key=len, reverse=True)
    total = Counter(int(r['cls_label']) for r in rows)
    if set(total) != {0, 1}:
        raise ValueError('Expected binary labels 0 and 1')
    counts = [[0, 0] for _ in fractions]
    stages = [[] for _ in fractions]
    for group in grouped:
        gc = Counter(int(rows[i]['cls_label']) for i in group)
        # Change in squared relative deviation from desired class counts.
        def cost(stage):
            return sum(((counts[stage][c] + gc[c] - total[c]*fractions[stage])**2
                        - (counts[stage][c] - total[c]*fractions[stage])**2)
                       / max(total[c]*fractions[stage], 1) for c in (0, 1))
        stage = min(range(len(stages)), key=cost)
        stages[stage].extend(group)
        for c in (0, 1):
            counts[stage][c] += gc[c]
    if any(min(c) == 0 for c in counts):
        raise ValueError('A stage lacks a class; reduce stage count or change fractions')
    for stage in stages:
        rng.shuffle(stage)
    train_smiles = set(groups)
    return {'version': 1, 'data_sha256': file_hash(data_path), 'seed': seed,
            'fractions': list(fractions), 'grouping': 'exact_smiles',
            'stages': stages, 'class_counts': counts,
            'duplicate_groups': sum(len(g) > 1 for g in grouped),
            'conflicting_label_groups': sum(len({rows[i]['cls_label'] for i in g}) > 1
                                            for g in grouped),
            'train_eval_smiles_overlap': {
                name: len(train_smiles & {r['smiles'] for r in records})
                for name, records in data['split'].items() if name != 'train'}}


def validate_split(manifest, data_path):
    if manifest['data_sha256'] != file_hash(data_path):
        raise ValueError('Dataset hash differs from the split manifest')
    rows = json.loads(Path(data_path).read_text(encoding='utf-8'))['split']['train']
    flat = [i for stage in manifest['stages'] for i in stage]
    if sorted(flat) != list(range(len(rows))):
        raise ValueError('Stages must cover train exactly once')
    owners = {}
    for t, stage in enumerate(manifest['stages']):
        if {int(rows[i]['cls_label']) for i in stage} != {0, 1}:
            raise ValueError('Every stage must contain both labels')
        for i in stage:
            smiles = rows[i]['smiles']
            if smiles in owners and owners[smiles] != t:
                raise ValueError('Duplicate molecule crosses stages')
            owners[smiles] = t


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--fractions', nargs='+', type=float, default=[.4,.15,.15,.15,.15])
    args = parser.parse_args()
    manifest = make_split(args.data, args.seed, args.fractions)
    validate_split(manifest, args.data)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('Choose a new output path; existing manifests are not overwritten')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in manifest.items() if k != 'stages'}, indent=2))
