"""Fits the embedding probe and listwise ranker on training references only; writes runs/model_bundle."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np

from common import DATA, load_inputs, parse_answer, read_jsonl
import features as F
import model as M
import textmodel as X

# Embedding probes: input is 'raw' | 'centered' | 'both'; model is 'linear' | 'mlp' | 'listwise'.
LINEAR_CENTERED = {'input': 'centered', 'model': 'linear', 'l2': 10.0}
LINEAR_RAW = {'input': 'raw', 'model': 'linear', 'l2': 10.0}
MLP_CENTERED = {'input': 'centered', 'model': 'mlp', 'hidden': 16, 'l2': 1000.0}
LISTWISE_CENTERED = {'input': 'centered', 'model': 'listwise', 'l2': 3000.0}
# Hashed n-gram bag of the candidate text, trained from scratch; tokens is 'all' | 'words' | 'noformat'.
NGRAM = {'input': 'text', 'model': 'ngram', 'tokens': 'all', 'n': 3, 'dim': 2**18, 'l2': 1.0}

DEFAULT = {
    'probes': [LINEAR_CENTERED, LINEAR_RAW],
    'probe_train': 'all',
    'groups': ['answer', 'vote', 'surface', 'reasoning', 'cross'],
    'similarity': None,
    'interactions': False,
    'boost': None,
    'rank_l2': 1.0,
    'rank_train': 'hard',
    'folds': 5,
    'seed': 7506,
}


def prepare(split, data_dir=DATA):
    """Inputs plus training-time correctness labels (references are read here only, never in predict)."""
    bank, arrays = load_inputs(split, data_dir)
    feats, answers = F.extract(bank)
    gold = [parse_answer(q['answer']) for q in read_jsonl(Path(data_dir)/f'gsm8k_{split}.jsonl')]
    correct = np.array([[a is not None and a == g for a in row] for row, g in zip(answers, gold)])
    return {'bank': bank, 'emb': arrays['embeddings'], 'feats': feats, 'answers': answers,
            'texts': [[c['text'] for c in row['candidates']] for row in bank],
            'valid': M.parsable_mask(answers), 'hard': M.hard_mask(answers), 'correct': correct}


def _subset(data, which):
    return data['hard'] if which == 'hard' else np.ones(len(data['hard']), bool)


def ngram_matrix(data, spec):
    key = ('ngram', spec['tokens'], spec['n'], spec['dim'])
    if key not in data:
        data[key] = X.featurize([t for row in data['texts'] for t in row], spec['tokens'], spec['n'], spec['dim'])
    return data[key]


def probe_scores(data, spec, params):
    """[N,4] logits of a fitted probe on every candidate of `data`."""
    if spec['model'] == 'ngram':
        return (X.matvec(ngram_matrix(data, spec), params['probe_w']) + params['probe_b']).reshape(-1, 4)
    return M.probe_logit(params, M.probe_inputs(data['emb'], data['valid'], spec['input']))


def fit_probe(data, spec, cfg, groups=None):
    which = spec.get('train', cfg['probe_train'])
    use = _subset(data, which) if groups is None else groups & _subset(data, which)
    if spec['model'] == 'ngram':
        keep = (use[:, None] & data['valid']).ravel()
        return X.fit(X.subset(ngram_matrix(data, spec), keep), data['correct'].ravel()[keep], spec['dim'], spec['l2'])
    X_in = M.probe_inputs(data['emb'], data['valid'], spec['input'])
    if spec['model'] == 'listwise':
        use = use & data['hard']
        return M.fit_listwise_probe(X_in[use], data['valid'][use], data['correct'][use], spec['l2'])
    m = use[:, None] & data['valid']
    if spec['model'] == 'mlp':
        return M.fit_mlp_probe(X_in[m], data['correct'][m], spec['hidden'], spec['l2'], cfg['seed'])
    return M.fit_probe(X_in[m], data['correct'][m], spec['l2'])


def oof_probe_logits(data, spec, cfg):
    """Out-of-fold probe logits on the training split, so the ranker sees honest probe scores."""
    n = len(data['hard'])
    fold = np.random.default_rng(cfg['seed']).permutation(n) % cfg['folds']
    out = np.zeros(data['valid'].shape, np.float32)
    for k in range(cfg['folds']):
        out[fold == k] = probe_scores(data, spec, fit_probe(data, spec, cfg, groups=fold != k))[fold == k]
    return out


def ranker_matrix(data, cfg, logits):
    extra = M.similarity_features(data['emb'], data['answers'], cfg['similarity']) if cfg['similarity'] else None
    patterns = M.vote_patterns(data['answers']) if cfg.get('interactions') else None
    return M.ranker_inputs(data['feats'], logits, data['valid'], F.columns(cfg['groups']), extra, patterns)


def fit(data, cfg, oofs=None):
    t0 = time.perf_counter()
    bundle = {'config': dict(cfg)}
    if oofs is None:
        oofs = [oof_probe_logits(data, spec, cfg) for spec in cfg['probes']]
    for k, spec in enumerate(cfg['probes']):
        bundle.update({f'p{k}_{name}': v for name, v in fit_probe(data, spec, cfg).items()})
    X = ranker_matrix(data, cfg, oofs)
    use = _subset(data, cfg['rank_train'])
    bundle.update(M.fit_combiner(X[use], data['valid'][use], data['correct'][use], cfg))
    bundle['train_seconds'] = time.perf_counter() - t0
    return bundle


def save(bundle, out):
    out.mkdir(parents=True, exist_ok=True)
    arrays = {k: np.asarray(v, np.int16 if k == 'tree_feat' else np.float32)
              for k, v in bundle.items() if k not in ('config', 'train_seconds')}
    np.savez(out/'weights.npz', **arrays)
    (out/'config.json').write_text(json.dumps(bundle['config'], indent=2) + '\n')
    files = {p.name: {'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
             for p in sorted(out.glob('*')) if p.name in ('weights.npz', 'config.json')}
    record = {'files': files, 'total_bytes': sum(f['bytes'] for f in files.values()),
              'train_seconds': bundle['train_seconds'], 'python': platform.python_version(),
              'numpy': np.__version__, 'machine': platform.processor() or platform.machine()}
    (out.parent/f'{out.name}.record.json').write_text(json.dumps(record, indent=2) + '\n')
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, help='JSON overriding the default configuration')
    parser.add_argument('--output', type=Path, default=Path('runs/model_bundle'))
    args = parser.parse_args()
    cfg = dict(DEFAULT)
    if args.config:
        cfg.update(json.loads(args.config.read_text()))
    data = prepare('train')
    record = save(fit(data, cfg), args.output)
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
