"""Baselines, ablations and diagnostics on train (CV) and validation; never touches test references."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

import features as F
import model as M
import train as T


def accuracy(choices, data):
    return data['correct'][np.arange(len(choices)), choices].astype(float)


def bootstrap_ci(ok, reps=2000, seed=0):
    rng = np.random.default_rng(seed)
    means = ok[rng.integers(0, len(ok), (reps, len(ok)))].mean(1)
    return float(100*np.percentile(means, 2.5)), float(100*np.percentile(means, 97.5))


def mcnemar(a, b):
    """Exact two-sided McNemar p-value on paired correctness vectors."""
    from math import comb
    n01, n10 = int(((a == 0) & (b == 1)).sum()), int(((a == 1) & (b == 0)).sum())
    n, k = n01 + n10, min(n01, n10)
    p = min(1.0, 2*sum(comb(n, i) for i in range(k + 1))/2**n) if n else 1.0
    return n01, n10, p


class Runner:
    def __init__(self):
        self.tr, self.va = T.prepare('train'), T.prepare('validation')
        self.oof = {}

    def oof_logits(self, cfg):
        out = []
        for spec in cfg['probes']:
            key = (json.dumps(spec, sort_keys=True), cfg['probe_train'], cfg['seed'], cfg['folds'])
            if key not in self.oof:
                self.oof[key] = T.oof_probe_logits(self.tr, spec, cfg)
            out.append(self.oof[key])
        return out

    def bundle(self, cfg):
        return T.fit(self.tr, cfg, self.oof_logits(cfg))

    def ranker_cv(self, cfg, folds=5):
        """Ranker-only CV on train-hard groups using honest out-of-fold probe logits."""
        tr = self.tr
        X = T.ranker_matrix(tr, cfg, self.oof_logits(cfg))
        fold = np.random.default_rng(cfg['seed']).permutation(len(X)) % folds
        train_use = T._subset(tr, cfg['rank_train'])
        ok = []
        for k in range(folds):
            fit_m, test_m = train_use & (fold != k), tr['hard'] & (fold == k)
            r = M.fit_combiner(X[fit_m], tr['valid'][fit_m], tr['correct'][fit_m], cfg)
            s = np.where(tr['valid'][test_m], M.combine(r, X[test_m]), -1e9)
            ok.append(accuracy(M.select(s, [tr['answers'][i] for i in np.flatnonzero(test_m)]),
                               {'correct': tr['correct'][test_m]}))
        return 100*float(np.concatenate(ok).mean())


def fixed_scores(data, kind):
    """Training-free baselines expressed as candidate scores for the same cluster selector."""
    f = data['feats']
    col = {n: i for i, n in enumerate(F.NAMES)}
    if kind == 'majority':
        s = np.zeros(data['valid'].shape)
    elif kind == 'majority+valid':
        s = 20*(f[..., col['integer']] + f[..., col['nonneg']])
    elif kind == 'shortest':
        s = -20*f[..., col['log_len']]
    elif kind == 'longest':
        s = 20*f[..., col['log_len']]
    return np.where(data['valid'], s, -1e9)


BASE = ['answer', 'vote', 'surface', 'reasoning']
ONE = [T.LINEAR_CENTERED]
BOOST = {'rounds': 100, 'depth': 3, 'lr': 0.1, 'min_leaf': 30, 'l2': 10.0, 'bins': 32}

VARIANTS = {
    # name: config overrides relative to train.DEFAULT
    'votes+answer (no text, no emb)': {'probes': [], 'groups': ['answer', 'vote']},
    'votes+answer+surface': {'probes': [], 'groups': ['answer', 'vote', 'surface']},
    'votes+answer+reasoning': {'probes': [], 'groups': ['answer', 'vote', 'reasoning']},
    'all handcrafted - cross (no emb)': {'probes': [], 'groups': BASE},
    'all handcrafted (no emb)': {'probes': []},
    'probe only (+votes via lse)': {'probes': ONE, 'groups': []},
    'probe+votes+answer': {'probes': ONE, 'groups': ['answer', 'vote']},
    'single probe, no cross': {'probes': ONE, 'groups': BASE},
    'single probe, no cross - surface': {'probes': ONE, 'groups': ['answer', 'vote', 'reasoning']},
    'single probe, no cross - reasoning': {'probes': ONE, 'groups': ['answer', 'vote', 'surface']},
    'single probe, no cross + similarity(raw)': {'probes': ONE, 'groups': BASE, 'similarity': 'raw'},
    'single probe, no cross, probe trained on hard': {'probes': ONE, 'groups': BASE, 'probe_train': 'hard'},
    'single probe, no cross, ranker trained on all': {'probes': ONE, 'groups': BASE, 'rank_train': 'all'},
    'single probe: linear raw': {'probes': [T.LINEAR_RAW], 'groups': BASE},
    'single probe: linear raw+centered concat': {'probes': [{**T.LINEAR_CENTERED, 'input': 'both'}], 'groups': BASE},
    'single probe: MLP centered h=16': {'probes': [T.MLP_CENTERED], 'groups': BASE},
    'single probe: listwise centered': {'probes': [T.LISTWISE_CENTERED], 'groups': BASE},
    'single probe (with cross)': {'probes': ONE},
    'ensemble: centered + raw, no cross': {'groups': BASE},
    'ensemble: centered + MLP': {'probes': [T.LINEAR_CENTERED, T.MLP_CENTERED]},
    'ensemble: centered + listwise': {'probes': [T.LINEAR_CENTERED, T.LISTWISE_CENTERED]},
    'ensemble: all four, no cross': {'probes': [T.LINEAR_CENTERED, T.LINEAR_RAW, T.MLP_CENTERED, T.LISTWISE_CENTERED],
                                     'groups': BASE},
    'ensemble: all four': {'probes': [T.LINEAR_CENTERED, T.LINEAR_RAW, T.MLP_CENTERED, T.LISTWISE_CENTERED]},
    'handcrafted + n-gram text probe (no emb)': {'probes': [T.NGRAM]},
    'default + n-gram text probe': {'probes': [T.LINEAR_CENTERED, T.LINEAR_RAW, T.NGRAM]},
    'default + n-gram text probe (words only)': {'probes': [T.LINEAR_CENTERED, T.LINEAR_RAW,
                                                            {**T.NGRAM, 'tokens': 'words'}]},
    'default + vote-pattern interactions': {'interactions': True},
    'default + boosted trees d3': {'boost': BOOST},
    'default + boosted trees d2': {'boost': {**BOOST, 'depth': 2}},
    'default + interactions + boosted trees d3': {'interactions': True, 'boost': BOOST},
    'full (default)': {},
}

# configurations re-run over several seeds (seed sets both the CV folds and the MLP initialisation)
SEED_CHECK = ['single probe, no cross', 'single probe (with cross)', 'ensemble: centered + raw, no cross',
              'ensemble: centered + MLP', 'ensemble: all four, no cross', 'ensemble: all four',
              'handcrafted + n-gram text probe (no emb)', 'default + n-gram text probe',
              'default + vote-pattern interactions', 'default + boosted trees d3', 'default + boosted trees d2',
              'default + interactions + boosted trees d3', 'full (default)']


def seed_check(R, seeds=(1, 2, 3, 4, 5)):
    va, out = R.va, {}
    for name in SEED_CHECK:
        val, cv = [], []
        for seed in seeds:
            cfg = {**T.DEFAULT, **VARIANTS[name], 'seed': seed}
            choices = M.select(M.score(R.bundle(cfg), va['feats'], va['emb'], va['answers'], va['texts']), va['answers'])
            val.append(100*accuracy(choices, va)[va['hard']].mean())
            cv.append(R.ranker_cv(cfg))
        out[name] = {'seeds': list(seeds), 'val_hard': val, 'train_hard_cv': cv,
                     'val_hard_mean': float(np.mean(val)), 'val_hard_std': float(np.std(val)),
                     'cv_mean': float(np.mean(cv)), 'cv_std': float(np.std(cv))}
        print(f"seeds {name:44s} val-hard {np.mean(val):.1f}±{np.std(val):.1f}  cv {np.mean(cv):.1f}±{np.std(cv):.1f}",
              flush=True)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('runs/experiments/results.json'))
    parser.add_argument('--sweep', action='store_true', help='also sweep probe/ranker L2')
    parser.add_argument('--seeds', action='store_true', help='also re-run top configurations over 5 seeds')
    args = parser.parse_args()
    t0 = time.perf_counter()
    R = Runner()
    va = R.va
    H = va['hard']
    rows, keep = [], {}

    def record(name, choices, cv=None, extra=None):
        ok = accuracy(choices, va)
        keep[name] = ok
        row = {'method': name, 'val_hard': 100*ok[H].mean(), 'val_hard_ci95': bootstrap_ci(ok[H]),
               'val_all': 100*ok.mean(), 'train_hard_cv': cv, **(extra or {})}
        rows.append(row)
        print(f"{name:44s} val-hard {row['val_hard']:5.1f} [{row['val_hard_ci95'][0]:.1f},{row['val_hard_ci95'][1]:.1f}]"
              f"  val-all {row['val_all']:5.1f}  train-hard-cv {cv if cv is None else round(cv, 1)}", flush=True)

    record('first candidate', [0]*len(H))
    for kind in ('shortest', 'longest', 'majority', 'majority+valid'):
        record(kind, M.select(fixed_scores(va, kind), va['answers']))

    configs = {name: {**T.DEFAULT, **over} for name, over in VARIANTS.items()}
    if args.sweep:
        for pl2 in (1.0, 10.0, 100.0):
            for rl2 in (0.1, 1.0, 10.0):
                configs[f'sweep probe_l2={pl2} rank_l2={rl2}'] = {**T.DEFAULT, 'rank_l2': rl2,
                                                                  'probes': [{**T.LINEAR_CENTERED, 'l2': pl2}]}
    bundles = {}
    for name, cfg in configs.items():
        b = R.bundle(cfg)
        bundles[name] = b
        t = time.perf_counter()
        s = M.score(b, va['feats'], va['emb'], va['answers'], va['texts'])
        record(name, M.select(s, va['answers']), R.ranker_cv(cfg),
               {'train_seconds': b['train_seconds'], 'score_seconds': time.perf_counter() - t,
                'n_params': int(sum(np.size(v) for k, v in b.items() if k.startswith(('p', 'rank_'))))})

    # single probe as a per-candidate classifier (no vote aggregation): argmax of its logit
    b1 = bundles['single probe, no cross']
    Xp = M.probe_inputs(va['emb'], va['valid'], 'centered')
    logit = np.where(va['valid'], M.probe_logit(M.probe_params(b1, 0), Xp), -1e9)
    record('probe argmax (no cluster lse)', [int(i) for i in logit.argmax(1)])
    b = bundles['full (default)']

    # ---- diagnostics for the default model ----
    full = keep['full (default)']
    s = M.score(b, va['feats'], va['emb'], va['answers'], va['texts'])
    conf = M.margins(s, va['answers'])[H]
    edges = np.quantile(conf, [0, .2, .4, .6, .8, 1])
    bins = np.clip(np.searchsorted(edges, conf, side='right') - 1, 0, 4)
    calibration = [{'conf_lo': float(edges[k]), 'conf_hi': float(edges[k+1]), 'mean_conf': float(conf[bins == k].mean()),
                    'acc': float(100*full[H][bins == k].mean()), 'n': int((bins == k).sum())} for k in range(5)]
    tests = {other: dict(zip(('only_full_right', 'only_other_right', 'p'), mcnemar(keep[other][H], full[H])))
             for other in ('majority', 'majority+valid', 'all handcrafted (no emb)', 'probe only (+votes via lse)',
                           'single probe, no cross', 'ensemble: centered + raw, no cross')}
    pattern = {}
    for i in np.flatnonzero(H):
        counts = sorted((sum(a == x for a in va['answers'][i]) for x in {a for a in va['answers'][i] if a}), reverse=True)
        key = str(tuple(counts))
        p = pattern.setdefault(key, {'n': 0, 'majority': 0, 'full': 0})
        p['n'] += 1; p['majority'] += keep['majority'][i]; p['full'] += full[i]
    col = F.NAMES.index('log_len')
    length_corr = float(np.corrcoef(logit[va['valid']], va['feats'][..., col][va['valid']])[0, 1])
    names = ([n for g in b['config']['groups'] for n in F.GROUPS[g]]
             + (M.SIMILARITY if b['config']['similarity'] else [])
             + [f'p{k}{suffix}' for k in range(len(b['config']['probes'])) for suffix in ('', '_rel')])
    weights = dict(zip(names, map(float, b['rank_w'])))
    diagnostics = {'calibration_val_hard': calibration, 'mcnemar_vs_full': tests, 'vote_patterns': pattern,
                   'probe_logit_vs_log_len_corr': length_corr,
                   'seed_check': seed_check(R) if args.seeds else None, 'ranker_weights_raw_scale': weights,
                   'total_seconds': time.perf_counter() - t0}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'rows': rows, 'diagnostics': diagnostics}, indent=2) + '\n')
    print(json.dumps(diagnostics, indent=2))


if __name__ == '__main__':
    main()
