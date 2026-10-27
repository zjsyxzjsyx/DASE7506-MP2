"""NumPy model pieces: embedding probe, listwise answer ranker and answer-cluster selection."""
from collections import Counter
import json
from pathlib import Path

import numpy as np

import features as F
import textmodel


def load(path):
    path = Path(path)
    with np.load(path/'weights.npz') as f:
        bundle = {k: f[k] for k in f.files}
    bundle['config'] = json.loads((path/'config.json').read_text())
    return bundle


def parsable_mask(answers):
    return np.array([[a is not None for a in row] for row in answers])


def hard_mask(answers):
    return np.array([len({a for a in row if a is not None}) >= 2 for row in answers])


def adam(grad_fn, params, steps, lr):
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    for t in range(1, steps + 1):
        for p, g, mi, vi in zip(params, grad_fn(params), m, v):
            mi *= 0.9; mi += 0.1*g
            vi *= 0.999; vi += 0.001*g*g
            p -= lr*(mi/(1 - 0.9**t))/(np.sqrt(vi/(1 - 0.999**t)) + 1e-8)
    return params


# ---- embedding inputs -------------------------------------------------------

def _group_center(emb, valid):
    w = valid[..., None].astype(np.float32)
    return emb - (emb*w).sum(1, keepdims=True)/np.maximum(w.sum(1, keepdims=True), 1)


def probe_inputs(emb, valid, mode):
    """[N,4,1024] -> [N,4,D]; 'centered' subtracts the mean of the group's parsable candidates."""
    if mode == 'raw':
        return emb
    centered = _group_center(emb, valid)
    return centered if mode == 'centered' else np.concatenate([emb, centered], -1)


SIMILARITY = ['sim_same_mean', 'sim_diff_mean', 'sim_diff_max', 'sim_others_mean']


def similarity_features(emb, answers, mode):
    """Cosine similarity of each candidate to same-answer and different-answer candidates in its group."""
    valid = parsable_mask(answers)
    e = _group_center(emb, valid) if mode == 'centered' else emb
    e = e/(np.linalg.norm(e, axis=-1, keepdims=True) + 1e-6)
    S = np.einsum('nid,njd->nij', e, e)
    ids = np.array([[row.index(a) if a is not None else 4 + i for i, a in enumerate(row)] for row in answers])
    both = valid[:, :, None] & valid[:, None, :] & ~np.eye(4, dtype=bool)
    same = both & (ids[:, :, None] == ids[:, None, :])
    diff = both & ~same

    def mean(mask):
        return (S*mask).sum(-1)/np.maximum(mask.sum(-1), 1)

    out = np.stack([mean(same), mean(diff), np.where(diff, S, -1).max(-1)*diff.any(-1), mean(both)], -1)
    return np.where(valid[..., None], out, 0).astype(np.float32)


def fit_probe(X, y, l2, steps=200, lr=0.05):
    """L2 logistic regression on standardised inputs; returns weights folded back to raw inputs."""
    mu, sd = X.mean(0), X.std(0) + 1e-4
    Z = ((X - mu)/sd).astype(np.float32)
    y = y.astype(np.float32)
    n = len(y)

    def grad(params):
        w, b = params
        p = 1/(1 + np.exp(-np.clip(Z@w + b, -30, 30)))
        return [Z.T@(p - y)/n + l2*w/n, np.array([(p - y).mean()], np.float32)]

    w, b = adam(grad, [np.zeros(Z.shape[1], np.float32), np.zeros(1, np.float32)], steps, lr)
    w_raw = (w/sd).astype(np.float32)
    return {'probe_w': w_raw, 'probe_b': (b - mu@w_raw).astype(np.float32)}


def fit_mlp_probe(X, y, hidden, l2, seed, steps=300, lr=0.003):
    """One-hidden-layer ReLU probe trained full-batch with BCE + L2; standardisation folded into W1."""
    mu, sd = X.mean(0), X.std(0) + 1e-4
    Z = ((X - mu)/sd).astype(np.float32)
    y = y.astype(np.float32)
    n, d = Z.shape
    rng = np.random.default_rng(seed)
    params = [rng.normal(0, np.sqrt(2/d), (d, hidden)).astype(np.float32), np.zeros(hidden, np.float32),
              rng.normal(0, np.sqrt(1/hidden), hidden).astype(np.float32), np.zeros(1, np.float32)]

    def grad(params):
        W1, b1, w2, b2 = params
        h = Z@W1 + b1
        a = np.maximum(h, 0)
        p = 1/(1 + np.exp(-np.clip(a@w2 + b2, -30, 30)))
        dl = (p - y)/n
        dh = np.outer(dl, w2)*(h > 0)
        return [Z.T@dh + l2*W1/n, dh.sum(0), a.T@dl + l2*w2/n, np.array([dl.sum()], np.float32)]

    W1, b1, w2, b2 = adam(grad, params, steps, lr)
    W1_raw = (W1/sd[:, None]).astype(np.float32)
    return {'probe_W1': W1_raw, 'probe_b1': (b1 - mu@W1_raw).astype(np.float32), 'probe_w2': w2, 'probe_b2': b2}


def fit_listwise_probe(X, valid, correct, l2):
    """Linear probe trained with the same listwise objective as the ranker (hard groups only carry signal)."""
    r = fit_ranker(X, valid, correct, l2, steps=300)
    return {'probe_w': r['rank_w'], 'probe_b': np.atleast_1d(r['rank_b'])}


def probe_params(bundle, k):
    prefix = f'p{k}_'
    return {key[len(prefix):]: v for key, v in bundle.items() if key.startswith(prefix)}


def probe_logit(params, X):
    if 'probe_W1' in params:
        return np.maximum(X@params['probe_W1'] + params['probe_b1'], 0)@params['probe_w2'] + params['probe_b2']
    return X@params['probe_w'] + params['probe_b']


# ---- listwise ranker --------------------------------------------------------

# Vote patterns of hard groups (sorted cluster sizes); every other group falls in the last slot.
PATTERNS = [(3, 1), (2, 1, 1), (2, 2), (2, 1), (1, 1, 1, 1), (1, 1, 1), (1, 1)]


def vote_patterns(answers):
    """Group-level one-hot of the vote pattern, [N, len(PATTERNS)+1]."""
    out = np.zeros((len(answers), len(PATTERNS) + 1), np.float32)
    for g, row in enumerate(answers):
        sizes = tuple(sorted(Counter(a for a in row if a is not None).values(), reverse=True))
        out[g, PATTERNS.index(sizes) if sizes in PATTERNS else len(PATTERNS)] = 1
    return out


def ranker_inputs(feats, logits, valid, cols, extra=None, patterns=None):
    """Handcrafted columns, optional extra columns, and each probe logit with its deviation from the group mean.

    With `patterns`, the vote share and each probe deviation are also crossed with the vote-pattern one-hot,
    so the ranker can trust votes and probes differently in a 2-2 tie than in a 3-1 split.
    """
    parts = [feats[..., cols]]
    if extra is not None:
        parts.append(extra)
    rels = []
    for logit in logits:
        w = valid.astype(np.float32)
        mean = (logit*w).sum(1, keepdims=True)/np.maximum(w.sum(1, keepdims=True), 1)
        rels.append(logit - mean)
        parts += [logit[..., None], rels[-1][..., None]]
    if patterns is not None:
        for x in [feats[..., F.NAMES.index('vote_share')]] + rels:
            parts.append(x[..., None]*patterns[:, None, :])
    return np.concatenate(parts, -1).astype(np.float32)


def listwise_grad(scores, valid, correct):
    """Loss -log P(correct answer cluster) with cluster score logsumexp of member scores.

    Equivalent to -(lse over correct candidates - lse over parsable candidates); returns d loss/d scores.
    """
    s = np.where(valid, scores, -np.inf)
    p = np.exp(s - s.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    q = np.where(correct, p, 0); q /= q.sum(1, keepdims=True)
    return np.where(valid, p - q, 0)


def fit_ranker(X, valid, correct, l2, steps=400, lr=0.05):
    use = (correct & valid).any(1) & valid.any(1)
    X, valid, correct = X[use], valid[use], correct[use]
    mu = X[valid].mean(0); sd = X[valid].std(0) + 1e-4
    Z = (X - mu)/sd
    n = len(Z)

    def grad(params):
        w, = params
        g = listwise_grad(Z@w, valid, correct)
        return [np.einsum('nkd,nk->d', Z, g)/n + l2*w/n]

    w, = adam(grad, [np.zeros(Z.shape[-1], np.float32)], steps, lr)
    w_raw = (w/sd).astype(np.float32)
    return {'rank_w': w_raw, 'rank_b': np.float32(-mu@w_raw)}


# ---- gradient-boosted trees on top of the linear ranker ----------------------

def _tree_predict(X, feat, thr, leaf, depth):
    node = np.zeros(X.shape[:-1], np.int64)
    for _ in range(depth):
        f = feat[node].astype(np.int64)
        node = 2*node + 1 + (np.take_along_axis(X, f[..., None], -1)[..., 0] > thr[node])
    return leaf[node - (2**depth - 1)]


def fit_boost(X, valid, correct, base, rounds, depth, lr, min_leaf, l2, bins):
    """Newton-boosted regression trees fitted to the listwise gradient, starting from the linear scores.

    Trees are complete binary trees of fixed depth stored as arrays; a node without a useful split sends
    every row left (threshold +inf).
    """
    use = (correct & valid).any(1) & valid.any(1)
    X, valid, correct, base = X[use], valid[use], correct[use], base[use]
    rows = np.argwhere(valid)
    R = X[valid]
    d = R.shape[1]
    edges = [np.unique(np.quantile(R[:, j], np.linspace(0, 1, bins + 1)[1:-1])) for j in range(d)]
    B = np.stack([np.searchsorted(edges[j], R[:, j], side='right') for j in range(d)], 1)
    width = bins + 1
    offset = np.arange(d)*width
    n_inner = 2**depth - 1
    feats = np.zeros((rounds, n_inner), np.int64)
    thrs = np.full((rounds, n_inner), np.inf, np.float32)
    leaves = np.zeros((rounds, 2**depth), np.float32)
    score = base.astype(np.float32).copy()
    for t in range(rounds):
        g_full = listwise_grad(score, valid, correct)
        s = np.where(valid, score, -np.inf)
        p = np.exp(s - s.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
        g, h = g_full[valid], (p*(1 - p))[valid] + 1e-6
        node = np.zeros(len(R), np.int64)
        for level in range(depth):
            for k in range(2**level - 1, 2**(level + 1) - 1):
                idx = np.flatnonzero(node == k)
                if len(idx) < 2*min_leaf:
                    continue
                flat = (B[idx] + offset).ravel()
                G = np.bincount(flat, np.repeat(g[idx], d), d*width).reshape(d, width).cumsum(1)
                H = np.bincount(flat, np.repeat(h[idx], d), d*width).reshape(d, width).cumsum(1)
                C = np.bincount(flat, minlength=d*width).reshape(d, width).cumsum(1)
                Gt, Ht, Ct = G[:, -1:], H[:, -1:], C[:, -1:]
                gain = G**2/(H + l2) + (Gt - G)**2/(Ht - H + l2) - Gt**2/(Ht + l2)
                gain[(C < min_leaf) | (Ct - C < min_leaf)] = -np.inf
                j, b = np.unravel_index(np.argmax(gain), gain.shape)
                if not np.isfinite(gain[j, b]) or gain[j, b] <= 0 or b >= len(edges[j]):
                    continue
                feats[t, k], thrs[t, k] = j, edges[j][b]
            node = 2*node + 1 + (R[np.arange(len(R)), feats[t][node]] > thrs[t][node])
        leaf_id = node - n_inner
        leaves[t] = -lr*np.bincount(leaf_id, g, 2**depth)/(np.bincount(leaf_id, h, 2**depth) + l2)
        update = np.zeros(valid.shape, np.float32)
        update[rows[:, 0], rows[:, 1]] = leaves[t][leaf_id]
        score = score + update
    return {'tree_feat': feats, 'tree_thr': thrs, 'tree_leaf': leaves}


def fit_combiner(X, valid, correct, cfg):
    model = fit_ranker(X, valid, correct, cfg['rank_l2'])
    if cfg.get('boost'):
        model.update(fit_boost(X, valid, correct, X@model['rank_w'] + model['rank_b'], **cfg['boost']))
    return model


def combine(model, X):
    s = X@model['rank_w'] + model['rank_b']
    if 'tree_feat' in model:
        depth = int(np.log2(model['tree_leaf'].shape[1]))
        for f, t, l in zip(model['tree_feat'], model['tree_thr'], model['tree_leaf']):
            s = s + _tree_predict(X, f, t, l, depth)
    return s


def select(scores, answers):
    """Pick the answer cluster with the largest logsumexp score; return one member index per group."""
    choices = []
    for s, row in zip(scores, answers):
        clusters = {}
        for i, a in enumerate(row):
            if a is not None:
                clusters.setdefault(a, []).append(i)
        if not clusters:
            choices.append(int(np.argmax(s)))
            continue
        best = max(clusters.values(), key=lambda idx: (np.logaddexp.reduce(s[idx]), -idx[0]))
        choices.append(int(max(best, key=lambda i: s[i])))
    return choices


def margins(scores, answers):
    """Probability of the chosen cluster under the cluster softmax (confidence)."""
    out = []
    for s, row in zip(scores, answers):
        clusters = {}
        for i, a in enumerate(row):
            if a is not None:
                clusters.setdefault(a, []).append(s[i])
        if not clusters:
            out.append(0.0)
            continue
        c = np.array([np.logaddexp.reduce(v) for v in clusters.values()])
        p = np.exp(c - c.max()); p /= p.sum()
        out.append(float(p.max()))
    return np.array(out)


def score(bundle, feats, emb, answers, texts=None):
    """Candidate scores from a loaded bundle (dict of arrays + config); texts is [N][4] candidate text."""
    cfg = bundle['config']
    valid = parsable_mask(answers)
    logits = [textmodel.logits(probe_params(bundle, k), texts, spec) if spec['model'] == 'ngram'
              else probe_logit(probe_params(bundle, k), probe_inputs(emb, valid, spec['input']))
              for k, spec in enumerate(cfg['probes'])]
    extra = similarity_features(emb, answers, cfg['similarity']) if cfg.get('similarity') else None
    patterns = vote_patterns(answers) if cfg.get('interactions') else None
    X = ranker_inputs(feats, logits, valid, F.columns(cfg['groups']), extra, patterns)
    return np.where(valid, combine(bundle, X), -1e9)
