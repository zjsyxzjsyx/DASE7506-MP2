"""Hashed n-gram text probe trained from scratch on training texts: no vocabulary, no pretrained parts.

Texts become sparse log-count vectors over crc32-hashed token n-grams (deterministic across processes),
L2-normalised per candidate, and a logistic regression is fitted with plain NumPy sparse products.
"""
from collections import Counter
import re
import zlib

import numpy as np

_TOKEN_RE = re.compile(r'[a-z]+|\d[\d,]*(?:\.\d+)?|\\[a-z]+|#+|\*\*|\$\$|[=+\-*/×÷%$(){}^<>≈]')
_FORMAT = re.compile(r'#+|\*\*|\$\$|\\(?:boxed|text|textbf|left|right|quad|qquad)$')


def tokens(text, mode):
    """mode 'all': words, numbers as N, maths symbols, layout marks; 'words': words only;
    'noformat': 'all' without layout/markdown marks."""
    out = []
    for t in _TOKEN_RE.findall(text.lower()):
        if t[0].isdigit():
            if mode == 'words':
                continue
            t = 'N'
        elif not t[0].isalpha():
            if mode == 'words' or (mode == 'noformat' and _FORMAT.fullmatch(t)):
                continue
        out.append(t)
    return out


def featurize(texts, mode, n, dim):
    """Flat list of texts -> sparse matrix as dict(doc, col, val, n_docs)."""
    docs, cols, vals = [], [], []
    for d, text in enumerate(texts):
        toks = tokens(text, mode)
        grams = Counter(zlib.crc32(' '.join(toks[i:i + k]).encode()) & (dim - 1)
                        for k in range(1, n + 1) for i in range(len(toks) - k + 1))
        if not grams:
            continue
        c = np.fromiter(grams.keys(), np.int64, len(grams))
        v = np.log1p(np.fromiter(grams.values(), np.float32, len(grams)))
        docs.append(np.full(len(c), d, np.int64)); cols.append(c); vals.append(v/np.linalg.norm(v))
    cat = (lambda xs, dt: np.concatenate(xs) if xs else np.zeros(0, dt))
    return {'doc': cat(docs, np.int64), 'col': cat(cols, np.int64), 'val': cat(vals, np.float32),
            'n_docs': len(texts)}


def matvec(S, w):
    return np.bincount(S['doc'], S['val']*w[S['col']], minlength=S['n_docs']).astype(np.float32)


def subset(S, keep):
    """Rows of S where keep[doc] is true, renumbered."""
    new_id = np.cumsum(keep) - 1
    m = keep[S['doc']]
    return {'doc': new_id[S['doc'][m]], 'col': S['col'][m], 'val': S['val'][m], 'n_docs': int(keep.sum())}


def fit(S, y, dim, l2, steps=200, lr=0.05):
    """L2 logistic regression on a sparse design matrix, full-batch Adam."""
    y = y.astype(np.float32)
    n = len(y)
    w = np.zeros(dim, np.float32); b = 0.0
    m = np.zeros(dim, np.float32); v = np.zeros(dim, np.float32); mb = vb = 0.0
    for t in range(1, steps + 1):
        p = 1/(1 + np.exp(-np.clip(matvec(S, w) + b, -30, 30)))
        r = (p - y)/n
        g = np.bincount(S['col'], S['val']*r[S['doc']], minlength=dim).astype(np.float32) + l2*w/n
        gb = float(r.sum())
        m = 0.9*m + 0.1*g; v = 0.999*v + 0.001*g*g
        w -= lr*(m/(1 - 0.9**t))/(np.sqrt(v/(1 - 0.999**t)) + 1e-8)
        mb = 0.9*mb + 0.1*gb; vb = 0.999*vb + 0.001*gb*gb
        b -= lr*(mb/(1 - 0.9**t))/(np.sqrt(vb/(1 - 0.999**t)) + 1e-8)
    return {'probe_w': w, 'probe_b': np.array([b], np.float32)}


def logits(params, texts, spec):
    """[N,4] texts -> [N,4] probe logits."""
    flat = [t for row in texts for t in row]
    S = featurize(flat, spec['tokens'], spec['n'], spec['dim'])
    return (matvec(S, params['probe_w']) + params['probe_b']).reshape(len(texts), 4)
