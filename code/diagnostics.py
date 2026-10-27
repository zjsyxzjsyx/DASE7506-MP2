"""Report diagnostics on train (CV) and validation only; never opens test references.

Covers: the val-hard vs train-hard CV gap, cross-feature statistics, disagreement between methods, the
structure of the remaining errors, length-matched pairwise comparisons, standardised ranker weights and a
few validation failure examples. Writes runs/diagnostics/{diagnostics.json, failure_examples.md}.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import re

import numpy as np

from common import parse_answer
import features as F
import model as M
import train as T
from experiments import VARIANTS, Runner, accuracy, fixed_scores


def pattern_of(row):
    return tuple(sorted(Counter(a for a in row if a is not None).values(), reverse=True))


def cv_per_group(R, cfg, frac=1.0, folds=5):
    """Per-group correctness of ranker CV on train-hard; frac < 1 subsamples each training fold."""
    tr = R.tr
    X = T.ranker_matrix(tr, cfg, R.oof_logits(cfg))
    rng = np.random.default_rng(cfg['seed'])
    fold = rng.permutation(len(X)) % folds
    ok = np.full(len(X), np.nan)
    for k in range(folds):
        fit_m = tr['hard'] & (fold != k)
        if frac < 1:
            fit_m &= rng.random(len(X)) < frac
        test_m = tr['hard'] & (fold == k)
        r = M.fit_combiner(X[fit_m], tr['valid'][fit_m], tr['correct'][fit_m], cfg)
        s = np.where(tr['valid'][test_m], M.combine(r, X[test_m]), -1e9)
        ok[test_m] = accuracy(M.select(s, [tr['answers'][i] for i in np.flatnonzero(test_m)]),
                              {'correct': tr['correct'][test_m]})
    return ok


def clusters(s, row):
    """[(cluster score, answer, member indices)] sorted by score, highest first."""
    members = {}
    for i, a in enumerate(row):
        if a is not None:
            members.setdefault(a, []).append(i)
    return sorted(((float(np.logaddexp.reduce(s[idx])), a, idx) for a, idx in members.items()), reverse=True)


def mean(x):
    return float(np.mean(x)) if len(x) else float('nan')


def gap_analysis(R, cfg, ok_va):
    tr, va = R.tr, R.va
    ok_tr = cv_per_group(R, cfg)
    maj_tr = accuracy(M.select(fixed_scores(tr, 'majority'), tr['answers']), tr)
    maj_va = accuracy(M.select(fixed_scores(va, 'majority'), va['answers']), va)
    Ht, Hv = np.flatnonzero(tr['hard']), np.flatnonzero(va['hard'])
    pt = [pattern_of(tr['answers'][i]) for i in Ht]
    pv = [pattern_of(va['answers'][i]) for i in Hv]
    table = {}
    for p in sorted(set(pt) | set(pv), key=lambda p: -pt.count(p)):
        it = [i for i, q in zip(Ht, pt) if q == p]
        iv = [i for i, q in zip(Hv, pv) if q == p]
        table[str(p)] = {'train_n': len(it), 'val_n': len(iv), 'train_share': len(it)/len(Ht), 'val_share': len(iv)/len(Hv),
                         'majority_train': mean(maj_tr[it]), 'majority_val': mean(maj_va[iv]),
                         'model_cv': mean(ok_tr[it]), 'model_val': mean(ok_va[iv])}
    w = {p: pv.count(p)/len(pv) for p in set(pv)}
    reweighted = sum(w[p]*mean(ok_tr[[i for i, q in zip(Ht, pt) if q == p]]) for p in w)
    small = cv_per_group(R, cfg, frac=0.8)
    a_va, a_tr = ok_va[Hv].mean(), ok_tr[Ht].mean()
    return {'per_pattern': table,
            'majority_train_hard': float(maj_tr[Ht].mean()), 'majority_val_hard': float(maj_va[Hv].mean()),
            'model_cv_train_hard': float(a_tr), 'model_val_hard': float(a_va),
            'cv_reweighted_to_val_pattern_mix': float(reweighted),
            'cv_with_80pct_training_folds': float(np.nanmean(small[Ht])),
            'se_val_hard': float(np.sqrt(a_va*(1 - a_va)/len(Hv))), 'se_cv': float(np.sqrt(a_tr*(1 - a_tr)/len(Ht)))}


def cross_stats(data):
    m = data['hard'][:, None] & data['valid']
    y = data['correct'][m]
    out = {}
    for name in F.GROUPS['cross']:
        x = data['feats'][..., F.NAMES.index(name)][m] > 0
        out[name] = {'n_present': int(x.sum()), 'p_correct_present': mean(y[x]),
                     'n_absent': int((~x).sum()), 'p_correct_absent': mean(y[~x])}
    return out


_BOXED = re.compile(r'\\boxed\{([^{}]+)\}')


def parser_hazard(data):
    """Candidates whose official parse differs from their last \\boxed{} answer (a markdown '####' header
    makes common.parse_answer read the number after the header instead)."""
    def boxed(text):
        found = _BOXED.findall(text)
        return parse_answer('\\boxed{' + found[-1] + '}') if found else None
    mismatch, hard_groups, boxed_right, became_easy = 0, set(), 0, 0
    for g, row in enumerate(data['texts']):
        right = {data['answers'][g][k] for k in range(4) if data['correct'][g, k]}
        for i, t in enumerate(row):
            b = boxed(t)
            if b is not None and b != data['answers'][g][i]:
                mismatch += 1
                if data['hard'][g]:
                    hard_groups.add(g)
                    boxed_right += b in right
        if data['hard'][g]:
            alt = [boxed(t) or data['answers'][g][i] for i, t in enumerate(row)]
            became_easy += len({a for a in alt if a is not None}) < 2
    return {'candidates': sum(len(r) for r in data['texts']),
            'candidates_with_####': sum('####' in t for r in data['texts'] for t in r),
            'parsed_differs_from_boxed': mismatch, 'hard_groups_affected': len(hard_groups),
            'in_hard_groups_boxed_answer_was_correct': boxed_right,
            'hard_groups_with_one_answer_if_parsed_from_boxed': became_easy}


def length_pairs(va, scores):
    """Pairwise accuracy on (correct, wrong) candidate pairs of the same val-hard group, by length difference."""
    loglen = va['feats'][..., F.NAMES.index('log_len')]
    pairs = []
    for g in np.flatnonzero(va['hard']):
        for i in range(4):
            for j in range(4):
                if va['valid'][g, i] and va['valid'][g, j] and va['correct'][g, i] and not va['correct'][g, j]:
                    pairs.append((g, i, j, loglen[g, i] - loglen[g, j]))
    g, i, j, d = (np.array(x) for x in zip(*pairs))
    subsets = {'all': np.ones(len(d), bool), 'length_matched_|dlog|<0.1': np.abs(d) < 0.1,
               'correct_longer_dlog>0.1': d > 0.1, 'correct_shorter_dlog<-0.1': d < -0.1}
    out = {}
    for name, sel in subsets.items():
        row = {'n_pairs': int(sel.sum())}
        for method, s in scores.items():
            row[method] = mean((s[g, i] > s[g, j])[sel])
        out[name] = row
    return out


def standardised_weights(R, cfg, bundle):
    tr = R.tr
    X = T.ranker_matrix(tr, cfg, R.oof_logits(cfg))
    rows = X[tr['hard']][tr['valid'][tr['hard']]]
    names = ([n for g in cfg['groups'] for n in F.GROUPS[g]] + (M.SIMILARITY if cfg.get('similarity') else [])
             + [f'p{k}{suffix}' for k in range(len(cfg['probes'])) for suffix in ('', '_rel')])
    if cfg.get('interactions'):
        names += [f'{x}*pattern{j}' for x in ['vote_share'] + [f'p{k}_rel' for k in range(len(cfg['probes']))]
                  for j in range(len(M.PATTERNS) + 1)]
    std = rows.std(0)
    w = bundle['rank_w']*std
    order = np.argsort(-np.abs(w))
    return [{'feature': names[k], 'std_weight': float(w[k]), 'raw_weight': float(bundle['rank_w'][k])} for k in order]


def failure_examples(va, s, ok, out_path, per_kind=1):
    """Markdown with a few val-hard errors: one minority-correct split, one tie, one all-distinct group."""
    wanted = {'(3, 1)': 'minority answer was right', '(2, 2)': 'tie between two pairs',
              '(2, 1, 1)': 'right answer outside the largest cluster', '(1, 1, 1, 1)': 'four different answers'}
    picked = []
    for g in np.flatnonzero(va['hard'] & (ok == 0)):
        p = str(pattern_of(va['answers'][g]))
        if p in wanted and sum(q == p for q, _ in picked) < per_kind:
            picked.append((p, g))
    lines = ['# Validation failure examples (final model)', '',
             'Chosen automatically: the first val-hard error of each listed vote pattern.', '']
    for p, g in picked:
        cl = clusters(s[g], va['answers'][g])
        prob = np.exp(np.array([c[0] for c in cl]) - cl[0][0]); prob /= prob.sum()
        lines += [f'## Pattern {p}: {wanted[p]}', '', '**Question.** ' + va['bank'][g]['question'], '',
                  '| answer | votes | cluster prob. | correct |', '|---|---|---|---|']
        for (score, a, idx), pr in zip(cl, prob):
            lines.append(f'| {a} | {len(idx)} | {pr:.2f} | {"yes" if va["correct"][g, idx[0]] else "no"} |')
        chosen = cl[0][2][int(np.argmax(s[g][cl[0][2]]))]
        right = [i for i in range(4) if va['correct'][g, i]][0]
        for label, i in (('Chosen (wrong) candidate', chosen), ('A correct candidate', right)):
            text = va['texts'][g][i].strip()
            note = ' — contains a "####" header, so the official parser reads the number after it' if '####' in text else ''
            lines += ['', f'**{label} (parsed answer {va["answers"][g][i]}{note}), last 600 characters:**', '',
                      '```text', text[-600:], '```']
        lines.append('')
    out_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return [{'pattern': p, 'group_index': int(g)} for p, g in picked]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', type=Path, default=Path('runs/model_bundle'))
    parser.add_argument('--output', type=Path, default=Path('runs/diagnostics'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    R = Runner()
    tr, va = R.tr, R.va
    bundle = M.load(args.bundle)
    cfg = {**T.DEFAULT, **bundle['config']}
    s = M.score(bundle, va['feats'], va['emb'], va['answers'], va['texts'])
    ok = accuracy(M.select(s, va['answers']), va)
    H = va['hard']

    others = {}
    for name in ('all handcrafted (no emb)', 'probe only (+votes via lse)'):
        b = R.bundle({**T.DEFAULT, **VARIANTS[name]})
        others[name] = M.score(b, va['feats'], va['emb'], va['answers'], va['texts'])
    ok_other = {k: accuracy(M.select(v, va['answers']), va) for k, v in others.items()}
    disagreement = {}
    for k, o in ok_other.items():
        agree = (np.array(M.select(others[k], va['answers'])) == np.array(M.select(s, va['answers'])))[H]
        disagreement[k] = {'oracle_either_right': mean(np.maximum(ok, o)[H]), 'other_acc': mean(o[H]),
                           'same_choice_rate': mean(agree), 'final_acc_when_same': mean(ok[H][agree]),
                           'final_acc_when_different': mean(ok[H][~agree]), 'other_acc_when_different': mean(o[H][~agree])}

    conf = M.margins(s, va['answers'])
    rank, sizes = Counter(), Counter()
    for g in np.flatnonzero(H & (ok == 0)):
        cl = clusters(s[g], va['answers'][g])
        pos = [k for k, c in enumerate(cl) if va['correct'][g, c[2][0]]][0]
        rank[pos + 1] += 1
        sizes[str((len(cl[pos][2]), len(cl[0][2])))] += 1
    singleton = [g for g in np.flatnonzero(H) if va['correct'][g].sum() == 1]
    errors = {'n_errors': int((H & (ok == 0)).sum()), 'n_val_hard': int(H.sum()),
              'rank_of_correct_cluster': dict(rank),
              '(correct cluster size, chosen cluster size)': dict(sizes),
              'mean_conf_errors': mean(conf[H & (ok == 0)]), 'mean_conf_correct': mean(conf[H & (ok == 1)]),
              'singleton_correct_groups': len(singleton), 'singleton_correct_acc': mean(ok[singleton])}

    loglen = va['feats'][..., F.NAMES.index('log_len')]
    pair_scores = {'final': s, 'handcrafted_only': others['all handcrafted (no emb)'],
                   'probe_only': others['probe only (+votes via lse)'], 'longest': loglen}
    result = {'gap': gap_analysis(R, cfg, ok), 'cross_stats_val_hard': cross_stats(va),
              'cross_stats_train_hard': cross_stats(tr), 'disagreement_val_hard': disagreement,
              'parser_hazard_train': parser_hazard(tr), 'parser_hazard_validation': parser_hazard(va),
              'errors_val_hard': errors, 'length_pairs_val_hard': length_pairs(va, pair_scores),
              'standardised_ranker_weights': standardised_weights(R, cfg, bundle),
              'failure_examples': failure_examples(va, s, ok, args.output/'failure_examples.md')}
    (args.output/'diagnostics.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'standardised_ranker_weights'}, indent=2))
    print('top standardised weights:', [(w['feature'], round(w['std_weight'], 2)) for w in result['standardised_ranker_weights'][:12]])


if __name__ == '__main__':
    main()
