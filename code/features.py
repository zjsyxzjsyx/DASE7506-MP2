"""Candidate features shared by training and inference; reads only bank text, never references."""
import ast
from collections import Counter
from fractions import Fraction
import math
import operator
import re

import numpy as np

from common import parse_answer

_NUM_RE = re.compile(r'(?<![\w.])-?\d[\d,]*(?:\.\d+)?')
_FRAC_RE = re.compile(r'\\[dt]?frac\{([^{}]+)\}\{([^{}]+)\}')
_EQ_RE = re.compile(r'([\d.\s+\-*/()]*\d[\d.\s+\-*/()]*)=\s*(-?\d+(?:\.\d+)?)(?![\d.]*\s*[*/+\-]?\s*\d)')
_APPROX_RE = re.compile(r'approx|\\approx|≈|round', re.I)
_HEDGE_RE = re.compile(r'\bwait\b|\bassum|\bhowever\b|\bactually\b|\binterpret|\bmaybe\b|\bperhaps\b', re.I)
_IMPOSSIBLE_RE = re.compile(r'not possible|cannot|can\'t|impossible|doesn\'t make sense|not an integer|fraction of', re.I)

# Feature groups, used for ablations. Order defines the column order of the feature matrix.
GROUPS = {
    'answer': ['parsable', 'integer', 'nonneg', 'log_abs', 'in_question', 'is_question_max'],
    'vote': ['vote_count', 'vote_share', 'n_clusters', 'is_top_cluster', 'singleton'],
    'surface': ['log_len', 'len_z', 'n_lines', 'n_display_math', 'n_headers', 'n_bold'],
    'reasoning': ['n_checks', 'n_errors', 'err_rate', 'q_coverage', 'answer_derived',
                  'approx', 'hedge', 'impossible', 'n_boxed'],
    # comparison with the other candidates' working: does an answer show up as someone else's intermediate value?
    'cross': ['mine_in_others', 'mine_result_in_others', 'others_in_mine', 'others_result_in_mine'],
}
NAMES = [name for group in GROUPS.values() for name in group]


def _num(s):
    try:
        return Fraction(s.replace(',', ''))
    except (ValueError, ZeroDivisionError):
        return None


def _normalise(text):
    t = text.replace('\\$', '').replace('$', ' ').replace('\\,', '').replace('\\!', '')
    t = re.sub(r'\\text\{[^{}]*\}', ' ', t)
    t = re.sub(r'\\(?:times|cdot)|×|·', '*', t)
    t = re.sub(r'\\div|÷', '/', t)
    t = re.sub(r'(?<=\d)\s*x\s*(?=\d)', '*', t)
    t = re.sub(r'(?<=\d),(?=\d{3})', '', t)
    for _ in range(2):
        t = _FRAC_RE.sub(r'((\1)/(\2))', t)
    return t.replace('{', '(').replace('}', ')')


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}


def _evaluate(expr):
    """Evaluates +-*/ arithmetic over numeric literals only; None for anything else."""
    def walk(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            value = walk(node.operand)
            return -value if isinstance(node.op, ast.USub) else value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](walk(node.left), walk(node.right))
        raise ValueError
    try:
        value = walk(ast.parse(expr, mode='eval').body)
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError, RecursionError):
        return None
    return value if math.isfinite(value) else None


def arithmetic_check(text):
    """Recompute 'expression = value' statements; returns (checked, wrong)."""
    checked = wrong = 0
    for line in _normalise(text).splitlines():
        for expr, value in _EQ_RE.findall(line):
            expr = expr.strip()
            if not re.search(r'\d\s*[+\-*/]\s*[\d(]', expr) or '**' in expr or len(expr) > 80:
                continue
            got = _evaluate(re.sub(r'(?<=[\d)])\s*\(', '*(', expr))
            if got is None:
                continue
            checked += 1
            want = float(value)
            if abs(got - want) > max(1e-6, 0.006 * abs(want)) and abs(round(got) - want) > 1e-9:
                wrong += 1
    return checked, wrong


_RESULT_RE = re.compile(r'=\s*\(*\s*(-?\d+(?:\.\d+)?)')


def working_numbers(text):
    """(all numbers, numbers stated as '= value') in the working before the final boxed answer."""
    body = _normalise(text.rsplit('\\boxed', 1)[0])
    nums = {v for v in (_num(m) for m in _NUM_RE.findall(body)) if v is not None}
    results = {v for v in (_num(m) for m in _RESULT_RE.findall(body)) if v is not None}
    return nums, results


def cross_features(answers, working):
    """Per candidate: share of other-answer candidates whose working contains this answer, and vice versa."""
    values = [Fraction(a) if a is not None else None for a in answers]
    out = []
    for j, a in enumerate(values):
        others = [k for k, b in enumerate(values) if b is not None and b != a]
        if a is None or not others:
            out.append([0.0]*4)
            continue
        other_answers = {values[k] for k in others}
        out.append([np.mean([a in working[k][0] for k in others]),
                    np.mean([a in working[k][1] for k in others]),
                    np.mean([b in working[j][0] for b in other_answers]),
                    np.mean([b in working[j][1] for b in other_answers])])
    return out


def candidate_text_features(question, text, answer, qnums):
    lines = [line for line in text.splitlines() if line.strip()]
    checked, wrong = arithmetic_check(text)
    tnums = {_num(m) for m in _NUM_RE.findall(text)}
    cover = len(qnums & tnums)/len(qnums) if qnums else 1.0
    value = Fraction(answer) if answer is not None else None
    body = text.rsplit('\\boxed', 1)[0]
    derived = bool(value is not None and re.search(r'=\s*\$?\s*' + re.escape(str(float(value)).rstrip('0').rstrip('.')) + r'(?![\d.])',
                                                   _normalise(body)))
    return {
        'parsable': float(value is not None),
        'integer': float(value is not None and value.denominator == 1),
        'nonneg': float(value is not None and value >= 0),
        'log_abs': math.log1p(abs(float(value))) if value is not None else 0.0,
        'in_question': float(value is not None and value in qnums),
        'is_question_max': float(value is not None and bool(qnums) and value > max(qnums)),
        'log_len': math.log(len(text) + 1),
        'n_lines': math.log1p(len(lines)),
        'n_display_math': math.log1p(text.count('$$')/2),
        'n_headers': math.log1p(len(re.findall(r'^#+ ', text, re.M))),
        'n_bold': math.log1p(text.count('**')/2),
        'n_checks': math.log1p(checked),
        'n_errors': math.log1p(wrong),
        'err_rate': wrong/checked if checked else 0.0,
        'q_coverage': cover,
        'answer_derived': float(derived),
        'approx': math.log1p(len(_APPROX_RE.findall(text))),
        'hedge': math.log1p(len(_HEDGE_RE.findall(text))),
        'impossible': math.log1p(len(_IMPOSSIBLE_RE.findall(text))),
        'n_boxed': math.log1p(text.count('\\boxed')),
    }


def extract(bank):
    """Returns (features [N,4,F] float32, answers list[list[str|None]])."""
    feats = np.zeros((len(bank), 4, len(NAMES)), np.float32)
    answers = []
    for g, row in enumerate(bank):
        qnums = {v for v in (_num(m) for m in _NUM_RE.findall(row['question'])) if v is not None}
        texts = [c['text'] for c in row['candidates']]
        ans = [parse_answer(t) for t in texts]
        answers.append(ans)
        votes = Counter(a for a in ans if a is not None)
        top = max(votes.values()) if votes else 0
        lens = np.log([len(t) + 1 for t in texts])
        len_z = (lens - lens.mean())/(lens.std() + 1e-6)
        cross = cross_features(ans, [working_numbers(t) for t in texts])
        for i, (t, a) in enumerate(zip(texts, ans)):
            f = candidate_text_features(row['question'], t, a, qnums)
            count = votes[a] if a is not None else 0
            f.update(vote_count=count, vote_share=count/4, n_clusters=len(votes),
                     is_top_cluster=float(a is not None and count == top), singleton=float(count == 1),
                     len_z=len_z[i], **dict(zip(GROUPS['cross'], cross[i])))
            feats[g, i] = [f[name] for name in NAMES]
    return feats, answers


def columns(groups):
    return [NAMES.index(name) for group in groups for name in GROUPS[group]]
