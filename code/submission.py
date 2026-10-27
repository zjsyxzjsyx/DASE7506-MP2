"""Submission interface: vote-aware listwise ranker over answer clusters (see train.py)."""
from pathlib import Path

import features as F
import model as M


def predict(bank, arrays, checkpoint):
    """Return one Python integer in 0..3 per question, in bank order.

    bank: list of question dictionaries with four candidate texts each.
    arrays['embeddings']: float32 NumPy array of shape [N, 4, 1024].
    checkpoint: pathlib.Path or None, supplied through --checkpoint.
    """
    bundle = M.load(checkpoint or Path(__file__).resolve().parent/'runs'/'model_bundle')
    feats, answers = F.extract(bank)
    texts = [[c['text'] for c in row['candidates']] for row in bank]
    scores = M.score(bundle, feats, arrays['embeddings'], answers, texts)
    return M.select(scores, answers)
