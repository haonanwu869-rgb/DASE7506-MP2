"""Input-only features; no references, IDs, external data or answer lookup."""
import re
import numpy as np
from common import parse_answer

FEATURE_NAMES = ['parseable', 'answer_agreement', 'distinct_answers', 'log_chars',
                 'log_words', 'log_lines', 'log_numbers', 'relative_chars',
                 'relative_words', 'question_overlap', 'simple_equations',
                 'equation_correct', 'equation_incorrect', 'cos_mean', 'cos_max', 'cos_min']
_NUM = r'[-+]?\d+(?:\.\d+)?'
_EQ = re.compile(rf'({_NUM})\s*([+*/-])\s*({_NUM})\s*=\s*({_NUM})')

def text_features(bank, emb):
    unit = emb / np.maximum(np.linalg.norm(emb, axis=-1, keepdims=True), 1e-8)
    cos = unit @ unit.transpose(0, 2, 1)
    out = np.zeros((len(bank), 4, len(FEATURE_NAMES)), dtype=np.float32)
    for i, row in enumerate(bank):
        texts = [c['text'] for c in row['candidates']]
        answers = [parse_answer(t) for t in texts]
        chars = np.array([len(t) for t in texts])
        words = np.array([len(t.split()) for t in texts])
        qwords = set(re.findall(r'\w+', row['question'].lower()))
        distinct = len(set(a for a in answers if a is not None))
        for j, t in enumerate(texts):
            tw = set(re.findall(r'\w+', t.lower()))
            good = bad = 0
            # Only whole-line elementary equalities, not fragments of larger formulas.
            for line in t.splitlines():
                match = _EQ.fullmatch(line.strip().strip('$').strip())
                if match:
                    a, op, b, c = match.groups(); a, b, c = float(a), float(b), float(c)
                    if op == '/' and b == 0:
                        bad += 1; continue
                    v = {'+': lambda: a+b, '-': lambda: a-b,
                         '*': lambda: a*b, '/': lambda: a/b}[op]()
                    correct = abs(v-c) <= 1e-6 * max(1, abs(c))
                    good += int(correct); bad += int(not correct)
            others = np.delete(cos[i, j], j)
            out[i, j] = [answers[j] is not None,
                answers.count(answers[j])/4 if answers[j] is not None else 0,
                distinct/4, np.log1p(chars[j]), np.log1p(words[j]),
                np.log1p(len(t.splitlines())), np.log1p(len(re.findall(_NUM, t))),
                chars[j]/max(1, chars.mean()), words[j]/max(1, words.mean()),
                len(tw & qwords)/max(1, len(qwords)), np.log1p(good+bad),
                np.log1p(good), np.log1p(bad), others.mean(), others.max(), others.min()]
    return out

def make_features(bank, arrays, mode):
    emb = np.asarray(arrays['embeddings'], dtype=np.float32)
    if mode == 'embedding':
        return emb
    extra = text_features(bank, emb)
    if mode in ('format', 'arithmetic', 'both'):
        from reliability_features import reliability_features
        return np.concatenate([emb, extra, reliability_features(bank, mode)], axis=-1)
    if mode == 'relative':
        return np.concatenate([emb, emb-emb.mean(axis=1, keepdims=True), extra], axis=-1)
    return np.concatenate([emb, extra], axis=-1)

def valid_mask(bank):
    return np.array([[parse_answer(c['text']) is not None for c in r['candidates']]
                     for r in bank], dtype=bool)

def choose(scores, valid):
    # All-invalid unseen groups fall back to model scores; supplied groups have a correct answer.
    mask = valid | ~valid.any(axis=1, keepdims=True)
    return np.where(mask, scores, -np.inf).argmax(axis=1).tolist()
