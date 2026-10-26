"""CPU inference for the four experiments. Neural checkpoints need NumPy only."""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '4'
import json
from pathlib import Path
import numpy as np
from advanced_features import make_features, valid_mask, choose

def softmax(x):
    e = np.exp(x-x.max(axis=-1, keepdims=True))
    return e/e.sum(axis=-1, keepdims=True)

def score_candidates(bank, arrays, checkpoint, feature_cache=None):
    root = Path(checkpoint)
    config = json.loads((root/'config.json').read_text())
    mode = config['features']
    if feature_cache is None:
        x = make_features(bank, arrays, mode)
    else:
        if mode not in feature_cache:
            feature_cache[mode] = make_features(bank, arrays, mode)
        x = feature_cache[mode]
    kind = config['kind']
    if kind == 'tree':
        import lightgbm as lgb
        model = lgb.Booster(model_file=str(root/'model.txt'))
        scores = model.predict(x.reshape(-1, x.shape[-1]), num_threads=4).reshape(-1, 4)
    else:
        with np.load(root/'weights.npz', allow_pickle=False) as z:
            w = {k: z[k] for k in z.files}
        x = (x-w['mu'])/w['sd']
        if kind == 'linear':
            scores = (x @ w['weight'] + w['bias']).squeeze(-1)
        else:
            h = np.maximum(x @ w['fc.weight'].T + w['fc.bias'], 0)
            if kind == 'attention':
                q = h @ w['q.weight'].T
                k = h @ w['k.weight'].T
                v = h @ w['v.weight'].T
                att = softmax((q @ k.transpose(0, 2, 1))/np.sqrt(h.shape[-1]))
                h = np.concatenate([h, att @ v], axis=-1)
            scores = (h @ w['out.weight'].T + w['out.bias']).squeeze(-1)
    return scores

def predict(bank, arrays, checkpoint):
    return choose(score_candidates(bank, arrays, checkpoint), valid_mask(bank))
