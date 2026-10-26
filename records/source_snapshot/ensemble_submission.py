"""Equal-weight average of candidate probabilities. Inputs only; CPU sequential."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '4'
import json
from pathlib import Path
import numpy as np
from advanced_features import choose, valid_mask
from advanced_submission import score_candidates, softmax

def predict(bank, arrays, checkpoint):
    root = Path(checkpoint)
    config = json.loads((root/'ensemble.json').read_text(encoding='utf-8'))
    if config['aggregation'] != 'equal_probability_mean' or not config['members']:
        raise ValueError('Unsupported ensemble configuration')
    valid = valid_mask(bank)
    mask = valid | ~valid.any(axis=1, keepdims=True)
    total = np.zeros((len(bank), 4), dtype=np.float64)
    cache = {}  # This call only; never reuse features across different banks.
    for member in config['members']:
        scores = score_candidates(bank, arrays, root/member, cache)
        total += softmax(np.where(mask, scores, -np.inf))
    return choose(total/len(config['members']), valid)
