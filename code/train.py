"""Reproduce the final checkpoint from supplied TRAIN only. CPU, four threads."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '4'
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from common import load_inputs, read_jsonl, parse_answer, check_data
from advanced_features import make_features

class Attention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim, 64)
        self.drop = nn.Dropout(.15)
        self.q = nn.Linear(64, 64, bias=False)
        self.k = nn.Linear(64, 64, bias=False)
        self.v = nn.Linear(64, 64, bias=False)
        self.out = nn.Linear(128, 1)
    def forward(self, x):
        h = self.drop(F.relu(self.fc(x)))
        a = torch.softmax(self.q(h) @ self.k(h).transpose(1, 2)/8, -1)
        return self.out(torch.cat([h, a @ self.v(h)], -1)).squeeze(-1)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, default=Path(__file__).resolve().parent.parent/'retrained_model')
    args = p.parse_args()
    if args.output.exists(): raise SystemExit('Choose a new output directory; existing assets are not overwritten.')
    start = time.perf_counter(); torch.set_num_threads(4)
    check_data(args.data_dir, ['train'])
    bank, arrays = load_inputs('train', args.data_dir)
    refs = read_jsonl(args.data_dir/'gsm8k_train.jsonl')
    if len(refs) != len(bank): raise ValueError('Reference length mismatch')
    labels = []
    for row, ref in zip(bank, refs):
        if row['id'] != ref['id'] or row['question'] != ref['question']: raise ValueError('Alignment mismatch')
        answer = parse_answer(ref['answer'])
        labels.append([answer is not None and parse_answer(c['text']) == answer for c in row['candidates']])
    y = np.asarray(labels, dtype=np.float32)
    if not y.any(1).all(): raise ValueError('Each group must have a correct candidate')
    x = make_features(bank, arrays, 'both')
    torch.manual_seed(7506); rng = np.random.default_rng(7506)
    flat = x.reshape(-1, x.shape[-1]); mu = flat.mean(0); sd = np.maximum(flat.std(0), 1e-5)
    xt = torch.from_numpy(np.ascontiguousarray((x-mu)/sd)); yt = torch.from_numpy(y)
    model = Attention(x.shape[-1])
    opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    history = []
    for epoch in range(1, 6):
        model.train(); total = 0.; order = rng.permutation(len(bank))
        for lo in range(0, len(order), 128):
            idx = order[lo:lo+128]; opt.zero_grad(); s = model(xt[idx])
            loss = (torch.logsumexp(s, 1)-torch.logsumexp(s.masked_fill(yt[idx] == 0, -torch.inf), 1)).mean()
            if not torch.isfinite(loss): raise ValueError('Nonfinite loss')
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.); opt.step()
            total += loss.item()*len(idx)
        history.append({'epoch': epoch, 'loss': total/len(bank)})
        print(history[-1], flush=True)
    args.output.mkdir(parents=True)
    config = {'kind':'attention','features':'both','loss':'listwise','hidden':64,'epochs':5,
              'seed':7506,'lr':.001,'weight_decay':.01,'dropout':.15,'batch_questions':128}
    (args.output/'config.json').write_text(json.dumps(config, indent=2)+'\n')
    np.savez(args.output/'weights.npz', mu=mu, sd=sd,
             **{k:v.detach().numpy() for k,v in model.state_dict().items()})
    # Training history is not an inference asset; keep it outside the checkpoint.
    args.output.with_suffix('.training.json').write_text(json.dumps({'seed':7506,'train_only':True,
        'seconds':time.perf_counter()-start,'history':history}, indent=2)+'\n')

if __name__ == '__main__': main()
