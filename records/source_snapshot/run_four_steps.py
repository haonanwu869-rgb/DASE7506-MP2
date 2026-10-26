"""Reproducible train-only tuning and official validation, never opens test data.

Run from code/: .venv/Scripts/python.exe run_four_steps.py
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '4'
import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
from common import DATA, check_data, load_inputs, read_jsonl, parse_answer
from advanced_features import make_features, valid_mask, choose, FEATURE_NAMES

BASE = Path(__file__).resolve().parent
SEED = 7506

def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')

def labels(bank, split):
    refs = read_jsonl(DATA/f'gsm8k_{split}.jsonl')
    if len(refs) != len(bank):
        raise ValueError('Reference count mismatch')
    y = []
    for row, ref in zip(bank, refs):
        if row['id'] != ref['id'] or row['question'] != ref['question']:
            raise ValueError('Reference alignment mismatch')
        answer = parse_answer(ref['answer'])
        y.append([answer is not None and parse_answer(c['text']) == answer for c in row['candidates']])
    y = np.array(y, dtype=np.float32)
    if not y.any(axis=1).all():
        raise ValueError('Every supplied group must contain a correct candidate')
    return y

def accuracy(y, scores, valid):
    chosen = choose(scores, valid)
    return float(y[np.arange(len(y)), chosen].mean())

def scaler(x):
    flat = x.reshape(-1, x.shape[-1])
    return flat.mean(0), np.maximum(flat.std(0), 1e-5)

class Experiments:
    def __init__(self, root, max_epochs):
        self.root = root; root.mkdir(parents=True, exist_ok=True)
        self.max_epochs = max_epochs
        check_data(DATA, ['train', 'validation'])
        self.bank, self.arrays = load_inputs('train')
        self.vbank, self.varrays = load_inputs('validation')
        self.y = labels(self.bank, 'train')
        self.vy = labels(self.vbank, 'validation')
        self.valid = valid_mask(self.bank)
        self.hard = np.array([len(set(a for a in (parse_answer(c['text']) for c in r['candidates'])
                                      if a is not None)) >= 2 for r in self.vbank])
        order = np.random.default_rng(SEED).permutation(len(self.bank))
        n = round(len(order)*.8)
        self.fit, self.hold = order[:n], order[n:]
        self.cache = {}
        self.results = []
        if (root/'results.json').exists():
            self.results = json.loads((root/'results.json').read_text(encoding='utf-8'))
        write_json(root/'split.json', {'seed': SEED, 'fit_question_indices': self.fit.tolist(),
                                      'holdout_question_indices': self.hold.tolist()})
        cpu_name = platform.processor()
        if sys.platform == 'win32':
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
                cpu_name = winreg.QueryValueEx(key, 'ProcessorNameString')[0].strip()
        write_json(root/'environment.json', {'cpu_name': cpu_name, 'python': sys.version, 'platform': platform.platform(),
            'processor': platform.processor(), 'logical_cpu_count': os.cpu_count(), 'threads': 4,
            'packages': {p: importlib.metadata.version(p) for p in
                         ['numpy', 'scipy', 'scikit-learn', 'torch', 'lightgbm']}})
        print(f'Train={len(self.bank)}, internal={n}/{len(self.hold)}, validation={len(self.vbank)}, '
              f'validation distinct-answer subset={self.hard.sum()}', flush=True)

    def features(self, mode):
        if mode not in self.cache:
            self.cache[mode] = make_features(self.bank, self.arrays, mode)
        return self.cache[mode]

    def evaluate(self, name, folder, seconds, extra, method='advanced_submission.py'):
        pred = self.root/f'{name}-predictions.json'
        score = self.root/f'{name}-score.json'
        start = time.perf_counter()
        commands = [[sys.executable, str(BASE/'predict.py'), '--method', str(BASE/method),
                     '--checkpoint', str(folder), '--split', 'validation', '--output', str(pred)],
                    [sys.executable, str(BASE/'score_predictions.py'), '--split', 'validation',
                     '--predictions', str(pred), '--output', str(score)]]
        for cmd in commands:
            p = subprocess.run(cmd, cwd=BASE, capture_output=True, text=True)
            if p.returncode:
                raise RuntimeError(p.stdout+'\n'+p.stderr)
        elapsed = time.perf_counter()-start
        scored = json.loads(score.read_text())
        choices = json.loads(pred.read_text())['selected']
        correct = self.vy[np.arange(len(choices)), choices]
        files = sorted(p for p in folder.rglob('*') if p.is_file())
        assets = {str(p.relative_to(folder)): {'bytes': p.stat().st_size,
                   'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in files}
        size = sum(v['bytes'] for v in assets.values())
        if size > 16*1024**2:
            raise ValueError(f'{name} exceeds checkpoint budget: {size}')
        record = dict(name=name, **scored, train_tune_seconds=seconds,
            external_prediction_and_score_seconds=elapsed, checkpoint_bytes=size,
            hard_n=int(self.hard.sum()), hard_correct=int(correct[self.hard].sum()),
            hard_accuracy=100*float(correct[self.hard].mean()), checkpoint=str(folder),
            assets=assets, **extra)
        self.results = [r for r in self.results if r['name'] != name] + [record]
        write_json(self.root/'results.json', self.results)
        print(f"RESULT {name}: {scored['correct']}/{scored['n']} = {scored['accuracy']:.2f}%; "
              f"hard={record['hard_accuracy']:.2f}%; {size/1024**2:.3f} MiB; wall={elapsed:.2f}s", flush=True)
        return record

    def baseline(self):
        self.evaluate('baseline', BASE/'runs'/'model_bundle', 0, {'stage': 0}, 'submission.py')
        # Separate the deterministic parser filter from learned-model gains.
        import pickle
        with (BASE/'runs'/'model_bundle'/'model.pkl').open('rb') as f:
            old = pickle.load(f)
        folder = self.root/'baseline_parseable'; folder.mkdir(exist_ok=True)
        write_json(folder/'config.json', {'kind': 'linear', 'features': 'embedding'})
        np.savez(folder/'weights.npz', mu=old['mu'], sd=old['sd'],
                 weight=old['w'][:-1, None], bias=old['w'][-1:])
        self.evaluate('baseline_parseable', folder, 0, {'stage': 0})

    def logreg(self):
        from sklearn.linear_model import LogisticRegression
        start = time.perf_counter(); x = self.features('embedding')
        mu, sd = scaler(x[self.fit]); xn = (x-mu)/sd
        trials = []; best = None
        for c in [.001, .01, .1, 1., 10.]:
            model = LogisticRegression(C=c, solver='lbfgs', max_iter=2000, tol=1e-5)
            model.fit(xn[self.fit].reshape(-1, x.shape[-1]), self.y[self.fit].ravel())
            s = model.decision_function(xn[self.hold].reshape(-1, x.shape[-1])).reshape(-1, 4)
            a = accuracy(self.y[self.hold], s, self.valid[self.hold])
            trial = {'C': c, 'accuracy': a, 'iterations': int(model.n_iter_[0])}; trials.append(trial)
            print('LogReg internal', trial, flush=True)
            if best is None or a > best['accuracy']: best = trial
        mu, sd = scaler(x); xn = (x-mu)/sd
        model = LogisticRegression(C=best['C'], solver='lbfgs', max_iter=2000, tol=1e-5)
        model.fit(xn.reshape(-1, x.shape[-1]), self.y.ravel())
        folder = self.root/'step1_logreg'; folder.mkdir(exist_ok=True)
        write_json(folder/'config.json', {'kind': 'linear', 'features': 'embedding', 'C': best['C']})
        np.savez(folder/'weights.npz', mu=mu, sd=sd, weight=model.coef_.T, bias=model.intercept_)
        write_json(self.root/'step1_search.json', trials)
        self.evaluate('step1_logreg', folder, time.perf_counter()-start,
                      {'stage': 1, 'internal_best': best, 'full_fit_iterations': int(model.n_iter_[0])})

    def neural(self, name, mode, kind, loss_name, hidden, stage, seed=SEED, fixed_epochs=None):
        import torch
        from torch import nn
        from torch.nn import functional as F
        torch.set_num_threads(4)
        start = time.perf_counter(); x = self.features(mode)

        class Network(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(x.shape[-1], hidden)
                self.drop = nn.Dropout(.15)
                if kind == 'attention':
                    self.q = nn.Linear(hidden, hidden, bias=False)
                    self.k = nn.Linear(hidden, hidden, bias=False)
                    self.v = nn.Linear(hidden, hidden, bias=False)
                self.out = nn.Linear(hidden*(2 if kind == 'attention' else 1), 1)
            def forward(self, z):
                h = self.drop(F.relu(self.fc(z)))
                if kind == 'attention':
                    att = torch.softmax(self.q(h) @ self.k(h).transpose(1, 2)/hidden**.5, -1)
                    h = torch.cat([h, att @ self.v(h)], dim=-1)
                return self.out(h).squeeze(-1)

        def loss_fn(s, y):
            if loss_name == 'bce': return F.binary_cross_entropy_with_logits(s, y)
            if loss_name == 'pairwise':
                pairs = (y[:, :, None] > y[:, None, :])
                diff = s[:, :, None]-s[:, None, :]
                costs = F.softplus(-diff)*pairs
                counts = pairs.sum(dim=(1, 2))
                informative = counts > 0
                return (costs.sum(dim=(1, 2))/counts.clamp_min(1))[informative].mean() if informative.any() else s.sum()*0
            # Multiple correct candidates: maximize their TOTAL softmax probability.
            return (torch.logsumexp(s, 1)-torch.logsumexp(s.masked_fill(y == 0, -torch.inf), 1)).mean()

        def train(indices, epochs, holdout=None):
            torch.manual_seed(seed); rng = np.random.default_rng(seed)
            mu, sd = scaler(x[indices])
            xt = torch.from_numpy(np.ascontiguousarray((x[indices]-mu)/sd))
            yt = torch.from_numpy(self.y[indices])
            model = Network(); opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
            if holdout is not None:
                xv = torch.from_numpy(np.ascontiguousarray((x[holdout]-mu)/sd))
            history = []; best_acc = -1; best_epoch = 1
            for epoch in range(1, epochs+1):
                model.train(); total = 0
                order = rng.permutation(len(indices))
                for lo in range(0, len(order), 128):
                    idx = order[lo:lo+128]
                    opt.zero_grad(); loss = loss_fn(model(xt[idx]), yt[idx])
                    if not torch.isfinite(loss): raise ValueError('Nonfinite training loss')
                    loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.); opt.step()
                    total += loss.item()*len(idx)
                rec = {'epoch': epoch, 'loss': total/len(indices)}
                if holdout is not None:
                    model.eval()
                    with torch.no_grad(): s = model(xv).numpy()
                    a = accuracy(self.y[holdout], s, self.valid[holdout]); rec['accuracy'] = a
                    if a > best_acc: best_acc = a; best_epoch = epoch
                history.append(rec)
                if epoch == 1 or epoch % 5 == 0:
                    print(name, 'tune' if holdout is not None else 'refit', rec, flush=True)
                if holdout is not None and epoch-best_epoch >= 10: break
            return model, mu, sd, history, best_epoch, best_acc

        if fixed_epochs is None:
            _, _, _, history, epochs, best = train(self.fit, self.max_epochs, self.hold)
        else:
            history, epochs, best = [], fixed_epochs, None
        model, mu, sd, full_history, _, _ = train(np.arange(len(x)), epochs)
        folder = self.root/name; folder.mkdir(exist_ok=True)
        config = {'kind': kind, 'features': mode, 'loss': loss_name, 'hidden': hidden,
                  'epochs': epochs, 'seed': seed, 'lr': .001, 'weight_decay': .01,
                  'dropout': .15, 'batch_questions': 128}
        write_json(folder/'config.json', config)
        weights = {k: v.detach().numpy() for k, v in model.state_dict().items()}
        np.savez(folder/'weights.npz', mu=mu, sd=sd, **weights)
        write_json(self.root/f'{name}-history.json', {'internal': history, 'full_train': full_history})
        # Compare exported NumPy decisions with training implementation on validation inputs.
        from advanced_submission import predict
        vx = make_features(self.vbank, self.varrays, mode)
        model.eval()
        with torch.no_grad(): scores = model(torch.from_numpy((vx-mu)/sd)).numpy()
        expected = choose(scores, valid_mask(self.vbank))
        if expected != predict(self.vbank, self.varrays, folder):
            raise ValueError('Exported neural inference differs from PyTorch')
        self.evaluate(name, folder, time.perf_counter()-start,
                      {'stage': stage, 'internal_accuracy': best, 'epochs': epochs, 'seed': seed,
                       'fixed_epochs': fixed_epochs, 'export_parity': True})

    def trees(self):
        import lightgbm as lgb
        x = self.features('text'); d = x.shape[-1]
        for objective in ['binary', 'lambdarank']:
            start = time.perf_counter(); trials = []; best = None
            for leaves in [7, 15]:
                params = dict(objective=objective, learning_rate=.035, num_leaves=leaves,
                    min_data_in_leaf=50, lambda_l2=10., feature_fraction=.8,
                    verbosity=-1, num_threads=4, seed=SEED, deterministic=True,
                    force_col_wise=True, metric='None')
                is_rank = objective == 'lambdarank'
                if is_rank: params['lambdarank_truncation_level'] = 4
                tr = lgb.Dataset(x[self.fit].reshape(-1, d), label=self.y[self.fit].ravel(),
                    group=[4]*len(self.fit) if is_rank else None)
                va = lgb.Dataset(x[self.hold].reshape(-1, d), label=self.y[self.hold].ravel(),
                    group=[4]*len(self.hold) if is_rank else None, reference=tr)
                def metric(pred, dataset):
                    return 'selection', accuracy(self.y[self.hold], pred.reshape(-1, 4), self.valid[self.hold]), True
                model = lgb.train(params, tr, num_boost_round=500, valid_sets=[va], feval=metric,
                    callbacks=[lgb.early_stopping(50, verbose=False)])
                rec = {'leaves': leaves, 'rounds': model.best_iteration,
                       'accuracy': model.best_score['valid_0']['selection']}
                trials.append(rec); print('Tree internal', objective, rec, flush=True)
                if best is None or rec['accuracy'] > best['accuracy']: best = rec; best_params = params
            full = lgb.Dataset(x.reshape(-1, d), label=self.y.ravel(),
                               group=[4]*len(x) if is_rank else None)
            model = lgb.train(best_params, full, num_boost_round=best['rounds'])
            name = 'step4_tree_'+objective; folder = self.root/name; folder.mkdir(exist_ok=True)
            model.save_model(str(folder/'model.txt'))
            write_json(folder/'config.json', {'kind': 'tree', 'features': 'text', 'params': best_params,
                                             'rounds': best['rounds']})
            write_json(self.root/f'{name}-search.json', trials)
            self.evaluate(name, folder, time.perf_counter()-start, {'stage': 4, 'internal_best': best})

    def summary(self):
        best = max(self.results, key=lambda r: r['accuracy'])
        source_files = ['advanced_features.py', 'advanced_submission.py', 'run_four_steps.py',
                        'evaluate_best.py', 'common.py', 'predict.py', 'score_predictions.py',
                        'submission.py', 'train_logreg.py']
        write_json(self.root/'source-sha256.json', {name: hashlib.sha256((BASE/name).read_bytes()).hexdigest()
                                                 for name in source_files})
        write_json(self.root/'best_model.json', {'name': best['name'], 'checkpoint': best['checkpoint'],
            'method': 'submission.py' if best['name'] == 'baseline' else 'advanced_submission.py',
            'selection_basis': 'official validation accuracy; no test set accessed'})
        lines = ['# 前四步实验结果', '',
            '仅训练集用于拟合。按题目固定划分 80%/20% 内部训练/验证，选择超参数或轮数后用完整训练集重训。',
            '下表来自官方 validation 评分器；测试集未使用。每个文件夹是独立模型，提交时只选一个，不能把全部实验资产一起打包。', '',
            '|方法|正确数/798|准确率|不同答案子集准确率|模型 MiB|本机预测+评分秒|',
            '|---|---:|---:|---:|---:|---:|']
        for r in self.results:
            lines.append(f"|{r['name']}|{r['correct']}|{r['accuracy']:.2f}%|{r['hard_accuracy']:.2f}%|"
                         f"{r['checkpoint_bytes']/1024**2:.3f}|{r['external_prediction_and_score_seconds']:.2f}|")
        lines += ['', f"当前验证集最佳：**{best['name']}**。这不保证测试集也最佳。",
            '', '## 方法与复现', '',
            '- step1：L2 逻辑回归，内部验证比较 C=0.001/0.01/0.1/1/10，LBFGS 最多 2000 轮。',
            '- step2：相同 128 隐层 MLP 比较 BCE、按题平均 RankNet 配对损失、多正确答案集合 softmax 损失。',
            '- step3：相对 embedding 与 16 个输入特征的 MLP；64 隐层单头候选自注意力模型。均使用集合 softmax。',
            '- step4：LightGBM 二分类和 LambdaRank，各比较 7/15 个叶节点，内部早停后全训练集重训。',
            '- 新方法统一排除官方解析器无法解析的候选；baseline_parseable 单独测量该过滤带来的变化。',
            '- 不同答案子集：至少有两个不同的可解析数值答案，不是两个不同正确答案。',
            '- 简单等式检查只识别整行 a 运算符 b = c，不证明整个推理正确；没有调用代码执行或外部模型。',
            '- 神经网络从零训练，导出 NumPy 权重；export_parity 表示与 PyTorch 验证集选择结果一致。',
            '- 所有配置、搜索/损失历史、官方预测与评分、资产 SHA256、环境信息均保存在本目录。',
            '- 本轮单个随机种子 7506；尚未检查跨种子稳定性或测试集表现，不能保证最优泛化。',
            '- 本机时间包含两个独立进程启动/导入/数据读取/预测/评分，限制线程为 4；不代表指定评分服务器耗时。',
            '- 时间与资产大小是本机观测，不替代教师正式资源审核；源码不能嵌入预测表。',
            '', '在 code 目录运行：', '', '```powershell',
            r'.\.venv\Scripts\python.exe run_four_steps.py', '```', '',
            '可用 --stage 1/2/3/4 单独重跑；默认结果目录 runs/four_steps。',
            '复现所选模型的验证评分（预先设置线程数）：`.venv/Scripts/python.exe evaluate_best.py`。',
            '全部开发结束并冻结方法后才运行 `evaluate_best.py --split test`；本轮没有运行该命令。',
            '依赖锁定：本目录 requirements-lock.txt；推理依赖见各模型类型（NumPy 或 NumPy+LightGBM）。',
            '', '## AI 使用披露', '',
            '本轮实验脚本、特征、训练执行和结果说明由 Codex 协助生成与运行。学生需阅读代码、理解损失和选择策略，按课程要求在报告披露。',
            '没有复制外部项目代码、使用外部训练样本或预训练参数。使用 scikit-learn、PyTorch、LightGBM 软件库；',
            '可参考 allRank（https://github.com/allegro/allRank）的排序方法体系，以及 LightGBM 官方项目。']
        (self.root/'RESULTS.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
        freeze = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], capture_output=True, text=True, check=True)
        (self.root/'requirements-lock.txt').write_text(freeze.stdout, encoding='utf-8')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['all', '1', '2', '3', '4'], default='all')
    parser.add_argument('--output', type=Path, default=BASE/'runs'/'four_steps')
    parser.add_argument('--max-epochs', type=int, default=50)
    args = parser.parse_args()
    exp = Experiments(args.output.resolve(), args.max_epochs)
    if args.stage == 'all': exp.baseline()
    if args.stage in ['all', '1']: exp.logreg(); exp.summary()
    if args.stage in ['all', '2']:
        for loss in ['bce', 'pairwise', 'listwise']:
            exp.neural('step2_mlp_'+loss, 'embedding', 'mlp', loss, 128, 2); exp.summary()
    if args.stage in ['all', '3']:
        exp.neural('step3_features_mlp', 'relative', 'mlp', 'listwise', 96, 3); exp.summary()
        exp.neural('step3_attention', 'text', 'attention', 'listwise', 64, 3); exp.summary()
    if args.stage in ['all', '4']: exp.trees(); exp.summary()

if __name__ == '__main__':
    main()
