"""Four predeclared equal-weight ensembles plus two single-model controls."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '4'
import hashlib
import json
import shutil
import time
import numpy as np
from run_four_steps import Experiments, BASE, write_json

ROOT = BASE/'runs'/'ensembles'
SEEDS = [7506, 17, 42, 2026, 3407]

def main():
    exp = Experiments(ROOT, 50)
    old = [BASE/'runs'/'stability'/f'attention_seed_{s}' for s in SEEDS]
    new = [BASE/'runs'/'reliability'/f'both_seed_{s}' for s in SEEDS]
    plans = [('old_single', [old[0]]), ('new_single', [new[0]]),
             ('old_new_pair', [old[0], new[0]]), ('old_five', old),
             ('new_five', new), ('old_new_ten', old+new)]
    write_json(ROOT/'protocol.json', {'aggregation': 'equal_probability_mean',
        'temperature': 1, 'seeds': SEEDS, 'plans': {n: [str(p) for p in ps] for n, ps in plans},
        'weight_search': False, 'retraining': False, 'test_used': False,
        'selection_rule': 'Highest official validation accuracy; ties prefer fewer asset bytes'})
    for name, members in plans:
        folder = ROOT/name; folder.mkdir(exist_ok=True)
        member_names = []
        for i, source in enumerate(members):
            destination = folder/f'member_{i:02d}'
            destination.mkdir(exist_ok=True)
            for filename in ['config.json', 'weights.npz']:
                shutil.copyfile(source/filename, destination/filename)
            member_names.append(destination.name)
        write_json(folder/'ensemble.json', {'aggregation': 'equal_probability_mean', 'members': member_names})
        exp.evaluate(name, folder, 0, {'stage': 'equal_weight_ensemble', 'members': len(members)},
                     method='ensemble_submission.py')
    best = min(exp.results, key=lambda r: (-r['correct'], r['checkpoint_bytes']))
    write_json(ROOT/'best_model.json', {'name': best['name'], 'checkpoint': best['checkpoint'],
        'method': 'ensemble_submission.py', 'selection_basis': 'validation accuracy; ties prefer smaller checkpoint'})
    baseline = json.loads((ROOT/'new_single-predictions.json').read_text())
    a = exp.vy[np.arange(len(exp.vy)), baseline['selected']].astype(bool)
    changes = {}
    for r in exp.results:
        p = json.loads((ROOT/f"{r['name']}-predictions.json").read_text())
        assert p['ids'] == baseline['ids']
        b = exp.vy[np.arange(len(exp.vy)), p['selected']].astype(bool)
        changes[r['name']] = {'fixes_new_single_errors': int((b & ~a).sum()),
                              'breaks_new_single_correct': int((a & ~b).sum())}
    write_json(ROOT/'complementarity.json', changes)
    # Verify both single-member controls preserve the previously frozen choices.
    for name, path in [('old_single', BASE/'runs'/'stability'/'attention_seed_7506-predictions.json'),
                       ('new_single', BASE/'runs'/'reliability'/'both_seed_7506-predictions.json')]:
        fresh = json.loads((ROOT/f'{name}-predictions.json').read_text())
        prior = json.loads(path.read_text())
        if fresh['ids'] != prior['ids'] or fresh['selected'] != prior['selected']:
            raise ValueError('Single-model control changed')
    lines = ['# 固定等权融合实验', '',
        '只使用已训练的模型。本轮不训练、不调权重、不搜索温度，不读取测试集。',
        '每个模型先过滤官方无法解析的候选，再对四个候选的分数做 softmax（温度固定 1），等权平均概率后取最大项。',
        '因此平均的是概率，而不是最终数值答案，也不是未经归一化的原始分数。',
        '旧单模型与新单模型均固定种子 7506；五种子为 7506、17、42、2026、3407。所有方案均在评分前写入 protocol.json。', '',
        '|方案|模型数|正确数/798|准确率|不同答案子集|总资产 MiB|本机完整预测+评分秒|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in exp.results:
        lines.append(f"|{r['name']}|{r['members']}|{r['correct']}|{r['accuracy']:.2f}%|{r['hard_accuracy']:.2f}%|"
                     f"{r['checkpoint_bytes']/1024**2:.3f}|{r['external_prediction_and_score_seconds']:.2f}|")
    lines += ['', f"本轮验证集优胜方案：**{best['name']}**。同分优先较小资产。旧实验目录与默认指针保留。", '',
        '## 相对新单模型的变化', '', '|方案|修复错题|新增错题|', '|---|---:|---:|']
    for name, r in changes.items():
        lines.append(f"|{name}|{r['fixes_new_single_errors']}|{r['breaks_new_single_correct']}|")
    lines += ['', '每个方案文件夹包含全部成员权重和配置，计入资产总量，不依赖其他实验目录。',
        '模型顺序执行，线程限制为 4；特征仅在一次预测调用内按特征类型复用，接受不同输入，不缓存样本答案。',
        '耗时包含独立预测进程和评分进程启动、导入、加载与处理；此处为本机 validation 测量，不代表指定服务器 test 性能。',
        '单成员融合与原保存预测逐项一致。各模型体积、SHA256 与官方评分保存在 results.json。',
        '验证集已用于多轮开发，这次最优仍可能有选择偏差，不保证测试集成绩，也不据此继续微调权重。', '',
        '## 复现', '', '在 code 目录运行：', '', '```powershell',
        r'.\.venv\Scripts\python.exe run_ensembles.py',
        r'.\.venv\Scripts\python.exe evaluate_best.py --experiments runs/ensembles', '```', '',
        '需要的推理源码：ensemble_submission.py、advanced_submission.py、advanced_features.py、reliability_features.py 及课程 common.py。推理只需要 NumPy。',
        '本轮代码、执行和分析由 Codex 协助，需按课程要求披露。']
    (ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    write_json(ROOT/'source-sha256.json', {n: hashlib.sha256((BASE/n).read_bytes()).hexdigest()
        for n in ['run_ensembles.py','ensemble_submission.py','advanced_submission.py','advanced_features.py','reliability_features.py']})
    print('SELECTED', best['name'], best['accuracy'], flush=True)

if __name__ == '__main__': main()
