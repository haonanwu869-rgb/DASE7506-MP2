"""Fixed-budget reliability-feature ablation. Training data only for fitting."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '4'
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from run_four_steps import Experiments, BASE, write_json
from reliability_features import FORMAT_NAMES, ARITH_NAMES, reliability_features, arithmetic_checks

ROOT = BASE/'runs'/'reliability'
SEEDS = [7506, 17, 42, 2026, 3407]

def report(exp):
    original = BASE/'runs'/'four_steps'
    baseline = np.array(json.loads((original/'step3_attention-predictions.json').read_text())['selected'])
    base_ok = exp.vy[np.arange(len(exp.vy)), baseline].astype(bool)
    diagnostics = {}
    for split, bank, y in [('train', exp.bank, exp.y), ('validation', exp.vbank, exp.vy)]:
        f = reliability_features(bank, 'both')
        diagnostics[split] = {'candidate_count': len(bank)*4,
            'hash_box_disagree': int((f[:, :, 4] > 0).sum()),
            'heading_like_hash': int((f[:, :, 2] > 0).sum()),
            'arithmetic_checked_candidates': int((f[:, :, 8] > 0).sum()),
            'arithmetic_bad_candidates': int((f[:, :, 12] > 0).sum()),
            'bad_among_final_correct': int(((f[:, :, 12] > 0) & (y > 0)).sum()),
            'bad_among_final_incorrect': int(((f[:, :, 12] > 0) & (y == 0)).sum())}
        if split == 'validation':
            selected_f = f[np.arange(len(bank)), baseline]
            diagnostics[split]['original_61_errors_with_bad_arithmetic'] = int(((selected_f[:, 12] > 0) & ~base_ok).sum())
            diagnostics[split]['original_61_errors_with_any_arithmetic_checked'] = int(((selected_f[:, 8] > 0) & ~base_ok).sum())
            diagnostics[split]['original_61_errors_hash_box_disagree'] = int(((selected_f[:, 4] > 0) & ~base_ok).sum())
            evidence = []
            for i in np.flatnonzero(~base_ok):
                evidence.append({'validation_index': int(i), 'selected_index': int(baseline[i]),
                    'features': dict(zip(FORMAT_NAMES+ARITH_NAMES, map(float, selected_f[i]))),
                    'arithmetic_checks': arithmetic_checks(bank[i]['candidates'][baseline[i]]['text'])})
            write_json(ROOT/'original_error_feature_evidence.json', evidence)
    write_json(ROOT/'feature_diagnostics.json', diagnostics)
    old_correct, new_correct = [], []
    comparisons = []
    for seed in SEEDS:
        old = json.loads((BASE/'runs'/'stability'/f'attention_seed_{seed}-predictions.json').read_text())
        new = json.loads((ROOT/f'both_seed_{seed}-predictions.json').read_text())
        assert old['ids'] == new['ids'] == [r['id'] for r in exp.vbank]
        a = exp.vy[np.arange(len(exp.vy)), old['selected']].astype(bool)
        b = exp.vy[np.arange(len(exp.vy)), new['selected']].astype(bool)
        old_correct.append(a); new_correct.append(b)
        comparisons.append({'seed': seed, 'old_correct': int(a.sum()), 'new_correct': int(b.sum()),
            'delta_correct': int(b.sum())-int(a.sum()), 'fixes': int((b & ~a).sum()), 'breaks': int((a & ~b).sum())})
    old_correct, new_correct = np.array(old_correct), np.array(new_correct)
    scores = 100*new_correct.mean(1); old_scores = 100*old_correct.mean(1)
    summary = {'comparisons': comparisons, 'old_mean': float(old_scores.mean()), 'new_mean': float(scores.mean()),
               'new_sample_std_pp': float(scores.std(ddof=1)), 'mean_delta_pp': float((scores-old_scores).mean()),
               'hard_new_mean': float(100*new_correct[:, exp.hard].mean()),
               'hard_old_mean': float(100*old_correct[:, exp.hard].mean())}
    write_json(ROOT/'paired_summary.json', summary)
    write_json(ROOT/'best_model.json', {'name': 'both_seed_7506',
        'checkpoint': str(ROOT/'both_seed_7506'), 'method': 'advanced_submission.py',
        'selection_basis': 'Primary prespecified seed 7506; original default pointer unchanged; validation only'})
    lines = ['# 解析格式与算术特征实验', '',
        '新增 8 个格式风险特征与 8 个算术特征；输入只来自候选文本，不使用参考答案生成特征。',
        '官方解析器、评分器、数据和旧模型均保持原样。本轮未读取测试集，未更换默认最佳模型指针。',
        '与原注意力模型使用相同 64 隐层、5 轮、学习率、dropout 和完整训练集；改变输入维数意味着初始化不能逐参数完全配对。',
        '预设实验：种子 7506 比较仅格式、仅算术、两者结合；两者结合另用 17/42/2026/3407 验证稳定性，不根据验证结果追加调参。', '',
        '|配置|种子|正确数/798|准确率|不同答案子集|完整预测+评分秒|', '|---|---:|---:|---:|---:|---:|',
        '|旧注意力|7506|737|92.36%|82.82%|见旧记录|']
    for r in exp.results:
        lines.append(f"|{r['name']}|{r['seed']}|{r['correct']}|{r['accuracy']:.2f}%|{r['hard_accuracy']:.2f}%|{r['external_prediction_and_score_seconds']:.2f}|")
    lines += ['', '## 五个种子的比较', '', '|种子|旧模型正确数|新组合正确数|净变化|修复/新增错误|', '|---|---:|---:|---:|---:|']
    for r in comparisons:
        lines.append(f"|{r['seed']}|{r['old_correct']}|{r['new_correct']}|{r['delta_correct']:+d}|{r['fixes']}/{r['breaks']}|")
    lines += ['', f"旧均值 {summary['old_mean']:.2f}%；新均值 {summary['new_mean']:.2f}%，"
        f"标准差 {summary['new_sample_std_pp']:.2f} 个百分点；平均变化 {summary['mean_delta_pp']:+.2f} 个百分点。",
        f"不同答案子集：旧均值 {summary['hard_old_mean']:.2f}%，新均值 {summary['hard_new_mean']:.2f}%。",
        '五种子共享同一验证集，不能当作五个独立数据集；不是泛化置信区间。', '', '## 检查覆盖', '',
        '```json', json.dumps(diagnostics, ensure_ascii=False, indent=2), '```', '',
        '格式特征：#### 是否存在/数量/标题风险，boxed 是否存在/彼此是否冲突/与官方解析值是否冲突/是否在 #### 之后，官方是否可解析。',
        '算术特征：已检查/正确/错误等式数，错误比例、是否存在错误、最后已检查等式是否错误、平均与最大相对误差。',
        '支持多项四则运算、括号、有限次幂、LaTeX 分数/乘除号、数值等式链与显式小数舍入。',
        '使用受限 AST 遍历与 Fraction 运算，不使用 eval/exec，不执行候选代码。',
        '整数近似/取整没有通用语义判断，仍可能被当成数值不一致；未对全部触发项人工标注，因此不能把触发率当精确率。',
        '遇到变量、单位文字和不支持的 LaTeX 时保守跳过；不检查单位换算、语义关系，也不保证找全错误。',
        '含百分号的等式以及以空格分隔短横线开头的歧义行跳过，避免百分数习惯写法与 Markdown 项目符号误报。',
        '初版在训练集文本审查中发现列表符号、单位屏障和百分号误报；该批已归档到 reliability_draft_v1（含当时源码），本目录为修正后完整重跑。没有依据验证成绩改变模型配置或挑选种子。',
        '第二版保守屏障会跳过数值等式链内部和带后续单位的内联数学式，进一步保留数学分隔符、仅跳过屏障紧邻的等式对；第二版结果及源码保存在 reliability_draft_v2。本目录为最终第三版，同样完整重跑全部预设配置，三个版本均已查看验证结果，应在报告披露开发过程。',
        '检查出数值等式不一致并不等于候选最终答案错误：中间推理与最终答案是不同目标，因此本轮只用作学习特征，不硬性删除候选。',
        '原模型错题的逐式证据见 original_error_feature_evidence.json。报告准确率一律来自官方评分器。', '',
        '## 复现', '', '在 code 目录：`.venv/Scripts/python.exe run_reliability.py`。',
        '仅重建报告：`.venv/Scripts/python.exe run_reliability.py --analysis-only`。',
        '验证本轮预设主种子组合模型：`.venv/Scripts/python.exe evaluate_best.py --experiments runs/reliability`。默认 evaluate_best.py 仍指向旧模型。',
        '模型文件夹包含 config.json 与 weights.npz；推理使用 advanced_submission.py，并需要 advanced_features.py 与 reliability_features.py。',
        '依赖、线程与硬件见 environment.json；搜索成本、资产大小与哈希见 results.json；代码哈希见 source-sha256.json。',
        'AI 使用披露：本轮代码、实验执行和分析由 Codex 协助。学生应复核与理解后写入报告。']
    (ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    write_json(ROOT/'source-sha256.json', {n: hashlib.sha256((BASE/n).read_bytes()).hexdigest()
        for n in ['run_reliability.py', 'reliability_features.py', 'advanced_features.py', 'advanced_submission.py', 'run_four_steps.py']})
    print(json.dumps(summary, indent=2), flush=True)

def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--analysis-only', action='store_true')
    args = parser.parse_args(); exp = Experiments(ROOT, 50)
    if not args.analysis_only:
        write_json(ROOT/'protocol.json', {'seeds': SEEDS, 'epochs': 5, 'modes': ['format', 'arithmetic', 'both'],
            'seed_7506_ablations': True, 'other_seeds_combined_only': True, 'test_used': False})
        for mode in ['format', 'arithmetic']:
            exp.neural(f'{mode}_seed_7506', mode, 'attention', 'listwise', 64, 'reliability', seed=7506, fixed_epochs=5)
        for seed in SEEDS:
            exp.neural(f'both_seed_{seed}', 'both', 'attention', 'listwise', 64, 'reliability', seed=seed, fixed_epochs=5)
    report(exp)

if __name__ == '__main__': main()
