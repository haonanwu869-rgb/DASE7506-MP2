"""Five predeclared seeds, fixed architecture/epochs; validation diagnostics only."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '4'
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
from run_four_steps import Experiments, BASE, write_json
from common import DATA, read_jsonl, parse_answer
from advanced_features import text_features

SEEDS = [7506, 17, 42, 2026, 3407]
ROOT = BASE/'runs'/'stability'
OLD = BASE/'runs'/'four_steps'

def analyze(exp):
    bank, y = exp.vbank, exp.vy
    refs = read_jsonl(DATA/'gsm8k_validation.jsonl')
    old = json.loads((OLD/'step3_attention-predictions.json').read_text())
    assert old['ids'] == [r['id'] for r in bank]
    picked = np.array(old['selected'])
    original_ok = y[np.arange(len(y)), picked].astype(bool)
    features = text_features(bank, exp.varrays['embeddings'])
    selected = []
    for seed in SEEDS:
        p = json.loads((ROOT/f'attention_seed_{seed}-predictions.json').read_text())
        assert p['ids'] == old['ids']
        selected.append(p['selected'])
    selected = np.array(selected)
    ok = y[np.arange(len(y))[None, :], selected].astype(bool)
    assert np.array_equal(selected[0], picked), 'Original seed did not reproduce'
    acc = 100*ok.mean(1); hard_acc = 100*ok[:, exp.hard].mean(1)
    errors = []
    groups = {}
    for i, row in enumerate(bank):
        answers = [parse_answer(c['text']) for c in row['candidates']]
        gold = parse_answer(refs[i]['answer'])
        counts = Counter(a for a in answers if a is not None)
        correct_votes = counts[gold]
        wrong_votes = max([n for a, n in counts.items() if a != gold] or [0])
        vote_relation = ('correct_more' if correct_votes > wrong_votes else
                         'tie' if correct_votes == wrong_votes else 'wrong_more')
        for group in [f'correct_candidates_{int(y[i].sum())}', f'distinct_answers_{len(counts)}',
                      'votes_'+vote_relation, 'hard' if exp.hard[i] else 'easy']:
            groups.setdefault(group, []).append(i)
        if not original_ok[i]:
            errors.append({'validation_index': i, 'id': row['id'], 'question': row['question'],
                'gold_answer': gold, 'reference_reasoning': refs[i]['answer'],
                'selected_index': int(picked[i]), 'selected_answer': answers[picked[i]],
                'correct_indices': np.flatnonzero(y[i]).tolist(), 'parsed_answers': answers,
                'correct_votes': correct_votes, 'max_wrong_votes': wrong_votes,
                'vote_relation': vote_relation, 'seed_wrong_count': int((~ok[:, i]).sum()),
                'seed_selections': selected[:, i].tolist(),
                'simple_equation_detected': bool((features[i, :, 10] > 0).any()),
                'selected_has_detected_bad_equation': bool(features[i, picked[i], 12] > 0),
                'candidates': row['candidates']})
    strata = {g: {'n': len(ix), 'original_correct': int(original_ok[ix].sum()),
                   'original_accuracy': 100*float(original_ok[ix].mean()),
                   'seed_mean_accuracy': 100*float(ok[:, ix].mean())} for g, ix in groups.items()}
    pairwise = []
    for j in range(1, len(SEEDS)):
        pairwise.append({'seed': SEEDS[j], 'fixes_original_errors': int((ok[j] & ~ok[0]).sum()),
                        'breaks_original_correct': int((~ok[j] & ok[0]).sum())})
    other_models = {}
    for name in ['step1_logreg', 'step2_mlp_listwise', 'step3_features_mlp', 'step4_tree_lambdarank']:
        p = json.loads((OLD/f'{name}-predictions.json').read_text())
        assert p['ids'] == old['ids']
        other_ok = y[np.arange(len(y)), p['selected']].astype(bool)
        other_models[name] = {'fixes_original_errors': int((other_ok & ~original_ok).sum()),
                             'breaks_original_correct': int((~other_ok & original_ok).sum())}
    stats = {'seeds': SEEDS, 'fixed_epochs': 5, 'original_seed_reproduced': True,
        'mean_accuracy': float(acc.mean()), 'sample_std_percentage_points': float(acc.std(ddof=1)),
        'min_accuracy': float(acc.min()), 'max_accuracy': float(acc.max()),
        'hard_mean_accuracy': float(hard_acc.mean()), 'hard_sample_std_pp': float(hard_acc.std(ddof=1)),
        'all_five_wrong': int((~ok.any(0)).sum()), 'at_least_one_seed_wrong': int((~ok.all(0)).sum()),
        'original_errors_fixed_by_any_other_seed': int((~original_ok & ok.any(0)).sum()),
        'original_error_vote_relations': dict(Counter(e['vote_relation'] for e in errors)),
        'original_errors_with_detected_equation': sum(e['simple_equation_detected'] for e in errors),
        'original_errors_selected_detected_bad_equation': sum(e['selected_has_detected_bad_equation'] for e in errors),
        'seed_complementarity': pairwise, 'other_model_complementarity': other_models,
        'strata': strata}
    write_json(ROOT/'analysis.json', stats)
    write_json(ROOT/'original_61_errors.json', errors)
    write_json(ROOT/'all_validation_seed_outcomes.json', [dict(validation_index=i,
        wrong_count=int((~ok[:, i]).sum()), selections=selected[:, i].tolist()) for i in range(len(y))])
    lines = ['# 原注意力模型的 61 道错题', '',
             '以下为验证集离线分析，包含参考答案，禁止导入推理代码或作为训练样本。',
             '编号为 validation 文件的零起始行号；候选编号为 0–3。语义错误类型需人工阅读确认。', '']
    for e in errors:
        lines += [f"## validation_index={e['validation_index']}", '', e['question'], '',
            f"参考数值：{e['gold_answer']}；模型选 C{e['selected_index']}={e['selected_answer']}；"
            f"正确候选：{e['correct_indices']}；5 个种子中 {e['seed_wrong_count']} 个选错。", '',
            '### 老师参考解答', '', e['reference_reasoning'], '']
        for c in e['candidates']:
            lines += [f"### 候选 C{c['index']}"+('（原模型选择）' if c['index']==e['selected_index'] else ''), '', c['text'], '']
    (ROOT/'ERROR_CASES.md').write_text('\n'.join(lines), encoding='utf-8')
    lines = ['# 注意力模型：多种子稳定性与错题分析', '',
        '预先固定种子 7506、17、42、2026、3407。固定原模型结构、特征、学习率和 5 轮完整训练，仅改变初始化、dropout 和训练题目顺序。',
        '5 轮来自上一轮训练集内部验证。本轮不为各个种子重新选择轮数，不依据验证结果挑种子。原始模型和最佳模型指针保持不变。',
        '所有参数仍仅拟合训练集；测试集未读取。验证集诊断不能用于把这些题的参考答案加入训练。', '',
        '|种子|正确数/798|整体准确率|不同答案子集（355题）|', '|---|---:|---:|---:|']
    for j, seed in enumerate(SEEDS):
        lines.append(f'|{seed}|{ok[j].sum()}|{acc[j]:.2f}%|{hard_acc[j]:.2f}%|')
    lines += ['', f"整体均值 **{acc.mean():.2f}%**，样本标准差 **{acc.std(ddof=1):.2f} 个百分点**；"
        f"范围 {acc.min():.2f}%–{acc.max():.2f}%。", '',
        f"不同答案子集均值 **{hard_acc.mean():.2f}%**，样本标准差 **{hard_acc.std(ddof=1):.2f} 个百分点**。",
        '这是同一验证集上的训练随机性波动，不是泛化误差置信区间；模型结构此前已通过该验证集选择。', '',
        '## 错题结构（精确计数，不代替语义归因）', '',
        f"- 原模型 61 道错题中，{stats['original_errors_fixed_by_any_other_seed']} 道至少被一个其他种子答对。",
        f"- {stats['all_five_wrong']} 道题五个种子全部选错；共 {stats['at_least_one_seed_wrong']} 道题至少一个种子选错。",
        f"- 错误答案票数超过正确答案：{stats['original_error_vote_relations'].get('wrong_more',0)} 道；"
        f"票数并列：{stats['original_error_vote_relations'].get('tie',0)} 道；"
        f"正确答案票数更多仍选错：{stats['original_error_vote_relations'].get('correct_more',0)} 道。",
        f"- 当前简单等式特征在错题中仅覆盖 {stats['original_errors_with_detected_equation']} 道；"
        f"选中候选被该规则发现算术错误：{stats['original_errors_selected_detected_bad_equation']} 道。",
        '这说明简单等式特征覆盖有限；没有检测到错误不等于推理正确，也不能据此断言不存在算术错误。', '',
        '## 分层准确率', '', '|分组|题数|原模型准确率|5种子均值|', '|---|---:|---:|---:|']
    for group, r in sorted(strata.items()):
        lines.append(f"|{group}|{r['n']}|{r['original_accuracy']:.2f}%|{r['seed_mean_accuracy']:.2f}%|")
    lines += ['', 'correct_candidates 为正确候选条数；votes 为相同解析数值的票数比较；hard 为至少两个不同可解析答案。', '',
        '## 已有模型的互补性', '', '|模型|修复原模型错题数|同时新增错题数|', '|---|---:|---:|']
    for name, r in other_models.items():
        lines.append(f"|{name}|{r['fixes_original_errors']}|{r['breaks_original_correct']}|")
    lines += ['', '互补性只表示存在不同错误模式，不意味着平均分数或投票一定提升；本轮没有训练或选择融合方案。', '',
        '## 复现', '', '在 code 目录运行 `.venv/Scripts/python.exe analyze_stability.py`；仅重新生成分析使用 `--analysis-only`。',
        '每个种子的配置、权重、损失历史、官方预测/评分、时间与资产哈希在当前目录。环境记录在 environment.json。',
        '全部 61 道原模型错题和四个候选见 ERROR_CASES.md / original_61_errors.json。',
        'AI 使用披露：实验脚本、训练执行及分析由 Codex 协助；报告需由学生理解、核对并按课程要求披露。']
    (ROOT/'ANALYSIS.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    annotations_path = ROOT/'review_annotations.json'
    if annotations_path.exists():
        annotations = json.loads(annotations_path.read_text(encoding='utf-8'))
        assert set(annotations) == {str(e['validation_index']) for e in errors}
        counts = Counter(v[0] for v in annotations.values())
        review = ['# 61 道错题逐题阅读记录', '',
            'Codex 已逐题阅读问题、老师参考解答与原模型所选候选。这是 AI 阅读归因，尚未经学生独立复核。',
            '每题选一个主要类型；可能同时有其他错误。归因描述候选文本的问题，不证明选择器内部的因果机制。',
            '所有统计保持官方评分不变；这些记录仅用于离线分析，不能作为训练数据或推理查表。', '',
            '|主要类型|题数|', '|---|---:|']
        for kind, n in counts.items(): review.append(f'|{kind}|{n}|')
        review += ['', '## 逐题依据', '', '|validation_index（零起始）|主要类型|依据|', '|---|---|---|']
        for e in errors:
            kind, note = annotations[str(e['validation_index'])]
            review.append(f"|{e['validation_index']}|{kind}|{note}|")
        review += ['', '## 有针对性的后续方向', '',
            '1. 先改进输入侧的格式风险特征：官方解析值与最后 boxed 数值是否一致、最后 #### 是否疑似正文标题。不修改官方解析器。',
            '2. 扩展算术特征覆盖多项和、LaTeX 分数与乘法、内联表达式，先用训练集内部验证检查误报；现有整行二元等式覆盖过窄。',
            '3. 针对单位与数量关系，考虑训练集内部构造受控负例或增加问题—候选数值/单位一致性特征；不能把验证错题及答案回灌训练。',
            '4. 原模型在只有一个正确候选的题目上为 74/117，五种子均值为 58.63%；可在训练集上研究按题难度重加权，避免过度依赖多数答案。',
            '5. 融合值得作为下一轮独立实验，但 42 题所有种子都错，光换种子不能覆盖这些失败；先比较固定等权方案，避免反复调权重过拟合验证集。',
            '6. 编号 32 与 661 存在题面/参考疑似冲突，编号 733 是解析规则问题，应在报告说明并请老师核实；不自行更改标签或评分。']
        (ROOT/'REVIEWED_ERRORS.md').write_text('\n'.join(review)+'\n', encoding='utf-8')
        with (ROOT/'ANALYSIS.md').open('a', encoding='utf-8') as f:
            f.write('\n## 全部错题的逐题阅读\n\n详见 REVIEWED_ERRORS.md，含 61 题逐题依据、主要类型计数、数据疑点与后续改进方向。\n')
        write_json(ROOT/'review_category_counts.json', dict(counts))
    write_json(ROOT/'source-sha256.json', {n: hashlib.sha256((BASE/n).read_bytes()).hexdigest()
        for n in ['analyze_stability.py','run_four_steps.py','advanced_features.py','advanced_submission.py']})
    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)

def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--analysis-only', action='store_true')
    args = parser.parse_args()
    exp = Experiments(ROOT, 50)
    if not args.analysis_only:
        write_json(ROOT/'protocol.json', {'seeds': SEEDS, 'epochs': 5, 'architecture': 'original 64-hidden attention',
            'vary': ['initialization', 'dropout', 'training shuffle'], 'test_used': False})
        for seed in SEEDS:
            exp.neural(f'attention_seed_{seed}', 'text', 'attention', 'listwise', 64, 'stability',
                       seed=seed, fixed_epochs=5)
    analyze(exp)

if __name__ == '__main__': main()
