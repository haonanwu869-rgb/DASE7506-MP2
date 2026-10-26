# 解析格式与算术特征实验

新增 8 个格式风险特征与 8 个算术特征；输入只来自候选文本，不使用参考答案生成特征。
官方解析器、评分器、数据和旧模型均保持原样。本轮未读取测试集，未更换默认最佳模型指针。
与原注意力模型使用相同 64 隐层、5 轮、学习率、dropout 和完整训练集；改变输入维数意味着初始化不能逐参数完全配对。
预设实验：种子 7506 比较仅格式、仅算术、两者结合；两者结合另用 17/42/2026/3407 验证稳定性，不根据验证结果追加调参。

|配置|种子|正确数/798|准确率|不同答案子集|完整预测+评分秒|
|---|---:|---:|---:|---:|---:|
|旧注意力|7506|737|92.36%|82.82%|见旧记录|
|format_seed_7506|7506|730|91.48%|80.85%|1.40|
|arithmetic_seed_7506|7506|737|92.36%|82.82%|1.72|
|both_seed_7506|7506|737|92.36%|82.82%|1.63|
|both_seed_17|17|738|92.48%|83.10%|1.65|
|both_seed_42|42|742|92.98%|84.23%|1.60|
|both_seed_2026|2026|732|91.73%|81.41%|1.67|
|both_seed_3407|3407|733|91.85%|81.69%|1.66|

## 五个种子的比较

|种子|旧模型正确数|新组合正确数|净变化|修复/新增错误|
|---|---:|---:|---:|---:|
|7506|737|737|+0|16/16|
|17|732|738|+6|18/12|
|42|734|742|+8|17/9|
|2026|731|732|+1|13/12|
|3407|732|733|+1|12/11|

旧均值 91.88%；新均值 92.28%，标准差 0.51 个百分点；平均变化 +0.40 个百分点。
不同答案子集：旧均值 81.75%，新均值 82.65%。
五种子共享同一验证集，不能当作五个独立数据集；不是泛化置信区间。

## 检查覆盖

```json
{
  "train": {
    "candidate_count": 19020,
    "hash_box_disagree": 30,
    "heading_like_hash": 1,
    "arithmetic_checked_candidates": 9200,
    "arithmetic_bad_candidates": 252,
    "bad_among_final_correct": 33,
    "bad_among_final_incorrect": 219
  },
  "validation": {
    "candidate_count": 3192,
    "hash_box_disagree": 8,
    "heading_like_hash": 1,
    "arithmetic_checked_candidates": 1524,
    "arithmetic_bad_candidates": 36,
    "bad_among_final_correct": 3,
    "bad_among_final_incorrect": 33,
    "original_61_errors_with_bad_arithmetic": 3,
    "original_61_errors_with_any_arithmetic_checked": 27,
    "original_61_errors_hash_box_disagree": 1
  }
}
```

格式特征：#### 是否存在/数量/标题风险，boxed 是否存在/彼此是否冲突/与官方解析值是否冲突/是否在 #### 之后，官方是否可解析。
算术特征：已检查/正确/错误等式数，错误比例、是否存在错误、最后已检查等式是否错误、平均与最大相对误差。
支持多项四则运算、括号、有限次幂、LaTeX 分数/乘除号、数值等式链与显式小数舍入。
使用受限 AST 遍历与 Fraction 运算，不使用 eval/exec，不执行候选代码。
遇到变量、单位文字和不支持的 LaTeX 时保守跳过；不检查单位换算、语义关系，也不保证找全错误。
含百分号的等式以及以空格分隔短横线开头的歧义行跳过，避免百分数习惯写法与 Markdown 项目符号误报。
初版在训练集文本审查中发现列表符号、单位屏障和百分号误报；该批已归档到 reliability_draft_v1（含当时源码），本目录为修正后完整重跑。没有依据验证成绩改变模型配置或挑选种子。
检查出数值等式不一致并不等于候选最终答案错误：中间推理与最终答案是不同目标，因此本轮只用作学习特征，不硬性删除候选。
原模型错题的逐式证据见 original_error_feature_evidence.json。报告准确率一律来自官方评分器。

## 复现

在 code 目录：`.venv/Scripts/python.exe run_reliability.py`。
仅重建报告：`.venv/Scripts/python.exe run_reliability.py --analysis-only`。
模型文件夹包含 config.json 与 weights.npz；推理使用 advanced_submission.py，并需要 advanced_features.py 与 reliability_features.py。
依赖、线程与硬件见 environment.json；搜索成本、资产大小与哈希见 results.json；代码哈希见 source-sha256.json。
AI 使用披露：本轮代码、实验执行和分析由 Codex 协助。学生应复核与理解后写入报告。
