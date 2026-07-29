# ScanNet-50 Local-Global Consistency 正式实验结论

## 1. 实验身份与完整性

- calibration：`3d5de75_c5c5ae0e55fe`，10 个场景、90/90 个窗口、5,000 帧；
- holdout：`3d5de75_35564d765bb5`，40 个场景、359/359 个窗口、19,930 帧；
- source run：`d33d98b_309a9a586242`；
- split digest：`69c283245c4f220965e6fde3b96192de298e292eb8ca625c94851fe8932cdb8a`；
- frozen threshold digest：`ee010678b5a4dc2c0639a3e3e4df1ca815748dad76393286f9ae759ccd9b4f09`。

上述 source/split 在 split、calibration、holdout 元数据中一致，threshold digest 在 calibration 完成标记、冻结阈值文件和 holdout 完成标记中一致；两次运行均标记 `protocol_complete=true`、`analysis_complete=true`。正式协议为窗口长度 100、步长 50、Camera iterations 4、`pad` 预处理。

冻结的 local-local p95 阈值为：token cosine distance `0.710832`、pose translation `0.031207`、pose rotation `14.603110 deg`，各由 calibration 的 4,000 个可评估重叠帧拟合。

## 2. 指标口径

- 凡含预测的误差或预测间位姿差异，均先使用 **aligned prediction**；GT 始终使用 **raw GT**，从不对 GT 做对齐或修正。
- 每帧退化量定义为 `global aligned error - median local aligned error`。正值表示 local 误差较小，负值表示 global 误差较小。
- local 误差是覆盖该帧的各 local 窗口独立对 raw GT 对齐后误差的中位数。
- global-local 与 local-local 检测分数全部是 prediction-only；GT 只生成独立评估标签，不参与分数、阈值或逐帧筛选。
- holdout 汇总先在场景内统计，再对 40 个场景等权平均。95% CI 为 scene bootstrap（10,000 次，seed 33），不能解释为逐帧独立抽样区间。

## 3. 数据直接支持

### 3.1 平移与旋转结论

| 指标（global - local） | 场景等权估计 | scene-bootstrap 95% CI |
|---|---:|---:|
| 平移退化量 mean | `+0.04447` | `[+0.03382, +0.05697]` |
| 平移退化量 median | `+0.04061` | `[+0.03079, +0.05182]` |
| 平移退化量为正的帧比例 | `83.77%` | `[80.13%, 87.13%]` |
| 旋转退化量 mean | `-3.3618 deg` | `[-4.7902, -2.0994] deg` |
| 旋转退化量 median | `-2.4459 deg` | `[-3.6066, -1.4064] deg` |
| 旋转退化量为正的帧比例 | `29.59%` | `[22.56%, 37.07%]` |

结论具有明确的分量差异：**local 在平移上更准，但 global 在旋转上更准**。两个 mean/median 的 CI 均不跨 0，不能把结果压缩为“local 整体优于 global”。

绝对误差与上述差值一致：

| aligned 绝对误差（场景内 mean，再场景等权） | global | median local |
|---|---:|---:|
| 平移 | `0.08806` `[0.07389, 0.10400]` | `0.04359` `[0.03700, 0.05080]` |
| 旋转 | `3.979 deg` `[3.428, 4.555]` | `7.341 deg` `[6.037, 8.826]` |

### 3.2 Global-local prediction-only 信号

下表相关系数均为“逐场景相关系数的等权平均”；四分位在每个场景内按 prediction-only 分数排序，分离量为 top-Q4 平移退化均值减 bottom-Q1 平移退化均值。

| prediction-only 分数 | 对应退化量 | Pearson（95% CI） | Spearman（95% CI） | 平移 Q4-Q1（95% CI） |
|---|---|---:|---:|---:|
| global-local token cosine | 平移 | `0.124` `[0.051, 0.195]` | `0.125` `[0.056, 0.194]` | `+0.01712` `[+0.00585, +0.02915]` |
| global-local pose translation | 平移 | `0.144` `[0.043, 0.235]` | `0.095` `[0.024, 0.162]` | `+0.01444` `[+0.00440, +0.02464]` |
| global-local pose rotation | 旋转 | `-0.228` `[-0.360, -0.096]` | `-0.190` `[-0.307, -0.069]` | `+0.00685` `[-0.00681, +0.02085]`* |

\* 此列仍是按该分数分组后的**平移**退化 Q4-Q1；区间跨 0。正式实现未发布旋转退化的四分位字段。

因此，token 与 pose-translation disagreement 对“global 平移相对 local 变差”有弱但可复现的排序能力；效应量不大，不能当作逐帧真值。pose-rotation disagreement 与旋转退化呈负相关，即 disagreement 越大时，local 相对 global 的旋转劣化反而更明显。

### 3.3 Local-local reliability gate

gate 未达到原先“通用可靠性筛选器”的预期：

- 只有重叠区的约 `80.03%` 帧可评估 gate；其中 token gate 保留 `97.12%`，pose gate 保留 `87.81%`，筛除强度有限。
- local-local 分数与退化量没有预期的正向关系。例：local-local pose-translation 对平移/旋转退化的 Pearson 分别为 `-0.093`、`-0.124`；local-local pose-rotation 分别为 `-0.076`、`-0.215`。
- gate 后部分平移指标确有改善：pose-translation 的 Pearson 从 `0.144` 到 `0.205`，平移 Q4-Q1 从 `0.01444` 到 `0.02350`；但 token 改善很小，pose-rotation 的旋转 Pearson 绝对值从 `0.228` 降到 `0.177`。这不足以支持“gate 普遍增强信号”的结论。

### 3.4 `scene0150_00` 特例

该场景按真实长度使用 430 帧和 8 个窗口（其余场景为 500 帧、9 个窗口），因此 holdout 总数为 359 个窗口、19,930 帧。它自身仍给出同向结果：平移 mean `+0.00441`，旋转 mean `-0.8516 deg`。剔除它后，40 场景等权 mean 仅由 `+0.04447/-3.3618 deg` 变为 `+0.04550/-3.4261 deg`，不改变结论方向。

## 4. 解释性推测（非结论）

- local 窗口较短可能降低长序列平移漂移，而独立窗口的姿态规范化或上下文不足可能损害旋转；当前结果只显示这种现象与数据相容，不能识别机制。
- local-local disagreement 与实际优劣不单调，可能同时受窗口边界、场景运动和对齐条件影响；本实验没有因果消融，不能据此归因。

## 5. 对 Round 2 / DiT 的约束与后续待验证

**当前约束：**

- Round 2 不得给帧或样本简单标注“local 整体优于 global”；至少要分别保留平移与旋转标签、分数和验收条件。
- prediction-only disagreement 只能作候选排序信号，不能替代 GT，也不能仅凭 local-local gate 把 local Camera pose 当成整体教师。
- DiT 数据对若采用 local 目标，必须避免把平移收益与旋转劣化合并成单一监督结论；完整位姿修正需要分量感知或明确的多指标约束。

**后续待验证：**

1. 在冻结 Camera Head 的 replacement/blending 实验中，验证候选 latent 是否真实改善平移，同时不显著恶化旋转。
2. 评估平移/旋转分头筛选、分头残差目标或 Pareto 约束；这些是待测方案，不是本实验已经证明的机制或最佳实现。
3. 在独立数据和不同窗口设置上复验相关性、四分位分离及 gate 阈值迁移性，再决定是否进入 DiT 规模化构造。
