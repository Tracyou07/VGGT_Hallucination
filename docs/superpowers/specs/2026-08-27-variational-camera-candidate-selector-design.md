# 长窗口 Camera Latent 候选排序器设计

状态：用户已选择方案 A，设计冻结

日期：2026-08-27

分支：`codex/vrfm-candidate-selector`

正式输出：`/data/yjh/output/variational_camera_selector/<run_id>/`

## 1. 目标

Phase 1 已证明：在只由长窗口 latent 生成的 32 个 VRFM 方向及多个连续步长中，
固定 10 个 calibration 场景的 80 个 overlap 有 48 个存在 GT-oracle 有用候选；
但完整一步通常过大，而且原始 VRFM 方向在 20-Q 结构化随机对照中并不特殊。

本阶段只回答一个新问题：**不读取短窗口和 GT，能否根据长窗口 latent 与候选
residual，自动选择较好的方向和步长？**

第一阶段不重训 VGGT 或 VRFM，不扩场景，不使用 20-Q 随机方向作为训练样本。
它先复用已经冻结的 10-scene 原始 VRFM 候选集合，训练一个小型
prediction-only 候选排序器。

## 2. 为什么先做排序器

共有三种可行路径：

1. 先训练候选排序器；
2. 直接把 GT 质量权重加入 VRFM 并重训；
3. 先把候选数据扩到全部 50 场景。

用户选择路径 1。它能把“候选生成是否有用”和“长窗口是否能选择候选”分开验证，
并且完全复用现有产物。若 held-out 场景不存在可学习的选择信号，直接重训更大的
quality-weighted VRFM 或扩到 50 场景都缺少依据。

20-Q 的随机正交变换是方向特异性的零假设控制，正式报告明确不支持训练归因；因此
它不能被重新解释成训练 target 或额外数据增强。

## 3. 固定输入与数据单位

输入 run 固定为：

```text
/data/yjh/output/variational_camera_latent/vrfm_camera_20260827T044926Z
```

代码生产提交固定为：

```text
e7f178587b9d80ebf730e9f6e3c2266d49b8b64b
```

场景角色继承原 source manifest，禁止重新随机划分：

- train：前 8 个场景；
- validation：后 2 个场景 `scene0325_01`、`scene0675_00`。

每个场景包含 8 个 primary overlap。每个 overlap 有：

```text
G:       完整长窗口 tokens                 [500, 2048]
X0:      当前 overlap 的长窗口 tokens       [50, 2048]
C_j:     第 j 个 VRFM corrected tokens      [50, 2048], j=0..31
Delta_j: C_j - X0                           [50, 2048]
alpha:   {0, .01, .02, .05, .1, .2, .5, 1}
```

候选状态为 `X0 + alpha * Delta_j`。`alpha=0` 对所有 32 个方向完全相同，每个
overlap 只保留一个 no-op，不能复制 32 次影响排序分布。每个非零 alpha 保留全部
32 个方向，因此每个 overlap 有 `1 + 32*7 = 225` 个选择。

训练共有 64 个 overlap、14,400 个选择；validation 共有 16 个 overlap、3,600
个选择。科学评价单位是 overlap，汇总单位优先是 scene，不能把 3,600 个候选当作
3,600 个独立验证样本。

## 4. Prediction-only 防泄漏契约

现有 source shard 同时包含离线短窗口教师，因此 selector 推理不能直接接收 source
shard 路径。新增一个很小的 long-context shard，只复制以下字段：

- `global_frame_ids [500]`；
- `global_camera_tokens [500,2048]`；
- `overlap_frame_ids [8,50]`；
- `overlap_long_tokens [8,50,2048]`；
- `span_starts [8]`、`source_sample_ids [8]`；
- source/candidate/protocol/producer SHA-256 与角色。

候选 latent 继续读取现有 candidate shard；它只含 `source_long_tokens`、
`corrected_camera_tokens`、`z`、sample seeds 和预测相机结果，不含短窗口或 GT。
新 manifest 将 long-context shard 与 candidate shard 绑定，不复制大型候选 tensor。

严格分成三个 API：

1. `PredictionCandidateDataset` 只接收 long-context/candidate manifest，不能接收
   privileged 路径；
2. `SelectorTrainingDataset` 在训练进程内按 scene、sample ID、sample seed、alpha
   和 SHA-256 将 prediction 记录与 privileged sidecar 连接；
3. `score_candidates` 只接收张量与 checkpoint，函数签名不得含 GT、quality、
   error、depth 或 privileged 参数。

训练标签只来自已经冻结的
`vrfm_residual_alpha_scan_full_context_privileged` sidecar 中的
`relative_improvement [8,32,8]`。任何 sample ID、alpha、seed、candidate digest、
source digest 或场景角色不一致都必须 fail closed。

## 5. 排序模型

第一版使用小型 latent ranker，不读取 RGB、pose、depth 或解码后的 GT 误差。

对 `G`、`X0` 和 `alpha*Delta` 使用共享 `2048 -> 128` token projector，分别汇总：

- 投影后 token 的 mean 与 standard deviation；
- 相邻帧一阶差分的 mean 与 standard deviation。

加入 8-way span embedding、连续 alpha embedding、候选 residual RMS 和可选的原始
16-D `z` 后，经两层 MLP 输出一个标量 utility score。模型保留时间变化摘要，但规模
足够小，适合只有 64 个 train overlap 的第一轮验证。

同时训练一个容量匹配的 residual-only 对照：它看 `X0`、`alpha*Delta`、span、alpha
和 residual RMS，但看不到完整 `G` 摘要。该对照用于区分“长窗口上下文可学习”与
“只根据步长/范数就能猜中”。

推理流程为：

1. 只运行/加载一次长窗口 Camera latent；
2. 用冻结 VRFM 从 prior 采样 32 个 `z` 并生成候选；
3. 为 no-op 与全部非零 alpha 候选打分；
4. 在同一 overlap 内选择最高分，包括允许选择 no-op；
5. 输出选择 ID、z、alpha、corrected latent 和 prediction-only score。

## 6. 质量加权与训练目标

“质量加权”只发生在训练 loss 中，不是推理输入。对同一 overlap 的 225 个候选：

```text
target_probability = softmax(relative_improvement / tau)
pred_probability   = softmax(predicted_score)
loss               = cross_entropy(target_probability, pred_probability)
```

初始 `tau=0.05`，写入冻结配置。它让更好的候选贡献更大，但仍给接近最优的连续
候选非零概率，不强迫模型把连续冗余压成一个硬类别。这正是质量加权与 variational
思路的关系：保留多种候选，只改变模型在给定长窗口下应偏好的概率质量。

训练不使用 20-Q 标签、不删除负候选、不把 GT 数值拼进 feature。标签只决定训练
目标分布。所有归一化统计只可在 8 个 train 场景拟合。

## 7. Smoke、calibration 与对照

### 7.1 One-scene smoke

固定 `scene0000_00`，只做技术和表达能力检查：

- manifest/digest/sample join 完整；
- prediction-only loader 无 privileged 参数；
- 一个 optimizer step 前后向 finite；
- 小数据 overfit 时 listwise loss 明显下降；
- score shard 不含 GT/quality/error/depth 字段；
- checkpoint、optimizer、RNG 和下一 step 可精确恢复。

Smoke 的训练集表现不作为科学结论。技术门控通过后自动进入 8/2 calibration。

### 7.2 固定 8/2 calibration

在 8 个 train 场景训练，在 2 个 validation 场景一次性评价。每个 validation
overlap 比较：

- no-op；
- 固定种子的 uniform random choice；
- residual-only ranker；
- full-context ranker；
- GT oracle upper bound。

报告以下 scene/overlap 级指标：

- selected relative improvement 的 mean、median 与每场景值；
- 改善超过 1% 的 overlap 数量；
- 相对 oracle 的 regret；
- full-context 相对 residual-only、random 和 no-op 的差值；
- top-k oracle coverage（k=1, 4, 8）；
- 校准曲线与 score/真实 utility 的 rank correlation。

固定 2 个 validation 场景只产生探索性结论，不声称统计显著性。分类为：

- `LEARNABLE_SIGNAL`：full-context 在两个 validation 场景的 scene-mean 都优于 no-op，
  且总体优于 residual-only 与固定随机基线；
- `WEAK_SIGNAL`：部分改善，但场景间不一致或不超过 residual-only；
- `NO_GENERALIZATION`：held-out 总体不优于 no-op。

无论属于哪一类，都写出完整报告；弱科学结果不是技术失败。

## 8. 输出契约

新 run 不修改已封存的 Phase 1 目录：

```text
/data/yjh/output/variational_camera_selector/<run_id>/
  run_metadata.json
  manifests/
    long_context_manifest.json
    candidate_binding_manifest.json
    privileged_binding_manifest.json
  prediction_only/
    long_context/<scene>.npz
    scores/<scene>.npz
    selections/<scene>.npz
  training/
    checkpoints/latest.pt
    metrics.jsonl
    training_state.json
  privileged_labels/
    evaluation/<scene>.npz
  reports/
    smoke_summary.json
    calibration_summary.json
  verified_completion.json
```

`prediction_only/scores` 和 `selections` 只保存 sample/candidate IDs、预测 score、
chosen index、z、alpha 与 corrected latent 引用/摘要。真实 relative improvement
只写入 `privileged_labels/evaluation` 和报告；不能回填 prediction-only 文件。

所有文件原子写入并记录 SHA-256。恢复时要求代码提交、输入 run、manifest、split、
模型配置和训练配置完全一致。

## 9. 技术失败与恢复

以下情况 fail closed：

- prediction manifest 或 checkpoint 与封存 Phase 1 digest 不一致；
- train/validation 场景泄漏；
- prediction-only schema 出现短窗口、GT、depth、error、quality 或 privileged 字段；
- candidate、alpha、sample ID、sample seed 或 sidecar join 不一致；
- 输入、loss、gradient、score 或 selected latent 出现 NaN/Inf；
- resume 的 config、source digest、optimizer/RNG 或 completed step 不一致；
- H20 identity、GPU、磁盘、工作树、输入完成标记或活跃任务不符合预检；
- 原子文件、SHA-256 manifest 或完成标记校验失败。

失败时保留所有有效 checkpoint 和已完成 shard；不得覆盖封存的 Phase 1 产物，不得
删除其他任务数据，不得把大型输出拉回 Windows。禁止使用此前暴露的 H20
Hugging Face token。

## 10. 验收与后续决策

CPU 测试必须覆盖 schema、防泄漏、精确 join、no-op 去重、scene split、模型 shape、
listwise loss、train-only normalization、resume、原子写入、报告单位和 H20 runner。
新增测试必须遵循 RED-GREEN；已有 64 个 `variational_camera_latent` 测试必须继续
通过。全仓 Windows 基线中 5 个旧 ScanNet 下载行为测试因测试 PATH 缺少 `dirname`
失败，属于记录在案的无关基线，不在本任务中修改或计入 selector 回归。

H20 完成 1-scene smoke 后自动跑固定 8/2 calibration，输出完整 checkpoint、
prediction-only score/selection、独立 privileged evaluation 和 verified completion。

只有结果为 `LEARNABLE_SIGNAL`，下一阶段才扩到全部 50 场景，并考虑把 selector 的
soft target 进一步蒸馏进 conditional prior 或 quality-weighted VRFM；否则先分析
失败来源，不盲目扩容。

