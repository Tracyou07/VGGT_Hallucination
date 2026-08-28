# Variational Camera Selector 安全门控正式实验报告

日期：2026-08-28
正式 run：`selector_safety_20260828T035545Z`
H20 输出：`/data/yjh/output/variational_camera_selector_safety/selector_safety_20260828T035545Z`
代码提交：`e2e56cbf37938aadb7f167c67b0619cef996a498`

## 一句话结论

简单的 prediction-only 置信度门控能把原 selector 的大部分坏修正挡掉，但仍不能在
跨场景 OOF 中得到正平均收益。严格协议因此冻结为全 no-op，结论为
`policy_deployable=false`、验证集 `SAFE_NOOP`。这批 gated latent 是安全、可复算的
诊断产物，但还不是可以扩成满意训练集的修正 latent。

## 实验设计

目标不是重新选择 225 个候选，而是在原 full-context ranker 的 top-1 和 no-op 之间
做最后一道安全判断。门控只看 prediction-only 信号：

- top-1 的 `alpha`；
- top-1 相对 no-op 的标准化分数优势；
- top-1 相对第二名的标准化间隔；
- residual-only ranker 对同一候选的标准化支持；
- full-context 与 residual-only 两个 ranker 的 top-1 是否一致。

8 个训练场景做严格 leave-one-scene-out：每一折只用另外 7 个场景训练 800 步，先对
held-out 场景生成 prediction-only score shard。8 个 score shard 全部封存后，才允许
读取各自 privileged utility。门控阈值只在这 64 个 OOF overlap 上拟合。

冻结条件为：

- coverage 至少 12.5%；
- 被执行修正中，正收益比例至少 70%；
- utility 小于 -0.05 的严重失败比例至多 5%；
- 64 个 overlap（包括被拒绝后记作 no-op 的样本）平均收益大于 0；
- 最差场景平均收益至少 -0.01；
- 额外再做一层 7 场景拟合、1 场景测试的 scene-level crossfit。

只有全部通过，规则才允许部署。之后先在两个锁定验证场景写出 gated latent，再读取
验证标签评价。

## 完整性与运行状态

- 8/8 个 OOF 折均完成 800/800 步；
- 8 个 held-out score shard 与 8 个 privileged evaluation sidecar 完整；
- 2 个锁定验证场景均生成 prediction-only gated latent；
- 2 个验证 evaluation sidecar 完整；
- 8 个 fold stderr 均为 0 字节；
- run 总大小约 4.6 GiB；
- `verified_completion.json` 已独立重算通过。

关键摘要：

- `verified_completion.json` SHA-256：
  `9aad3fc0e040cc95328410850b16a8932c84c245ab5d75003be77dde16ab30dc`
- `oof_gate_fit.json` SHA-256：
  `7bb47ea745c27c74425767a2ef403b548e136ed14f66bf9a5ec5a4992938a27f`
- `validation_summary.json` SHA-256：
  `02c82a1b2b65d708f5f0a729f3302b088f9fb8872484a6cf1f638ff40d858c25`

## OOF 结果

原始 OOF top-1（不加门）在 64 个 overlap 上：

- 平均 utility：`-0.236492`；
- 正收益：14/64；
- 严重失败（utility < -0.05）：13/64。

共枚举 640 条冻结候选规则。通过各单项门槛的规则数：

| 条件 | 通过规则数 / 640 |
|---|---:|
| coverage | 16 |
| 正收益比例 | 6 |
| 严重失败率 | 614 |
| 平均 utility > 0 | **0** |
| 最差场景下界 | 614 |
| 全部条件 | **0** |

因此没有可部署规则。最接近的两条规则是：

| 规则 | coverage | 正收益比例 | 严重失败 | 全体平均 utility | 最差场景 |
|---|---:|---:|---:|---:|---:|
| 最佳平均值 | 19/64 | 12/19 = 63.16% | 2/19 = 10.53% | -0.002078 | -0.012432 |
| 最佳精度 | 12/64 | 9/12 = 75.00% | 2/12 = 16.67% | -0.002096 | -0.012432 |

两条规则都只允许很小的步长（`max_alpha=0.02`；实际被选中的 top-1 均为
`alpha=0.01`）。最佳精度规则还要求两个 ranker 的 top-1 完全一致，但仍出现两次
严重失败：

- `scene0013_02:overlap_000`：utility `-0.126627`；
- `scene0084_01:overlap_007`：utility `-0.088800`。

这两次失败并不是“低置信度异常”：它们都有正的 score advantage、正的 prominence、
正的 residual support，而且 full-context 与 residual-only top-1 一致。换句话说，当前
五个简单置信度特征无法把它们和真正有益的小步修正分开。

## 锁定验证结果

由于 OOF 没有规则通过，冻结 policy 按协议变成全 no-op：

- `policy_deployable=false`；
- validation coverage：0/16；
- validation mean utility：0；
- classification：`SAFE_NOOP`。

这不是“模型修好了”，而是安全系统正确拒绝了一组尚不能证明有效的修正。它没有给
验证 latent 带来伤害，也没有产出新的高质量修正样本。

## 人话解释

小步长确实比原来的大步 argmax 安全得多：平均损失从 `-0.2365` 缩小到约
`-0.0021`。但是偶尔仍会出现一次看起来非常自信、实际上方向完全错了的修正；一次
`-0.1266` 的错误足以吃掉很多个 `+0.01` 左右的小收益。

所以问题已经从“步子太大”收缩成“模型不知道自己在哪些场景会自信地判断错”。继续
放松门槛只会把坏样本重新放进训练集，不能解决问题。

## 对 latent 训练数据的含义

本 run 提供了严格绑定、prediction-only、可复算的 gated latent，以及与其物理分离的
privileged 标签，可用于开发和测试下一个风险模型。但冻结 policy 是 no-op，因此这些
gated latent 不应被冒充为新的优质修正训练数据。

## 下一步建议

不要继续调这五个阈值。下一步应扩大 scene-level OOF 数据，再训练一个真正的
downside-aware selector：

1. 从 ScanNet-50 余下场景生成更多 VRFM 候选和 privileged utility，至少把开发场景
   扩到 20–30 个；
2. 输入仍保持 prediction-only，但使用更丰富的长窗口/候选 latent 特征；
3. 同时预测期望 utility、`P(utility > 0)` 和负向分位数（例如 10% quantile）；
4. 只有预测下分位数仍大于 0 时才执行修正，否则 no-op；
5. 继续按 scene 做 OOF，最后留出从未用于设计的新场景验证。

这一步才会直接回答：VRFM 的随机方向中，模型能否学会避开“高置信度但方向错误”的
尾部风险，并稳定生产可用于训练的 corrected latent。
