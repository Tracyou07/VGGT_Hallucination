# Variational Camera Selector A 正式实验报告

日期：2026-08-27  
正式 run：`selector_A_20260827T153304Z`  
H20 输出：`/data/yjh/output/variational_camera_selector/selector_A_20260827T153304Z`  
代码提交：`21be5b2214b963c45880aa1e21ccbd83d7242b35`

## 一句话结论

长窗口 latent 确实提供了一点额外排序信息，但当前“直接取最高分候选”的 selector
不能在新场景中稳定超过 no-op，因此结论是 `NO_GENERALIZATION`。它现在适合做诊断，
还不适合直接批量生产训练数据。

## 实验问题

对每个 50 帧 overlap，只给模型长窗口 prediction-only Camera latent、VRFM 候选方向
和步长，不给 GT、深度、短窗口 latent 或误差标签。模型需要从 225 个选择中选一个：

- 1 个 no-op；
- 32 个 VRFM 方向；
- 每个方向 7 个非零步长。

固定 8 个场景、64 个 overlap 训练；固定另外 2 个从未参与训练的场景、16 个 overlap
评价。GT relative improvement 只在独立 privileged sidecar 中用于训练 loss 和最后评价。

## 产物与技术门控

- one-scene smoke：30/30 步通过，两个 ranker 的 loss 都下降；
- calibration：800/800 步完成；
- full-context 与容量匹配的 residual-only 对照同时训练；
- 两个 validation 场景都生成 prediction-only score shard；
- 两个 validation 场景都生成实际选中的 corrected Camera latent，而不只是索引；
- privileged evaluation 与 prediction-only 产物物理分离；
- 独立重跑 `verify` 通过，stderr 为空；
- run 总大小约 709 MiB。

校验摘要：

- `verified_completion.json` SHA-256：
  `5d2e562b7e6797f69bfcb4af976d1b17afb7d7b6f777cf8a548ae181e9747e20`
- `calibration_summary.json` SHA-256：
  `28bd556a998496ddd747414afa71787b4d17216d716d7a1c5b098aa5089bfd01`

## Held-out 结果

| 方法 | 16 个 overlap 平均 relative improvement | 中位数 | >1% 的数量 |
|---|---:|---:|---:|
| no-op | 0.0000 | 0.0000 | 0 |
| uniform random | -2.5776 | -0.1675 | 5 |
| residual-only selector | -0.3020 | -0.0092 | 0 |
| full-context selector | -0.2867 | 0.0000 | 1 |
| GT oracle upper bound | +0.1676 | +0.0252 | 8 |

full-context 相比 residual-only 高 `+0.01525`，相比 random 高 `+2.29087`，但相比
no-op 低 `-0.28675`。它的候选分数与真实 utility 的 overlap-mean Spearman 相关为
`0.60175`；oracle top-1 / top-4 / top-8 coverage 分别为 `31.25% / 37.5% / 37.5%`。

按场景看：

| validation 场景 | full-context | residual-only | no-op | oracle |
|---|---:|---:|---:|---:|
| `scene0325_01` | -0.5717 | -0.6017 | 0.0000 | +0.1963 |
| `scene0675_00` | -0.0018 | -0.0023 | 0.0000 | +0.1390 |

## 人话解释

模型不是完全没学到东西。它明显比随机乱选安全，整体排序相关性也不低；长窗口版本
还略好于看不到完整长窗口的 residual-only 对照。问题出在最后一步“取最高分”：

- 在 `scene0325_01`，模型对前几个 overlap 太自信，常选 `alpha=0.2` 或 `0.5`。
  例如 overlap 3 选了 `alpha=0.5`，真实收益为 `-1.6503`，而该 overlap 的 oracle
  只需 `alpha=0.02`，收益约 `+0.0042`；
- 在 `scene0675_00`，模型多数时间选择 no-op，基本避免了灾难，但也错过了 3 个明显
  有益的机会，其中一个 oracle 可达 `+0.5271`。

所以当前现象不是“长窗口完全没有可学信号”，而是“有粗粒度排序信号，但跨场景的
风险校准不够”。直接 argmax 会把少数过度自信的大步长错误放大，平均收益因此低于
永远不修正。

## 对 latent 训练数据的含义

本 run 已经产出格式正确、prediction-only 的 selected corrected latent，可用于检查数据
结构和训练接口。但因为 held-out 平均收益低于 no-op，这批 selection 不能被当成
“满意的大规模优质训练集”直接扩到 50 场景。否则会把过度修正的坏 latent 一起放大。

## 下一步

下一步不扩场景，先在 8 个训练场景内部做 leave-one-scene-out 的风险门控：

1. 同时预测候选的期望收益和“优于 no-op 的概率”；
2. 只有保守下界仍大于 0 时才允许离开 no-op；
3. 大步长惩罚或 alpha 上限只能用 train-scene 交叉验证确定，不能回看这两个已经用过的
   validation 场景调参；
4. 规则冻结后，再用新的未见场景做一次真正验证。

目标是保留当前已经出现的排序信号，同时把“宁可 no-op，也不要高置信度犯大错”写进
选择规则。只有新的 held-out 结果稳定超过 no-op，才进入 50 场景 latent 数据生成。
