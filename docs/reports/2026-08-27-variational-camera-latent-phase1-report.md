# 长窗口 Camera Latent 修正：Phase 1 当前报告

日期：2026-08-27

代码分支：`codex/vrfm-random-null20`

代码提交：`e7f178587b9d80ebf730e9f6e3c2266d49b8b64b`

正式运行：`/data/yjh/output/variational_camera_latent/vrfm_camera_20260827T044926Z`

## 1. 我们在验证什么

目标是判断：只给模型一次 500 帧长窗口的 VGGT Camera latent，能否产生类似
100 帧短窗口所提供的有用修正。短窗口只在离线构造训练数据时使用；正式推理不得
读取短窗口、GT pose、GT depth 或误差标签。

数据始终分成两部分：

- `prediction_only` 保存长窗口 latent、候选修正和生成所需信息；
- `privileged_labels` 单独保存 GT 与误差标签，只通过 sample ID 关联。

Phase 1 先回答“候选集合里是否存在有用修正”，还没有训练出可部署的自动选择器。

## 2. 已完成的实验链

### 2.1 基础 VRFM

在 10 个 calibration 场景的 80 个相邻 overlap 上，模型从完整 500 帧 latent
读取上下文，并为每个 50 帧 overlap 采样 32 个固定 `z` 的修正候选，共 2,560
个候选。推理只使用长窗口 latent。

直接走完整一步时信号较弱：只有 4 个 overlap 得到正改善，best-of-32 的中位
相对改善为 `-4.4576`。这说明模型产生了不同方向，但完整修正通常过大或方向不准。

### 2.2 短窗口方向的步长扫描

将短窗口提供的方向放回完整 500 帧上下文解码，并扫描
`alpha = {0, .01, .02, .05, .1, .2, .5, 1}`：

- 45/80 overlap 存在至少 1% 的改善；
- 中位 best nonzero 相对改善为 `0.04357`；
- 42 个 overlap 是“小步有用、整步无用”；
- 左右教师分别成为最佳方向 20 次和 25 次。

因此短窗口确实经常提供有用方向，但最佳修正多为连续的小步，而不是必须跳到一个
完整、固定的离散终点。

### 2.3 VRFM 候选方向的步长扫描

对 2,560 个 VRFM 候选方向做同样的完整上下文步长扫描：

- 1,181/2,560 个方向在 oracle 评价下有用；
- 48/80 overlap 至少有一个有用候选；
- 中位 best nonzero 相对改善为 `0.06364`；
- 42 个 overlap 由小步修正救回；
- 只有 6 个 overlap 的完整一步有用。

这证明 VRFM 候选集合包含有价值的修正方向，但目前必须靠 GT oracle 挑方向和步长。

### 2.4 20-Q 结构化随机零假设

为检验原始 VRFM 修正坐标是否特殊，固定相同场景和 oracle 预算，生成 20 个预注册
的结构化随机正交变换 `Q`。正式结果为：

| 指标 | 数值 |
|---|---:|
| 原始未旋转方向得分 | 0.083859 |
| 20 个随机 Q 得分中位数 | 0.124357 |
| 随机 Q 最小 / 最大 | 0.080115 / 0.172191 |
| 随机 Q 优于 / 劣于原始方向 | 17 / 3 |
| 原始方向在 21 个方案中的名次 | 18 |
| 原始方向异常好的一侧 p 值 | 0.857143 |
| 原始方向异常差的一侧 p 值 | 0.190476 |
| 双侧 p 值 | 0.380952 |

pilot `Q0` 的得分 `0.159445` 仅作参考，未进入正式零分布或 p 值。

场景差异非常明显：

- `scene0207_01` 的原始方向得分为 `0.2810`，20/20 个随机 Q 都没有超过它；
- `scene0029_01` 与 `scene0675_00` 的原始方向得分为 0，20/20 个随机 Q 都超过它。

所以不存在一个对所有场景都固定更好的全局修正方向。

## 3. 当前结论（人话）

长窗口 latent 附近确实存在不少能改善相机结果的方向，短窗口可以帮助找到它们。
但证据更像“很多连续方向里，要根据当前场景挑一个方向，再决定走多远”，而不是
“天然只有左、右两个彼此分离的答案”。

原始 VRFM 方向也不是唯一特殊方向；随机旋转后经 oracle 挑选，很多时候反而更好。
这说明当前真正缺少的不是更多随机性，而是一个能根据长窗口内容选择方向和步长的
机制。

## 4. 证据边界

- 所有改善数字都是固定 10 个 calibration 场景上的 GT-oracle 上界；
- 20-Q 是方向特异性的零假设检验，不证明随机旋转本身可以作为正式训练方法；
- 当前结果不支持“已经发现若干稳定离散分支”；
- 当前结果也不等于已经有可部署 selector；
- 还没有拼接或评估完整 500 帧修正轨迹。

## 5. 下一步建议

先训练一个小型 prediction-only 候选排序器：输入完整长窗口 latent、overlap 位置、
候选 residual 和步长 `alpha`，输出该候选的预测效用。训练时从独立
`privileged_labels` sidecar 读取排序标签；推理时只生成并读取长窗口候选，不打开
sidecar。

第一轮仍按 calibration-first：

1. 1 个场景做数据 join、前后向和泄漏防火墙 smoke；
2. smoke 通过后自动使用固定 8 个 train 场景；
3. 在固定 2 个 validation 场景上与随机选择、固定原始方向和 no-op 比较；
4. 只有 held-out 排序结果显示长窗口中存在可学习信号，才扩到完整 50 场景并重训
   quality-weighted VRFM/conditional prior。

这个顺序能最快回答最关键的问题：不看短窗口和 GT，长窗口 latent 自己能不能告诉
我们“该往哪边走、走多远”。

## 6. 可追溯产物

- 正式报告：
  `/data/yjh/output/variational_camera_latent/vrfm_camera_20260827T044926Z/reports/matched_random_vs_vrfm_20q_report.json`
- 报告 SHA-256：
  `17134cec3e43730add86298b39a5463218ba158372e2953580d80129015ae695`
- 正式完成标记：
  `/data/yjh/output/variational_camera_latent/vrfm_camera_20260827T044926Z/matched_random_ablation_20q_verified_completion.json`
- 完成标记 SHA-256：
  `5ceee674da604de9a261843986d734b23f08bc46da322b9bff5bb619c28f95fa`
- 规模：20 replicates × 10 scenes，200 个 prediction-only 产物与 200 个
  privileged sidecar；完整审计和两次幂等 finalize 均通过。

