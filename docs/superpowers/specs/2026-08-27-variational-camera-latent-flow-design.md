# 长窗口 Camera Latent 的 Variational Rectified Flow Matching 设计

状态：已获用户确认，待实施计划

日期：2026-08-27

分支：`codex/camera_velocity_ambiguity_02_pre_experiment`

正式输出：`/data/yjh/output/variational_camera_latent/<run_id>/`

## 1. 背景与目标

现有 CVA02 前置实验比较了完整 500 帧 global VGGT Camera prediction 与
length-100、stride-50 的 local predictions。10 个 calibration 场景的结果支持
“短窗口经常提供有用修正”，但尚未严格证明存在稳定的离散多模态；观察更接近
窗口选择问题或连续修正。

本实验不预设离散分支已经成立，而是探索性验证 Variational Rectified Flow
Matching（VRFM）能否从**仅有长窗口 latent 的输入**中，生成多样、可解码且
部分有效的短窗口式 latent 修正。参考方法为 Guo 与 Schwing 的
《Variational Rectified Flow Matching》：
<https://proceedings.mlr.press/v267/guo25i.html>。

用户的最终约束是：

- 短窗口只允许在离线训练数据构造阶段运行；
- 正式推理只能加载一次长窗口，不能依赖短窗口图像或 latent；
- flow 直接工作在官方 VGGT Camera Head 的 normalized camera token 空间；
- 主数据完全 prediction-only；GT 位姿、深度和误差标签放在独立
  `privileged_labels` sidecar，通过 sample ID 关联；
- 第一阶段先拿到可分析、可继续训练的 latent 候选，不要求立即生成一条完整
  500 帧最终轨迹。

## 2. 非目标

第一阶段明确不做以下工作：

- 不修改或微调 VGGT、Aggregator 或 Camera Head；
- 不让正式推理读取短窗口、GT pose、GT depth 或 oracle 标签；
- 不拼接八个 overlap 的结果为完整 500 帧修正轨迹；
- 不自动选择唯一最终分支；
- 不把 latent 多样性自动解释为真实离散相机解；
- 不因信号较弱而删除原始候选或中止数据导出。

## 3. 数据单位与张量定义

官方 Camera Head 为每帧产生一个 2048 维 normalized camera token。对普通
ScanNet 场景：

```text
G:       完整长窗口 camera tokens，[500, 2048]
S_i:     第 i 个短窗口 camera tokens，[100, 2048]
i:       0..8，窗口起点为 0, 50, ..., 400
```

相邻短窗口 `S_i` 与 `S_{i+1}` 共享 50 个真实帧。对第 i 个 primary overlap：

```text
G_i:     G 中对应共享帧的切片，[50, 2048]
L_i:     S_i 的后 50 帧，[50, 2048]
R_i:     S_{i+1} 的前 50 帧，[50, 2048]
```

`G_i` 是两条训练配对完全相同的 flow 起点，`L_i` 和 `R_i` 是两个等权教师
终点：

```text
(G, span_i, G_i -> L_i)
(G, span_i, G_i -> R_i)
```

完整 `G` 始终作为全局条件；flow 状态和监督只覆盖共享 50 帧。这样能够严格
测试“同一个长 latent 起点是否对应多个短窗口修正方向”。原始的九组
`G[start:start+100] -> S_i` 同时保存在预测数据中，供后续 100 帧或完整轨迹
模型复用，但不作为第一阶段 VRFM 的主训练单位。

Camera tokens 只按 scene/frame identity 对齐，不在 token 空间施加 Sim(3) 或
其他几何变换。global/local gauge 的 prediction-only 对齐只作用于冻结 Camera
Head 解码后的相机结果。把不同窗口产生的 normalized Camera tokens 放在同一条
线性 flow 路径上是本实验要验证的假设，而不是预先保证成立的事实。

只使用协议中的 primary shared-50 overlaps。非 50 帧 overlap 可保留为描述性
数据，但不能混入主训练或主评价。

## 4. Prediction-only 数据构造

每个场景只运行一次 500 帧 global VGGT，再离线运行九个 100 帧 local VGGT。
两类推理使用同一 checkpoint、Camera iterations、图像预处理、frame IDs 和
provenance 约束。

每个 scene shard 使用无 object dtype 的原子 `.npz`，并通过 JSON manifest
记录相对路径、角色、样本数、SHA-256、schema version、代码 commit、checkpoint
digest 和协议 digest。数值数组必须 finite，frame IDs 必须严格递增，left/right
overlap 必须逐帧与 `G_i` 对齐。

正式训练前先对每个 smoke overlap 执行 latent compatibility preflight：解码
`G_i`、`L_i`、`R_i` 三个端点，并分别检查两条线性插值路径在
`t={0,0.25,0.5,0.75,1}` 上是否 finite。中间路径质量较弱只进入报告，不作为
技术失败；端点无法由冻结 Camera Head 稳定复现或出现非有限值时才 fail closed。

Prediction-only scene shard 至少包含：

- `global_camera_tokens [N, 2048]`；
- `short_camera_tokens [W, 100, 2048]`；
- `short_starts [W]` 与 `short_frame_ids [W, 100]`；
- `overlap_long_tokens [P, 50, 2048]`；
- `overlap_left_tokens [P, 50, 2048]`；
- `overlap_right_tokens [P, 50, 2048]`；
- `overlap_frame_ids [P, 50]`、`overlap_starts [P]` 和稳定 sample IDs；
- global/local 的 raw 9D camera predictions 与 activated predictions；
- prediction-only alignment 所需的最小相机字段。

本实验使用新的 schema 和 study type，不修改 021 数据构造分支已有的
`full_hidden_sequence_refiner` schema。短窗口 Camera tokens 在这里是
prediction-only 教师终点，不是 GT latent，因此必须显式命名为
`short_camera_tokens`，不得使用含混的 `short_hidden`。

## 5. VRFM 模型

### 5.1 Flow 路径

对等概率采样的教师终点 `X_1 in {L_i, R_i}`：

```text
X_0 = G_i
X_t = (1 - t) X_0 + t X_1
U   = X_1 - X_0
```

velocity network 学习：

```text
v_theta(X_t, t, z, context=G, span=span_i) -> U_hat [50, 2048]
```

flow 状态始终位于真实 2048 维 Camera Head token 空间。网络内部允许用线性
adapter 投影到较低的工作维度，但最终速度必须回到 `[50, 2048]`，不能把 PCA
或 pose residual 空间冒充为 Camera latent flow。

### 5.2 全局条件编码

完整长窗口 `G` 先通过共享的 `2048 -> 256` 投影。velocity backbone 使用四个
宽度 256、八头的 residual Transformer blocks；每个 block 对 50 帧 flow state
执行 self-attention，并以 projected `G` 为 key/value 执行 cross-attention。
overlap start 位置使用可学习 span embedding，time `t` 与 latent `z` 使用 MLP
embedding 加入 flow queries。输出 adapter 为 `256 -> 2048`。

该设计让模型训练和推理都能看到完整长窗口上下文，同时把昂贵的 flow 状态限制
在当前 50 帧区域。完整 500 帧 token 不作为生成目标。

### 5.3 变分变量 z

训练期 recognition network `q_phi` 读取 projected `G`、`X_0`、所选教师
`X_1` 及 span embedding，输出 16 维 diagonal Gaussian 的 `mu` 和
`log_var`。通过 reparameterization 采样一个 `z`。

同一个 `z` 在全部 50 帧和整条 ODE 积分路径上保持不变；禁止逐帧重新采样，
避免相机轨迹在分支之间抖动。prior 固定为 `p(z)=N(0,I)`。正式推理删除
recognition network，直接从 prior 采样 `z`，所以不需要短窗口。

### 5.4 训练目标

主目标为等权 VRFM loss：

```text
L = mean_squared_error(U_hat, U) + beta * KL(q_phi(z) || N(0, I))
```

第一版不使用 GT 质量加权，不按左右教师优劣重新采样，也不因 oracle 标签丢弃
某一支。`beta`、优化器和训练步数必须写入冻结 run config；初始 `beta` 为
`1e-4`，前 20% steps 线性 warm-up。训练同时记录 posterior variance、不同 z
的输出方差和 KL，便于发现 posterior collapse。

### 5.5 必要对照

训练一个结构和容量尽量一致、但移除 `z` 与 recognition network 的
deterministic RFM。它使用完全相同的数据、flow interpolation、全局条件和优化
预算。对照用于描述普通 MSE 是否趋向平均方向；第一阶段不把显著优于该对照设为
硬性继续门槛。

## 6. 推理与候选导出

正式推理输入只有完整长窗口 Camera tokens `G`。对每个 primary overlap：

1. 取 `X_0=G_i`；
2. 从 `N(0,I)` 独立采样 32 个 `z`；
3. 每个 z 在整条 flow 上固定，使用 16-step Heun integration；
4. 得到 32 个 `[50, 2048]` corrected Camera latent candidates；
5. 使用冻结的官方 `CameraHead.decode_pose_tokens` 解码为 raw/activated 9D camera；
6. 保存全部原始候选，并分别在 latent 空间与解码相机空间生成聚类摘要。

第一阶段不在八个 overlaps 之间联合选支，也不拼接完整 500 帧。原始候选的保存
优先于聚类结果；聚类算法或阈值改变时不得重新运行 VGGT/VRFM 才能重做分析。

## 7. 输出契约

正式 run 目录至少包含：

```text
/data/yjh/output/variational_camera_latent/<run_id>/
  run_metadata.json
  prediction_only/
    manifest.json
    source_shards/<scene>.npz
    vrfm_candidates/<scene>.npz
    deterministic_candidates/<scene>.npz
    cluster_summaries/<scene>.json
  privileged_labels/
    manifest.json
    <scene>.npz
  reports/
    smoke_summary.json
    calibration_summary.json
```

VRFM candidate shard 至少包含 sample IDs、`z [P,32,16]`、corrected tokens
`[P,32,50,2048]`、decoded cameras、latent cluster IDs 和 camera cluster IDs。
所有训练用 token 与候选以 float32 保存，避免把存储量优化与科学信号混在一起。

`privileged_labels` sidecar 通过相同 sample IDs 关联，包含 raw GT c2w/depth
引用、每个教师和候选的对齐后相机误差、相对 global baseline 的改进幅度和有效性
标签。Prediction-only loader 的函数签名不得接收 privileged root；正式推理和
候选生成不得打开 sidecar。

这批输出能够继续训练：

- 仅依赖长窗口 latent 的多候选 VRFM refiner；
- 对 VRFM 候选进行排序的 prediction-only selector；
- candidate confidence/uncertainty model；
- 后续覆盖 100 帧或完整 500 帧的 camera latent refiner。

## 8. 运行顺序与数据角色

运行只使用 H20，开始前重新核对 remote identity、GPU、磁盘、活跃任务、代码
worktree、checkpoint 和 ScanNet manifest。不得使用旧 H20 Hugging Face token，
也不得把大型 tensor 输出拉回本地。

执行顺序为：

1. 一个 calibration 场景的 one-scene overfit smoke，验证 schema、前后向、
   z 使用、ODE、Camera Head 解码和原子输出；
2. smoke 无技术错误后，自动扩到固定的 10 个 calibration 场景；
3. 10 场景按既定顺序固定为前 8 个 train、后 2 个 validation，禁止 scene
   leakage；
4. 对 train 和 validation 都导出候选，但 manifest 明确标记数据角色；
5. 生成探索性报告，不自动进入完整 500 帧训练。

Smoke 允许有意过拟合，只证明实现能表达左右教师方向。Validation 用来观察
跨场景信号，但第一阶段不要求统计显著性。

## 9. 探索性评价

每个 validation overlap 对同一 `G` 和 span 采样多组 z，同时检查 latent 与冻结
Camera Head 解码结果。评价记录：

- 不同 z 的 latent 方差、pairwise distance 和聚类稳定性；
- 不同 z 解码后的相机轨迹差异与有限性；
- 候选对左右教师方向的覆盖；
- best-of-32 候选相对 global baseline 的 GT sidecar 改进；
- VRFM 与 deterministic RFM 的描述性差异；
- KL、posterior variance 和 z-output sensitivity。

报告分为三个探索性等级：

- `PROMISING`：z 产生多样、可解码且至少部分改善的修正；
- `WEAK_SIGNAL`：z 产生变化，但改善不稳定或主要是连续冗余；
- `NO_SIGNAL`：模型忽略 z，或变化主要表现为无效噪声。

第一阶段不要求形成严格的两个簇、不要求左右教师都 oracle-valid、不要求存在插值
能量障碍，也不要求统计显著优于 deterministic RFM。以上项目仍完整记录，供后续
决定是否收紧门槛。

## 10. 技术门控与恢复

以下情况必须 fail closed：

- global/local frame IDs 或 overlap tensor 无法逐帧对齐；
- 输入、velocity、corrected token 或 decoded camera 出现 NaN/Inf；
- 冻结 Camera Head 无法复现教师 token 对应的 endpoint camera；
- checkpoint、代码 commit、schema、protocol 或 manifest digest 不一致；
- prediction-only 路径读取了 GT/privileged 文件；
- H20 身份、GPU、磁盘或目标 worktree 不符合运行配置；
- 原子 shard、manifest 或 SHA-256 校验失败。

弱科学信号不是技术失败。所有完成的 source shards、checkpoints、候选 shards 和
sidecars 都按 sample ID 幂等恢复；不覆盖 provenance 不同的既有 run，也不删除
原始候选。训练 checkpoint、optimizer、scheduler、random seeds 和已完成 scene
记录必须支持 scene/step 级恢复。

## 11. 测试与验收

CPU 单元测试必须覆盖：

- 500/100/50 张量与 frame ID 对齐；
- 同一 `G_i` 精确生成 left/right 两条等权训练记录；
- prediction-only 与 privileged schema 物理隔离；
- recognition posterior、velocity network 和 deterministic baseline 的 shape；
- 一个 z 在 50 帧和全部 ODE steps 中保持不变；
- atomic write、manifest digest、resume 和 provenance mismatch 拒绝；
- 聚类和报告可只依赖已保存候选重跑。

H20 smoke 验收只要求：

- one-scene 数据构造完成且 schema/digest 通过；
- VRFM 能在有限 loss 下完成训练和恢复；
- 改变 z 后至少能产生 finite latent/camera 候选；
- 冻结 Camera Head endpoint 和候选解码路径可用；
- 原始候选、聚类摘要与 sidecar 全部成功写入正式输出目录。

Smoke 通过后自动运行 10-scene calibration。最终交付是完整 manifest、训练日志、
VRFM/deterministic checkpoints、prediction-only latent 候选、独立 privileged
sidecar 和探索性总结；不以获得特定科学结论作为交付前提。
