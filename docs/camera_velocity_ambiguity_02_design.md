# Camera Velocity Ambiguity 02：前置实验设计

状态：冻结设计，尚未运行正式推理

日期：2026-08-24

分支：`codex/camera_velocity_ambiguity_02_pre_experiment`
基线：`015-local-global-consistency@a85bcba9356be72d00f970e948ffc461f58c95e8`

## 1. 当前真实状态

本实验继承 015 的代码与历史，但使用全新的命名空间、配置、输出和结论，
不延续旧实验编号，也不修改旧结果。

- H20 隔离工作树：
  `/home/ubuntu/yjh/vggt/.worktrees/camera_velocity_ambiguity_02_pre_experiment`
- 代码根目录仍属于 `/home/ubuntu/yjh/vggt`。
- 正式输出固定为：
  `/data/output/camera_velocity_ambiguity/<run_id>/`
- 官方 VGGT-1B 权重已存在：
  `/data/yjh/share/pretrained/VGGT-1B/model.safetensors`。
- 仓库中只有 4 个冻结的 500-frame global diagnostics：
  `scene0000_00`、`scene0013_02`、`scene0029_01`、`scene0691_00`。
- 仓库中的旧 local/global 标量结果不能替代 50 场景逐窗口输入。
- 完整 50 场景 global/local 逐帧预测尚未被验证为可用，因此 pipeline 必须
  支持在数据完整后补齐，而不能假设 README 中的结果真实存在。
- ScanNet 下载当前为 60/100 已上传，H20 侧为 24 个 `.sens` 和 36 个 PLY；
  仍有 16 个可断点续传的 `.partial`。
- `/data` 已满，下载器已安全停止；`verified_completion.json` 尚不存在。
- 基线 CPU 测试为 90/90 通过。数据校验通过前，不允许启动 GPU。

## 2. 研究问题

固定一条长序列的 global VGGT 相机预测 `G`。相邻两个 100-frame、
stride-50 的短窗口 `L` 和 `R` 会在共享区域给出两份相机预测。分别只用
预测值把 `L`、`R` 对齐到对应 global gauge 后，定义共享帧上的相机中心残差：

```text
d_L = C_L_aligned - C_G
d_R = C_R_aligned - C_G
```

本实验判断这两个修复方向是：

1. 同一方向或仅有数值噪声；
2. 只有一个方向有效，即窗口选择问题；
3. 两端不同且均有效，并由低代价插值连续连接，即连续冗余；
4. 两端均有效，但中间路径出现独立观测能量障碍，即支持多模态速度。

实验不训练 V-RFM、Diffusion、DiT 或其他生成模型。只有第 4 类在跨场景
稳定成立时，才支持进入 V-RFM 训练；即使如此，也不等同于证明真实相机轨迹
存在断开的物理解集合。

## 3. 与 015 和 FastVGGT 的边界

### 3.1 从 015 继承什么

直接复用以下稳定能力：

- 官方 VGGT 相机分支的加载和 camera-only 推理方式；
- `build_sliding_windows` 的 100/50 窗口定义；
- global/local NPZ 的严格 finite/schema 校验思想；
- prediction-only Sim(3) 对齐；
- 原子 JSON/NPZ 写入、run provenance、恢复执行；
- scene-level bootstrap，固定 seed 33、10,000 次。

以下旧语义不得直接复用：

- 每个 local window 各自重新对齐 GT 的评价；
- 把 token disagreement 当作候选有效性；
- sequential stitching 或按边界距离选一个窗口后丢弃 pair identity；
- 旧 calibration/holdout 的“fresh holdout”称呼。

### 3.2 从 FastVGGT 继承什么

固定参考 `mystorm16/FastVGGT@6526e275a29572653a034762bb3c6c9ce280ff55`。

尽量少改动地复用其 ScanNet 评测控制流：

- 官方 50-scene 顺序；
- 有效图像 ID 与有限 GT pose ID 的交集；
- 保留第一帧、随后按 floor stride 取样的 frame selection；
- VGGT `crop` 预处理：宽度 518，高度按比例并对齐 patch 约束；
- scene 级恢复执行、显式输出目录和 metadata。

本实验仍使用当前仓库的官方 VGGT-1B Camera Head，不启用 FastVGGT token
merging，也不改 `vggt/`。

FastVGGT 的 `umeyama_alignment` 和 `eval_trajectory` 将逐字 vendor，专门生成
官方风格 pose trajectory 图。其输入/坐标解释存在已知语义问题，所以：

- 原函数一字不改；
- 图像不得后处理；
- 输出放入 `visualizations/reproduction_only/`；
- sidecar 固定 `eligible_for_primary_metrics=false`；
- 其 ATE/ARE/RPE 永远不能进入阈值、事件分类或最终结论。

## 4. 冻结数据协议

### 4.1 场景与帧

- 场景：`configs/fastvggt_scannet50.txt` 的 50 个场景，顺序固定。
- 普通场景：选择 500 帧。
- `scene0150_00`：仅有 430 个有效帧，全部保留。
- 每个场景先生成一次 global prediction。
- 然后以 length=100、stride=50 生成 local windows。

协议机械计数：

| cohort | scenes | global runs | local windows | primary shared-50 | secondary shared-70 |
|---|---:|---:|---:|---:|---:|
| calibration | 10 | 10 | 90 | 80 | 0 |
| development | 40 | 40 | 359 | 318 | 1 |
| total | 50 | 50 | 449 | 398 | 1 |

`scene0150_00` 的最后一对窗口共享 70 帧，只进入 secondary 描述，不能进入
阈值、主 prevalence 或 bootstrap。

### 4.2 split v2

旧 split v1 的成员和顺序保持不变：

```text
69c283245c4f220965e6fde3b96192de298e292eb8ca625c94851fe8932cdb8a
```

新配置不能伪装成旧 source run。v2 必须记录：

- v1 文件摘要和父 split digest；
- 新 frame-selection 协议；
- 50/449/398/1 机械计数；
- calibration 10 scenes 与 development 40 scenes；
- 本设计和实现配置的 digest。

旧 40 场景已经被历史研究查看过，统一称为 `development_evaluation`。

### 4.3 数据完整性门控

GPU runner 启动前必须同时满足：

1. ScanNet 官方 50-scene 清单精确匹配；
2. 100/100 文件上传完成；
3. 每个官方 URL 的 Content-Length 与本地文件一致；
4. 本地 SHA-256 与 H20 文件 SHA-256 一致；
5. H20 生成并通过 `verified_completion.json`；
6. 每场景能提取有效 color、pose、depth 和 calibration；
7. 权重文件存在且 provenance 写入 run metadata；
8. `/data` 有足够输入和输出余量；
9. GPU、已有进程和目标 worktree 再次核验。

任一条件失败均 fail closed，不允许 scene skip 后继续形成“完整结果”。

## 5. 预测层

预测层只负责生成 raw model outputs，不做科学结论。

### 5.1 global prediction

每个场景使用完整 500/430 selected frames，保存：

- 精确 frame IDs；
- raw predicted `c2w`；
- normalized Camera Tokens；
- 推理耗时和 peak CUDA memory；
- checkpoint identity、commit、selection/preprocess protocol digest。

### 5.2 local prediction

每个窗口使用同一场景 global frame list 的连续位置切片，保存相同最小字段，
并记录 window index/start/stop/frame IDs。支持：

- scene shard；
- 完整 artifact + completion sidecar 的幂等 resume；
- schema 或 provenance 不匹配时拒绝复用；
- 不完整 `.tmp` 不视为完成。

候选构造函数的签名不得接收 GT。GT 仅由独立 oracle 层读取。

## 6. 几何与评价

### 6.1 local 到 global

`L` 和 `R` 各自使用完整 100 帧、prediction-only 地对齐到对应 global segment。
严禁：

- 用 shared-50 单独拟合；
- `L` 直接对齐 `R`；
- 使用 GT；
- 对齐退化后仍保留样本。

对齐需要记录 scale、rotation determinant、rank、condition diagnostic、RMS
residual 和有效标志。

### 6.2 唯一 global 到 GT

每场景只允许用完整 500/430 global trajectory 与 raw GT 拟合一次 Sim(3)。
该对象绑定 scene、full frame-ID digest、fit count 和 transform digest，并冻结后
用于 global baseline、L、R 和全部 alpha。评价期间禁止重新拟合。

GT 是 `PRIVILEGED_GT` 离线诊断，不可进入窗口选择或候选构造。

### 6.3 修复方向和插值

对 shared frames 计算：

- flattened residual cosine；
- scene-scale-normalized RMS separation；
- per-frame direction agreement；
- left/right residual magnitude；
- 两个 prediction-only alignment residual；
- boundary distance、有效性和排除原因。

固定：

```text
alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
C(alpha) = C_G + (1-alpha) d_L + alpha d_R
```

rotation 和 FoV 始终复制 global；不得在 raw 9D pose encoding 中插值。

### 6.4 凸性限制

在冻结 affine Sim(3) 与 translation L2/RMS 下，`C(alpha)` 的误差是凸的。
因此 translation-only oracle 不可能合法地产生“内部严格差于两个端点”的障碍。

实现必须把这一点作为运行时断言：若 translation-only 曲线违反凸性，结论不是
第 4 类，而是实现或数值协议错误。

translation-only 层只能支持：

- `NOT_SUPPORTED`；
- `SELECTOR_PROBLEM`；
- `CONTINUOUS_REDUNDANCY`。

## 7. RGB-D 非凸观测门

`MULTIMODAL_VELOCITY_SUPPORTED` 只能由与 GT pose 独立的 RGB-D/reprojection/
occlusion energy 识别。该层：

- 只接收 RGB、depth、sensor intrinsics 和 candidate poses；
- 不允许传入 GT pose；
- scene scale 只在完整 global observation 上冻结一次；
- 固定 pixel grid、edge pairs、depth range、occlusion/free-space/coverage penalty；
- correspondence 不足、scale 落边界或能量曲线过平时返回 `INVALID`；
- calibration 冻结 barrier margin，development 禁止调参。

若第一轮尚未实现或真实数据 gate 无效，程序必须报告：

```text
MULTIMODAL_VELOCITY_SUPPORTED = UNIDENTIFIABLE_WITH_TRANSLATION_ONLY
```

不能把“没有证据”写成“已否定多模态”。

## 8. 证据防火墙与事件分类

所有指标必须携带来源：

```text
PREDICTION_ONLY
PRIVILEGED_GT
OBSERVATION_RGBD
PRESENTATION_ONLY
```

- direction/separation：`PREDICTION_ONLY`；
- endpoint validity：允许 `PRIVILEGED_GT`，但只能离线诊断；
- interior barrier：必须是有效 `OBSERVATION_RGBD`；
- FastVGGT plot metrics：只能是 `PRESENTATION_ONLY`。

事件分类 fail closed：未知字段或 presentation-only 字段进入 decision API 时直接
报错。第四类必须同时满足方向分离、双端有效、alignment 有效、RGB-D gate
有效、内部 barrier 超过冻结 margin、连续时间支持和 scene prevalence 门。

## 9. 负控制

正式分析至少包含：

1. 同一窗口与自身比较；
2. 纯 global gauge copy；
3. 匹配的随机错误窗口；
4. 残差取反；
5. 小数值扰动；
6. 对齐退化轨迹。

控制组必须经过与真实 pair 相同的 pipeline，并单独报告通过率，不能只展示图。

## 10. Calibration、冻结与 development

执行状态只能单向变化：

```text
INPUTS_VERIFIED -> CALIBRATION_COMPLETE -> POLICY_FROZEN
                -> DEVELOPMENT_COMPLETE -> DECISION_COMPLETE
```

- calibration 必须精确包含 10 scenes、80 primary pairs；
- frozen policy 绑定 config、split-v2、input manifest、commit 和 calibration rows；
- 已冻结 policy 不可覆盖；
- development CLI 不接受 threshold override，也不能调用 fitter；
- development 精确包含 40 scenes、318 primary pairs；
- 统计先聚合为 scene rows，再以 scene 为单位 bootstrap；
- shared-70 只进入 secondary 文件。

## 11. 输出与图

数值输出分层保存：

```text
<run_id>/
  manifests/
  predictions/global/
  predictions/local/
  calibration/
  frozen_policy/
  development/
  controls/
  visualizations/primary/
  visualizations/reproduction_only/
  decision/
```

至少生成：

1. left/right residual direction similarity；
2. interpolation energy curve；
3. 四类事件的 scene-level prevalence；
4. FastVGGT 原样 trajectory plot（reproduction-only）。

大型 NPZ、图像、数据集和 checkpoint 不提交到 Git。只允许提交代码、测试、
冻结配置、标量 CSV/JSON、小图和最终报告。

## 12. 三人分工

### Part A：数据、协议与预测 pipeline

负责 split-v2、数据完整性门控、FastVGGT frame selection、global/local camera-only
推理、artifact schema、resume/shard 和 H20 runner 的推理阶段。

退出条件：CPU fake-model 测试通过；真实数据验证前 runner 必须拒绝 GPU；计数严格
为 50/449/398/1。

### Part B：科学几何、oracle 与判定

负责 prediction-only alignment、唯一 frozen Sim(3)、residual/interpolation、凸性
断言、RGB-D gate、控制组、事件分类、冻结状态机和 scene bootstrap。

退出条件：合成轨迹覆盖四类事件；translation-only 永远不能产生第四类；GT、RGB-D
和 presentation metric 防火墙测试通过。

### Part C：FastVGGT 复现、可视化与报告

负责逐字 vendoring 两个上游函数、hash/AST parity、reproduction adapter、三张主图、
上游 trajectory 图、标量 exporter、README 和最终 GO/NO-GO 报告模板。

退出条件：vendored 函数 hash 固定；图指标无法进入决策；删除所有 PNG 不改变任何
数值结论。

三部分按接口顺序 A -> B -> C 集成；每部分先测试后实现，并由非实现者复核。

## 13. 最终结论约束

最终只能输出：

- `NOT_SUPPORTED`
- `SELECTOR_PROBLEM`
- `CONTINUOUS_REDUNDANCY`
- `MULTIMODAL_VELOCITY_SUPPORTED`

如果 RGB-D gate 缺失或无效，第四项保持不可识别。最终报告同时给出是否进入
V-RFM 的 `GO / NO-GO`，逐条列出支持证据、负控制和协议限制。
