# CVA02 当前状态

更新时间：2026-08-24 13:13（下载状态快照；重新检查前不要把下面的计数当作实时值）

## 1. 权威身份

| 项目 | 固定值 |
|---|---|
| 主分支 | `codex/camera_velocity_ambiguity_02_pre_experiment` |
| 继承基线 | `015-local-global-consistency@a85bcba9356be72d00f970e948ffc461f58c95e8` |
| FastVGGT 上游 | `mystorm16/FastVGGT@6526e275a29572653a034762bb3c6c9ce280ff55` |
| H20 工作树 | `/home/ubuntu/yjh/vggt/.worktrees/camera_velocity_ambiguity_02_pre_experiment` |
| H20 数据 | `/data/yjh/share/datasets/ScanNet` |
| H20 权重 | `/data/yjh/share/pretrained/VGGT-1B/model.safetensors` |
| H20 输出 | `/data/output/camera_velocity_ambiguity/<run_id>/` |

CVA02 继承 015 的相机分支加载、100/50 sliding windows、schema 校验、
prediction-only Sim(3) 和原子产物写入，但不继承旧的 GT 重对齐评价、
sequential stitching、token disagreement 判定或“fresh holdout”语义。

## 2. 研究问题与证据边界

固定一条 full-scene global 预测 `G`，对相邻的 100-frame local windows `L/R`
分别只用预测值对齐到对应 global segment，比较共享帧上的相机中心残差。
实验要区分四种情况：

1. 同一修正方向或纯数值噪声；
2. 只有一侧有效，属于窗口选择问题；
3. 两端均有效且低代价插值连续，属于连续冗余；
4. 两端均有效但中间有独立 RGB-D 观测能量障碍，才支持多模态速度。

prediction-only 的 Sim(3)+L2/RMS 路径是凸的，不能单独识别中点势垒。
没有独立观测能量时，结论必须写成 `unidentifiable`，不能升级成物理离散分支。
前置实验不训练 V-RFM、Diffusion、Flow Matching 或 DiT。

## 3. 冻结协议

- 官方 50-scene 顺序固定；49 个场景各 500 帧，`scene0150_00` 为 430 帧；
- 449 个 length-100、stride-50 windows；
- 399 个相邻 pairs：primary shared-50 为 398，secondary shared-70 为 1；
- calibration 为 10 场景、development evaluation 为 40 场景；
- alpha 固定为 `(0, 0.25, 0.5, 0.75, 1)`；
- global-to-GT 只做一次 full-scene Sim(3)，GT 保持 raw；
- local 左右窗口只对齐到 matching global segment；
- FastVGGT pose trajectory 图严格复现上游，只做 presentation，不进入科学指标；
- calibration 完成并冻结阈值前，不得运行 development evaluation。

详细定义见 [`CVA02 冻结设计`](camera_velocity_ambiguity_02_design.md)，执行顺序见
[`CVA02 计划`](superpowers/plans/2026-08-24-camera-velocity-ambiguity-02.md)。

## 4. 当前数据与运行门控

ScanNet 下载器使用修复后的幂等上传重试逻辑：每次上传重试都会先检查服务端
final 文件，避免“服务端已 rename、客户端因 SSH 断线误判失败”。本次修复的
下载/校验回归测试共 6 项，已通过。

2026-08-26 已完成 ScanNet-50 严格校验：H20 为 50 scenes / 100 assets，
总计 37,587,327,416 bytes，`verified_completion.json` 通过本地与远端
Content-Length/SHA-256 核对。

历史下载快照（已完成，不再作为当前门控）：

```text
Downloader PID: 33292
State: 67 uploaded / 16 downloading / 17 queued
H20 files: 26 sens + 41 ply
/data: 186 GiB available
stderr: 0 bytes
verified_completion.json: completed and verified on H20
```

下载日志位于本地 `.codex_runtime/`：

```text
scannet50_resume_20260824_131151.out.log
scannet50_resume_20260824_131151.err.log
```

数据完整性门控严格按以下顺序执行：

1. 100/100 资产上传完成；
2. 官方清单、Content-Length、本地文件和 H20 SHA-256 逐项一致；
3. 生成并检查 `verified_completion.json`；
4. 重新确认 H20 身份、磁盘、GPU、环境和工作树；
5. 才允许正式 GPU 推理。

校验失败或 `/data` 空间不足时，停止推进 GPU；不得删除其他用户数据，也不得把
大型产物拉回本地。

## 5. 实施状态与责任边界

| 部分 | 责任 | 当前状态 |
|---|---|---|
| CVA02 现象验证 | 500/100/50 prediction、冻结 oracle、RGB-D 与分类 | 10-scene calibration 已完成 |
| VRFM Phase 1 数据 | long-only inference、左右短窗等权 teacher、固定 segment-z | 实现完成，待 H20 smoke/calibration |
| Prediction-only 数据 | source shards、VRFM raw candidates、deterministic baseline | 独立目录与 SHA-256 manifest |
| Privileged sidecar | GT pose、冻结 Sim(3)、candidate error | 物理隔离，仅 sample ID 关联 |
| H20 集成 | 1-scene smoke 自动扩到 10 scenes、最终 verify | runner 已实现 |

VRFM Phase 1 输出固定为 `/data/yjh/output/variational_camera_latent/<run_id>/`。
它只修正独立 50-frame overlap，不在本阶段拼接完整 500-frame 轨迹；弱信号仍是
技术成功，不能把 two-means 结果直接宣称为离散多模态。

## 6. 如何更新本页

本页是唯一的当前进度来源。每次有实质进展时更新：

1. 时间戳和当前分支 HEAD；
2. ScanNet 状态、H20 文件数和 `/data` 可用空间；
3. 已通过的测试或完整性门；
4. 下一项可执行任务和阻塞原因。

历史 `doc/`、`log/` 和 `results/` 文件只记录当时运行，不修改成实时状态。
