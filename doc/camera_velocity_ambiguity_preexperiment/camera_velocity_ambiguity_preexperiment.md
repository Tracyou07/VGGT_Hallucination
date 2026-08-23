# 多个短上下文 VGGT 预测是否产生多模态相机修复速度？

## Do Overlapping Short-Context VGGT Predictions Induce Multimodal Camera-Repair Velocities?

> 文档角色：三文档中的详细前置实验协议与结果报告
>
> 状态：`protocol in progress; no scientific conclusion yet`
>
> 日期：2026-08-23

---

## 0. 一页读懂这个实验

VGGT 对一条 500 帧序列整体预测时，可能在长距离上出现漂移。把同一序列切成九个
长度 100、stride 50 的短窗口后，每个短窗口看到的上下文更局部。相邻两个窗口在
共享 50 帧上会给出两份修复建议：左窗口残差 $d_L$ 和右窗口残差 $d_R$。

这并不自动等于“多解”。实验依次问四个问题：

1. 两个局部结果是否都能可靠对齐到 global prediction？
2. $d_L$、$d_R$ 各自应用后，是否都比原 global trajectory 更接近 GT？
3. 两个有效方向是否真的分离，而不只是数值噪声或 gauge copy？
4. 二者的均值或内部插值是否比两个端点明显更差？

四种答案对应四种工程路线：

| 标签 | 通俗含义 | 下一步 |
|---|---|---|
| `NOT_SUPPORTED` | 没看到稳定的多候选有效现象 | 改候选、数据或评价 |
| `SELECTOR_PROBLEM` | 往往只有一个窗口建议是好的 | 训练/设计 selector |
| `CONTINUOUS_REDUNDANCY` | 两端和中间都好，可以连续融合 | 确定性融合优先 |
| `MULTIMODAL_VELOCITY_SUPPORTED` | 两端都好且方向分开，但平均/内部更差 | 才进入 V-RFM |

本实验中的 GT 是 privileged offline diagnosis，只用于回答科学问题。正式推理阶段
不能依赖 GT 选择候选。

---

## 1. 研究问题与可证伪假设

### 1.1 研究问题

对同一条 500 帧 global VGGT camera trajectory，多个重叠的 100 帧 local VGGT
predictions 是否会在共享帧上产生多个有效、方向分离、不可安全平均的 camera-center
repair velocities？

这里的“多模态”限定在**修复速度**，不指多个物理世界，也不指 raw window outputs
形成多个视觉簇。

### 1.2 五级假设链

实验不做一个笼统的 yes/no，而按以下链条逐级验证：

- `H0 / eligibility`：local-to-global prediction-only alignment 数值可靠；
- `H1 / endpoint validity`：$d_L$、$d_R$ 至少有一个能改善 global prediction；
- `H2 / double validity`：两个端点都通过冻结有效性门槛；
- `H3 / directional separation`：两个有效残差按冻结指标显著分离；
- `H4 / unsafe interpolation`：均值或预注册内部插值显著差于两个有效端点；
- `H5 / prevalence`：上述事件不是少数 scene 偶然，而在 scene-level bootstrap 下稳定。

只有 `H0` 到 `H5` 全部成立，结果才进入 `MULTIMODAL_VELOCITY_SUPPORTED`。
任何一级失败都必须保留样本和原因，不能只汇报通过案例。

### 1.3 实验不回答什么

- 不证明固定完整观测下存在多个物理真轨迹；
- 不证明多个窗口输出来自真实 posterior；
- 不证明 V-RFM 一定优于 Diffusion 或 deterministic refiner；
- 不选择 V-RFM 网络宽度、层数或优化器；
- 不把 GT 用作部署时 selector。

---

## 2. 当前 provenance 与证据状态

### 2.1 已提交且可核验的事实

2026-08-23 对 H20 主项目 `/home/ubuntu/yjh/vggt` 做了只读检查：

| 字段 | 已核验内容 |
|---|---|
| host / user | `VM-0-11-ubuntu` / `ubuntu` |
| 主 worktree | `codex/camera_solution_space_01_theory_foundation@cc1d8ac`，clean |
| 实验 worktree | `codex/camera_solution_space_01_stage1@dcc55a2`，clean |
| Stage 1 范围 | 固定观测相机解空间、SE(3) geometry 与 eligibility diagnostics |
| V0 结果卡 | `PENDING_COMMITTED_RESULT_CARD` |
| V0 run ID / artifact | `PENDING_COMMITTED_RESULT_CARD` |

`camera_solution_space_01_stage1@dcc55a2` 可以提供几何合法性、对齐和 eligibility 的
实现经验，但它不是 500/100 重叠窗口速度实验，不能作为本报告的正面或负面结果。

### 2.2 什么信息才允许进入结果章节

结果只能来自一个已提交、不可歧义定位的 result card。最低字段如下：

```yaml
experiment_branch: ...
code_commit: ...
input_manifest_path: ...
input_manifest_sha256: ...
dataset_and_split: ...
run_id: ...
command: ...
config_path: ...
artifact_root: ...
artifact_manifest_sha256: ...
scene_count: ...
eligible_pair_count: ...
decision_counts: ...
scene_bootstrap_seed: ...
```

若任何字段缺失，文档可以说明“执行中”或“provenance 不完整”，但不能填科学结论。

### 2.3 旧 40 scenes 的地位

任何已经看过、调过阈值或用于发现问题的 40 scenes 只能标为
`development evaluation`。它们可以用于 debug、图形模板和阈值校准，不能与未见过的
evaluation scenes 混合后声称无偏 prevalence。

---

## 3. 输入冻结协议

### 3.1 Scene manifest

每个 scene 必须在 manifest 中显式记录：

- dataset、split、scene ID；
- 500 个有序 frame IDs，而不是只写 start/end；
- 每帧图像路径、内容 hash、原始尺寸和时间戳；
- GT pose 路径、合法性 mask 与缺失原因；
- resize/crop/normalize/color-space 版本；
- VGGT weights 标识与 hash；
- inference dtype、batching、随机种子和代码 commit。

若 scene 不足 500 帧、frame identity 不连续或 GT 有 sentinel/invalid pose，必须按冻结
eligibility 规则处理，不能在看完指标后临时删帧。

### 3.2 窗口冻结

对每条 500 帧序列固定九个窗口：

```text
W0 = frames[  0:100]   W1 = frames[ 50:150]
W2 = frames[100:200]   W3 = frames[150:250]
W4 = frames[200:300]   W5 = frames[250:350]
W6 = frames[300:400]   W7 = frames[350:450]
W8 = frames[400:500]
```

由此得到八个相邻窗口 pair，每个 pair 有恰好 50 个共享 frame IDs。实验单位是：

```text
(scene_id, left_window_id, right_window_id, shared_frame_ids[50])
```

统计时不能把同一 scene 的八个 pair 当成八个独立 scene。

### 3.3 VGGT 推理冻结

同一 scene 的 global pass 和九个 local passes 必须共享：

- 同一 weights 与 camera head；
- 同一图像预处理和分辨率；
- 同一 pose representation 解码；
- 同一有效帧规则；
- 同一确定性/随机设置。

global pass 一次输入完整 500 帧；每个 local pass 只输入对应 100 帧。不能把 global
features 偷送进 local pass，否则不再测“短上下文产生的候选”。

---

## 4. 预测值对齐与候选构造

### 4.1 为什么必须先对齐

global prediction 和每个 local prediction 都可能采用不同全局平移、旋转和尺度。
直接相减会把 gauge difference 当作 repair。候选构造必须完全使用预测值，不看 GT。

### 4.2 Local-to-global prediction-only Sim(3)

记 global centers 为 $c_i^G$，窗口 $W_k$ 的 local centers 为 $c_i^{(k)}$。在窗口覆盖
的 100 个共同 frame IDs 上，拟合

$$
A_k^*=\operatorname*{arg\,min}_{A\in\mathrm{Sim}(3)}
\sum_{i\in W_k}w_i\left\|A(c_i^{(k)})-c_i^G\right\|_2^2.
$$

权重、robust loss、最小有效帧数和退化判据全部由 calibration 冻结。得到

$$
\widetilde c_i^{(k)}=A_k^*(c_i^{(k)}),\qquad
d_k(i)=\widetilde c_i^{(k)}-c_i^G.
$$

对于相邻 pair $(W_j,W_{j+1})$ 的 50 个共享帧：

$$
d_L(i)=d_j(i),\qquad d_R(i)=d_{j+1}(i).
$$

### 4.3 对齐 eligibility

每个窗口至少记录：

- 有效 camera center 数；
- centered point cloud rank / singular values；
- fitted scale、rotation、translation；
- alignment RMS 和 normalized alignment RMS；
- reflection 是否被拒绝；
- near-zero baseline、non-finite pose、sentinel pose 等失败标志。

若任一窗口 alignment 不合格，整个 pair 标为 `INELIGIBLE_ALIGNMENT`，不进入四类科学
决策的分母，但必须单独报告比例和 scene 分布。不能把退化 pair 默默删除。

### 4.4 候选只修 camera center

V0 第一版只作用于 translation/camera center：

$$
c_i^{(\alpha)}=c_i^G+d(\alpha,i).
$$

Rotation 和 FoV 保持 global VGGT 输出不变。这样可以把实验问题限制为“平移修复方向
是否多模态”，避免 rotation representation 和 FoV 同时改变。它不是对未来方法能力的
永久限制。

---

## 5. GT 只用于冻结后的离线评价

### 5.1 Global-to-GT Sim(3) 每个 scene 只拟合一次

在所有预先判定为 GT-valid 的 global frames 上拟合一次：

$$
B_s^*=\operatorname*{arg\,min}_{B\in\mathrm{Sim}(3)}
\sum_{i\in\mathcal V_s}
\left\|B(c_i^G)-c_i^{GT}\right\|_2^2.
$$

$B_s^*$ 一旦拟合便冻结。global、$d_L$、$d_R$、均值和全部插值候选都使用同一个
$B_s^*$。严禁为每个候选重新对齐 GT；否则一个错误候选可能通过重新拟合 gauge 获得
不公平优势。

### 5.2 GT 泄漏检查

GT 不得进入：

- window 选择；
- local-to-global $A_k^*$；
- $d_L,d_R$ 构造；
- candidate filtering；
- 部署条件或推理 prior。

GT 可以进入：离线 endpoint/interpolation 指标、calibration 阈值选择和最终科学分类。
报告必须把这称为 privileged diagnosis，而不是 deployable selector。

---

## 6. 插值实验

### 6.1 固定插值网格

对每个 eligible pair 计算：

$$
d(\alpha)=(1-\alpha)d_L+\alpha d_R,
\qquad
\alpha\in\{0,0.25,0.5,0.75,1\}.
$$

- $\alpha=0$：left endpoint；
- $\alpha=1$：right endpoint；
- $\alpha=0.5$：最直接的平均修复；
- $0.25,0.75$：避免只检查中心点而漏掉内部坏区。

若后续增加更密网格，必须在 evaluation 前预注册；不能看到某条曲线后只挑最坏点。

### 6.2 评价范围

主分析只评价 pair 的共享 50 帧，保证两个残差定义在完全相同的 frame IDs 上。另可
报告 overlap 内的相对运动和平滑性，但不得把未被两个窗口共同覆盖的帧混入端点比较。

### 6.3 “内部变差”定义

对误差指标 $E$，定义平均惩罚

$$
P_{0.5}=E(d(0.5))-\max\{E(d_L),E(d_R)\},
$$

和预注册内部 barrier

$$
P_{\mathrm{int}}
=\max_{\alpha\in\{0.25,0.5,0.75\}}E(d(\alpha))
-\max\{E(d_L),E(d_R)\}.
$$

只有 $P_{0.5}$ 或 $P_{\mathrm{int}}$ 超过 calibration 冻结的绝对/相对 margin，才能称
“平均或内部插值更差”。两端仅仅数值略好不够。

---

## 7. 指标定义

### 7.1 Alignment residual

$$
R_{\mathrm{align}}^{(k)}
=\sqrt{\frac{1}{|W_k|}\sum_{i\in W_k}
\|\widetilde c_i^{(k)}-c_i^G\|_2^2}.
$$

同时除以 global window path length 或 centered RMS 得到 normalized value，避免 scene
尺度直接决定阈值。

### 7.2 Direction cosine

将 overlap 上所有 residual 展平：

$$
\mathbf d_L=[d_L(i_1);\ldots;d_L(i_{50})],\qquad
\mathbf d_R=[d_R(i_1);\ldots;d_R(i_{50})].
$$

定义

$$
\cos(d_L,d_R)=
\frac{\langle\mathbf d_L,\mathbf d_R\rangle}
{\|\mathbf d_L\|_2\|\mathbf d_R\|_2+\varepsilon}.
$$

它回答整体方向是否一致，但小幅残差会使 cosine 不稳定，因此必须与 magnitude 和
separation 一起使用。

### 7.3 Normalized RMS separation

$$
S_{\mathrm{NRMS}}
=\frac{\operatorname{RMS}(d_L-d_R)}
{\tfrac12[\operatorname{RMS}(d_L)+\operatorname{RMS}(d_R)]+\varepsilon}.
$$

该量接近零表示两候选相似；较大表示相对各自修复幅度而言，两方向明显不同。

### 7.4 逐帧方向一致率

仅在两侧逐帧残差幅度都超过 calibration noise floor 的帧上计算

$$
r_{\mathrm{agree}}
=\frac{1}{|\mathcal J|}
\sum_{i\in\mathcal J}
\mathbf 1[\cos(d_L(i),d_R(i))\ge\tau_{\mathrm{frame}}].
$$

若有效集合 $\mathcal J$ 太小，pair 标为 `INELIGIBLE_SMALL_RESIDUAL`，不能用随机
cosine 产生假分离。

### 7.5 Translation error

对任一候选 $X$，在冻结 $B_s^*$ 下计算 overlap ATE：

$$
E_{\mathrm{ATE}}(X)
=\sqrt{\frac1{50}\sum_i
\|B_s^*(c_i^X)-c_i^{GT}\|_2^2}.
$$

报告相对 global 的改善：

$$
\Delta_{\mathrm{ATE}}(X)
=E_{\mathrm{ATE}}(G)-E_{\mathrm{ATE}}(X).
$$

正值表示候选改善 global prediction。

### 7.6 Relative translation error

对预注册的 frame gaps $\Delta$，比较候选和 GT 的相对位移。RTE 能降低单个全局偏移
对 ATE 的支配，并检查候选是否真的修复轨迹形状，而不只是整体移动 overlap。

主结论需同时满足 ATE 与 RTE 的方向一致；若二者冲突，标为灰区并单独分析。

---

## 8. 阈值冻结与 pair 分类

### 8.1 Calibration 只做一次

在 calibration split 上冻结：

- $\tau_{\mathrm{align}}$：最大 normalized alignment residual；
- $\tau_{\mathrm{valid}}^{abs},\tau_{\mathrm{valid}}^{rel}$：端点改善门槛；
- $\tau_{\mathrm{sep}}$：方向分离门槛；
- $\tau_{\mathrm{frame}}$：逐帧方向一致阈值；
- $\tau_{\mathrm{barrier}}^{abs},\tau_{\mathrm{barrier}}^{rel}$：内部变差门槛；
- 所有灰区 margin 和最小有效帧数。

冻结值及其配置 hash 写入 result card。evaluation 期间不能修改。

### 8.2 Endpoint validity

候选 $X\in\{L,R\}$ 只有同时满足以下条件才算有效：

1. pair alignment eligible；
2. $\Delta_{\mathrm{ATE}}(X)$ 超过绝对和相对改善门槛；
3. RTE 不恶化超过冻结 margin；
4. 指标不落在灰区；
5. 不触发 non-finite、sentinel 或 extreme-scale guard。

“比 global 好一点点”不能自动算有效。

### 8.3 四类决策的可执行规则

1. **`NOT_SUPPORTED`**
   - 没有双端有效；且不形成清晰的“一好一坏”；或
   - 两候选不分离；或
   - 事件只出现在 calibration/development，evaluation 不复现。

2. **`SELECTOR_PROBLEM`**
   - 恰好一个端点有效；
   - 另一端明确无效而非灰区；
   - 跨 scene 稳定出现可区分的一好一坏。

3. **`CONTINUOUS_REDUNDANCY`**
   - 两端都有效；
   - 方向达到分离门槛或至少不是同一数值点；
   - 所有预注册内部插值保持有效；
   - $P_{0.5}$、$P_{\mathrm{int}}$ 不超过 barrier margin。

4. **`MULTIMODAL_VELOCITY_SUPPORTED`**
   - 两端都有效；
   - $S_{\mathrm{NRMS}}$、cosine/逐帧指标共同支持方向分离；
   - 平均或内部插值超过冻结 barrier margin；
   - 负控制正常；
   - scene-level prevalence 的置信区间通过 GO 门槛。

落入任一灰区的 pair 单列 `INDETERMINATE`，不强行归类。

---

## 9. 负控制与正控制

### 9.1 Self-pair

令 $d_R=d_L$。预期 separation 为零、插值曲线恒定。若不成立，说明指标或候选应用
实现有 bug。

### 9.2 Gauge copy

对同一 local trajectory 施加已知 Sim(3)，再走完整 prediction-only alignment。
预期恢复原候选。若被判为分离，说明 gauge handling 失败。

### 9.3 随机错误窗口

从预注册的不匹配 scene/pair 池中取 residual，并按相同幅度归一化后应用到当前 overlap。
预期 endpoint validity 显著下降。该控制检验评价是否真的拒绝错误方向，而不是“任何
小改动都看起来更好”。

### 9.4 残差取反

对有效候选测试 $-d_L$。如果正反方向都同样改善，可能存在评价不敏感、尺度过小或
连续弱方向，不能轻易称 selector/多模态。

### 9.5 小扰动

在 residual 上加入低于 calibration noise floor 的小扰动。决策应稳定；若标签频繁
翻转，说明阈值和灰区太窄。

### 9.6 退化对齐

构造 near-zero baseline、collinear centers、重复 pose、invalid sentinel 和极少有效帧。
预期全部进入明确 eligibility failure，而不是产生巨大 scale 或虚假分离。

---

## 10. 统计协议

### 10.1 分析单位

原始观察是 pair，但独立重采样单位是 scene。每个 scene 内先计算：

- eligible pair 数/8；
- 四类 pair 比例；
- 至少出现一个各类事件的 scene indicator；
- pair 指标的 scene median 和 worst case。

总体 bootstrap 对 scenes 有放回采样，保留每个 scene 内全部 pairs，避免伪重复。

### 10.2 报告内容

每个标签报告：

- scene prevalence 与 95\% bootstrap CI；
- pair prevalence，仅作描述；
- eligibility failure prevalence；
- calibration 与 evaluation 分开；
- scene 数、pair 数和有效帧数；
- seed、bootstrap 次数和统计代码 commit。

### 10.3 GO / NO-GO

正式数值阈值必须在 calibration 阶段填写到 result card。结构性规则固定如下：

- `GO_VRFM`：`MULTIMODAL_VELOCITY_SUPPORTED` scene prevalence 的置信区间下界超过
  预注册门槛，控制实验正常；
- `GO_SELECTOR`：一好一坏事件占主导，多模态事件未过门槛；
- `GO_DETERMINISTIC`：双端有效但内部安全，或候选近似一致；
- `NO_GO_PIPELINE`：alignment/GT eligibility failure 过高，先修管线；
- `NO_GO_EVIDENCE`：样本量或 provenance 不足，不做模型结论。

---

## 11. 必须保存的 artifact

建议每次冻结 run 至少产生：

```text
<run_root>/
  input_manifest.json
  config.yaml
  provenance.json
  scene_summary.parquet
  pair_metrics.parquet
  interpolation_metrics.parquet
  control_metrics.parquet
  decision_card.json
  figures/
    overlap_protocol.pdf
    interpolation_curves/
    representative_cases/
    failure_cases/
  artifact_manifest.sha256
```

`pair_metrics` 至少含 scene/window/frame identity、eligibility、alignment、cosine、NRMS、
endpoint ATE/RTE、五个 alpha 指标、decision 和 failure reason。`decision_card` 是文档
回填的唯一摘要来源。

---

## 12. 结果呈现模板

### 12.1 总结果卡

| 字段 | 冻结值 |
|---|---|
| Status | `protocol in progress; no scientific conclusion yet` |
| Branch / commit | `PENDING_COMMITTED_RESULT_CARD` |
| Run ID | `PENDING_COMMITTED_RESULT_CARD` |
| Dataset / split | `PENDING_COMMITTED_RESULT_CARD` |
| Input manifest hash | `PENDING_COMMITTED_RESULT_CARD` |
| Scene / eligible pair count | `PENDING_COMMITTED_RESULT_CARD` |
| Decision prevalence + CI | `PENDING_COMMITTED_RESULT_CARD` |
| GO / NO-GO | `PENDING_COMMITTED_RESULT_CARD` |

### 12.2 每类结论必须配什么图

- `SELECTOR_PROBLEM`：同一 overlap 的 global/GT/left/right 轨迹和一好一坏指标；
- `CONTINUOUS_REDUNDANCY`：五个 alpha 的平坦/安全曲线；
- `MULTIMODAL_VELOCITY_SUPPORTED`：双端有效、内部升高的真实曲线及 residual arrows；
- `NOT_SUPPORTED`：最常见失败原因和一个代表性 false candidate；
- 每种正面案例同时给一个失败或反例，禁止只挑最好看的 scene。

### 12.3 结果语言模板

若 GO：

> 在冻结的 evaluation split 和阈值下，观察到跨 scene 稳定的双端有效、方向分离且
> 内部插值变差事件。这支持对 camera repair velocity 使用 latent-conditioned model；
> 不构成多个物理真轨迹的证明。

若 selector：

> 多数分离候选表现为一好一坏，证据更支持候选选择问题，而非同时保留多个修复速度。

若 deterministic：

> 多个有效候选之间的预注册插值保持有效，尚无证据表明确定性融合的平均方向落入坏区。

若证据不足：

> 当前运行未满足 eligibility、样本量或 provenance 门槛，不对修复速度分布作科学结论。

---

## 13. 复现清单

冻结结果前，另一位研究者应能仅凭仓库完成以下检查：

- [ ] checkout result card 指定 commit；
- [ ] 校验 input manifest 与 artifact manifest 的 SHA-256；
- [ ] 确认 500 个 frame IDs 和九个窗口逐项一致；
- [ ] 复现 global + 9 local VGGT predictions；
- [ ] 确认 local-to-global alignment 未访问 GT；
- [ ] 确认 global-to-GT Sim(3) 每 scene 只拟合一次；
- [ ] 重算 $d_L,d_R$ 和五个 alpha；
- [ ] 重算 eligibility、ATE、RTE、cosine、NRMS 与逐帧一致率；
- [ ] 从冻结 thresholds 重建每个 pair decision；
- [ ] 以 scene 为单位重跑 bootstrap；
- [ ] 检查所有负控制；
- [ ] 对照 artifact hashes 和代表性 case IDs；
- [ ] 运行命令、环境与代码 commit 全部记录。

任何口头补充都不应成为复现所必需的信息。

---

## 14. 当前状态与下一次更新条件

当前只有协议，没有 V0 的冻结 result card：

> `protocol in progress; no scientific conclusion yet`

下一次允许更新科学结果的条件是：实验分支提交完整 provenance、冻结阈值、逐 pair
metrics、scene-level bootstrap、控制结果和 decision card。届时只按冻结字段回填，
不在文档分支重新挑阈值或筛样本。

---

## 参考文献

- Wang et al. *VGGT: Visual Geometry Grounded Transformer*. CVPR, 2025.
- Hartley and Zisserman. *Multiple View Geometry in Computer Vision*. 2nd ed., 2004.
- Triggs et al. *Bundle Adjustment—A Modern Synthesis*. 2000.
- Liu, Gong, and Liu. *Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow*. ICLR, 2023.
- Guo and Schwing. *Variational Rectified Flow Matching*. ICML, 2025.
