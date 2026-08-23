# 用 Variational Rectified Flow 融合 VGGT 多短窗修复方向

> 文档角色：三文档中的概念方法设计
>
> 状态：`protocol in progress; no scientific conclusion yet`
>
> 前置门槛：只有 V0 冻结为 `GO_VRFM` 才正式实现
>
> 日期：2026-08-23

---

## 0. 三分钟版本

一条 500 帧序列先由 VGGT 整体预测。整体上下文长，但相机轨迹可能漂移。我们再运行
九个长度 100、stride 50 的短窗口。它们更关注局部几何，却会在共享区域给出不同的
修复建议。

最简单的做法是平均所有建议。但假设左窗口说“向左修”，右窗口说“向右修”，两个
方向各自都能到达一个好区域，平均以后却变成“原地不动”或落到坏区域。此时一个只
输出单一平均方向的确定性模型可能不合适。

普通 Rectified Flow（RF）在同一个 noisy state、time 和 condition 上只输出一个
确定性速度。若训练中同一点出现多个目标方向，MSE 会把它们压成条件均值。

Variational Rectified Flow Matching（V-RFM）增加 latent $z$：

$$
v_\theta(r_t,t,c,z).
$$

同一个可见状态可以在不同 $z$ 下产生不同修复速度。对一条长序列只采样一次 $z$，
并在整段 ODE integration 中保持不变，从而让整条轨迹采用一致的修复策略。

但 V-RFM 只负责**生成多个 coherent repairs**，不负责判断哪个短窗口正确。独立 scorer
或几何评价仍然必需。更重要的是，local window outputs 只是候选来源，不天然就是多个
训练 targets；它们必须先被独立验收并组成完整、连贯的序列级 target residuals。

---

## 1. 实际问题：global 看得远，local 看得细

### 1.1 输入场景

设 global VGGT 输出 500 帧 camera centers：

$$
C^G=[c_0^G,\ldots,c_{499}^G]\in\mathbb R^{500\times3}.
$$

九个 local VGGT windows 经过 prediction-only alignment 后，得到局部 camera centers
$\widetilde C^{(k)}$ 和 residuals

$$
D^{(k)}=\widetilde C^{(k)}-C^G
$$

（只在各自覆盖帧上定义）。相邻窗口在共享 50 帧上提供 $d_L,d_R$。

### 1.2 为什么不能先平均

用一维例子：当前点在 0，两个有效修复目标分别在 $-1$ 和 $+1$。若训练目标各半，
单一 MSE 回归的均值是 0。这个均值可能恰好什么也没修。

映射到相机轨迹时，$d_L,d_R$ 是 50 帧或更长时域上的向量场，不只是一个标量。它们
可以在整体方向、时间形状或局部弯曲上冲突。只有前置实验确认“两个端点都好、内部
平均变差”，这个例子才对应真实 VGGT 现象。

### 1.3 不能跳过的前置结论

- `NOT_SUPPORTED`：不做 V-RFM，先改候选或评价；
- `SELECTOR_PROBLEM`：优先做窗口/候选 selector；
- `CONTINUOUS_REDUNDANCY`：优先 deterministic fusion；
- `MULTIMODAL_VELOCITY_SUPPORTED`：才进入本文方法。

---

## 2. 普通 RF 为什么会把速度平均

### 2.1 Rectified Flow 的基本训练对

从 source residual $r_0$ 和 target residual $r_1$ 构造直线：

$$
r_t=(1-t)r_0+t r_1,\qquad t\sim U(0,1),
$$

目标速度为 $r_1-r_0$。普通 conditional RF 学习

$$
v_\theta(r_t,t,c),
$$

其中 $c$ 是 VGGT condition。

### 2.2 冲突发生在哪里

若不同训练 coupling 在相同 $(r_t,t,c)$ 上给出不同 $r_1-r_0$，MSE 的最优确定性
输出是这些目标速度的条件均值。模型可以通过弯曲后续轨迹绕开冲突，但当积分步数少、
数据有限或冲突强时，均值速度可能成为困难点。

这与“最终 target distribution 有几个簇”不是同一句话。V-RFM 论文研究的是
data-time space 的 multi-modal velocity：轨迹可以在相同位置和时间相交，但具有不同
方向。

---

## 3. V-RFM 的核心变化

### 3.1 给速度加一个身份变量

V-RFM 使用

$$
p(v\mid r_t,t,c,z)
=\mathcal N\!\left(v;v_\theta(r_t,t,c,z),I\right).
$$

对 $z$ 积分后，$p(v\mid r_t,t,c)$ 可以成为混合分布。直觉上，$z$ 不是“第几个
窗口”的标签，而是整条修复路径的隐藏身份。

### 3.2 训练 posterior 与推理 prior

训练时，recognition model 可以看 target residual：

$$
q_\phi(z\mid r_0,r_1,r_t,t,c).
$$

推理时 $r_1$ 不存在，只能从 prior 采样。最忠实于原始 V-RFM 的第一版使用

$$
p(z)=\mathcal N(0,I).
$$

以后可以研究 condition-dependent prior $p_\psi(z\mid c)$，但这会增加 prior-posterior
matching 难度，不应在第一版同时引入。

### 3.3 序列级 latent

原始 V-RFM 推理在 ODE integration 前采样一次 $z$，随后保持不变。我们把这个原则
提升到整条 VGGT 长序列：

- 一条 500 帧序列只采样一次 $z$；
- 同一 $z$ 供所有帧、窗口覆盖和 integration times 使用；
- 禁止每帧或每个 overlap 重新采样。

这样做的目的不是让 $z$ 更神秘，而是避免相机轨迹在时间上突然从一种修复策略切换
到另一种。

---

## 4. VGGT 条件输入

### 4.1 不先压成一个 local input

condition encoder 接收：

1. global camera centers $C^G$；
2. 所有九个 aligned local camera trajectories；
3. 每个 local window 的 coverage mask；
4. alignment confidence、scale 和 eligibility features；
5. 可选的 VGGT camera/image tokens；
6. frame position 和窗口 ID embedding。

所有 local trajectories 必须作为集合/序列保留，不能在进模型前先平均成一个 local
trajectory。否则最关键的方向差异已经被抹掉。

### 4.2 一个具体张量视图

对 $N=500,K=9$：

```text
global centers:        [N, 3]
aligned local centers: [K, N, 3]   # 未覆盖位置填零
coverage mask:         [K, N]
alignment features:    [K, F_a]
camera features:       [N, F_c]    # 可选
```

condition encoder 输出按帧 context $h_{1:N}$ 和序列 summary $h_{seq}$。velocity network
在每个 flow time 使用 $r_t$、$h_{1:N}$、$h_{seq}$ 和同一个 $z$。

### 4.3 第一版不修什么

输出只生成 camera-center residual：

$$
\widehat C=C^G+\widehat R.
$$

Rotation 和 FoV 保持 global VGGT 输出。原因是减少变量、直接对齐 V0 的证据，不是
理论上永远不修它们。只有 translation 版本过门槛后，才分别扩展 rotation/FoV。

---

## 5. 最关键的开放问题：target residual 从哪里来

### 5.1 多个窗口不是天然的多个 target

如果训练数据有唯一 GT trajectory，那么

$$
R^{GT}=C^{GT}_{\text{frozen gauge}}-C^G
$$

是一个确定 target。仅仅因为 condition 中有多个 local windows，并不会自动把 target
变成多模态。若只用唯一 GT residual 监督，V-RFM 可能退化成一个不必要的随机模型。

反过来，直接把所有 $D^{(k)}$ 当作 targets 也不成立，因为其中可能有错误窗口、gauge
伪差或局部不连续。V-RFM 会学习输入给它的分布，不会自动清洗标签。

### 5.2 MVP 的离线 target builder

在 V0 为 `GO_VRFM` 后，训练前增加一个不参与推理的 target builder：

1. 对每个 overlap 只保留通过冻结 evaluator 的 local residual endpoints；
2. 用预注册的时序连续性与 coverage 规则，把局部残差组装成完整 500 帧候选；
3. 从不同 coherent assignments 或不同优化初始化得到多个 full-sequence residuals；
4. 使用同一个独立 evaluator 验收整条修复轨迹；
5. 去除 gauge copies、近重复项和明显不连续项；
6. 得到序列级有效 target set
   $\mathcal R_s=\{R_s^{(1)},\ldots,R_s^{(M_s)}\}$。

训练时从 $\mathcal R_s$ 采样 $r_1$。这个 target set 的形成规则必须固定并有 provenance。
如果大多数 scene 最终只有一个有效 full-sequence target，就没有足够理由训练多模态
V-RFM。

### 5.3 为什么 sequence target builder 仍需单独验证

V0 的 pairwise 证据只说明局部 overlap 上有多方向。八个 pair 的局部选择组合起来，
可能产生时序不一致或指数级组合。序列级 target builder 是 V0 之后、正式训练之前的
第二道门：它必须证明多个局部有效方向能组成多个完整 coherent repairs。

这也是本方法设计保持“粗糙”的地方。我们现在冻结接口和证据要求，不提前承诺某种
动态规划、图搜索或优化器一定最好。

---

## 6. 一个核心训练目标

从 scene 的有效 target set 采样 $r_1$，从 source distribution 采样 $r_0$，再采样
$t\sim U(0,1)$ 并构造 $r_t$。训练 posterior 采样 $z$。教学性目标写为：

$$
\mathcal L=
\mathbb E\!\left[
\mathbb E_{z\sim q_\phi}
\left\|v_\theta(r_t,t,c,z)-(r_1-r_0)\right\|_2^2
+\beta D_{KL}\!\left(q_\phi(z\mid r_0,r_1,r_t,t,c)\,\|\,p(z)\right)
\right].
$$

第一项让不同 $z$ 重建不同目标速度；第二项让训练 posterior 不至于与推理 prior 完全
分离。原始 V-RFM 对应 $p(z)=\mathcal N(0,I)$，并在推理前采样一次 $z$。

这一个公式足以说明概念。具体 $eta$、latent dimension、网络宽度、solver 和 time
sampling 必须在未来实现计划中通过 baseline 与消融确定。

---

## 7. 训练流程

对每条训练序列：

1. 读取冻结 global/local VGGT predictions 和 coverage/alignment metadata；
2. 从已验收 target set $\mathcal R_s$ 采样一个 $r_1$；
3. 采样 source residual $r_0$ 和 flow time $t$；
4. 编码全部 condition，不提前平均 local trajectories；
5. posterior 读取 $r_1$，采样一次 sequence-level $z$；
6. velocity network 对全部 500 帧预测 target velocity；
7. 优化 velocity reconstruction 与 KL；
8. 记录 latent usage、mode coverage 和 target identity，不只看总 loss。

### 7.1 Mask 与坐标处理

- 所有 residual 在同一 global prediction gauge 中；
- source/target normalization 只用 training statistics；
- coverage mask 防止填零位置被当作真实 local center；
- loss 按有效帧和 scene 归一化，不能让长有效区支配；
- 输出应用后仍使用合法 camera-center 表示，rotation 不变。

### 7.2 Posterior collapse

若 velocity network 不看 $z$ 也能降低 MSE，posterior 可能塌缩到 prior，生成样本几乎
相同。必须同时监控：

- KL 是否长期接近零；
- 固定 condition 下改变 $z$ 是否改变 coherent trajectory；
- 不同 target identity 是否映射到不同 latent regions；
- sample diversity 是否只是高频噪声；
- diversity 是否通过独立 evaluator。

不能为了增大 diversity 而简单减弱 KL；那可能扩大 prior-posterior gap，使推理样本
失效。

---

## 8. 推理流程

1. 运行一次 500 帧 global VGGT；
2. 运行九个 100 帧 local VGGT windows；
3. prediction-only alignment，并生成 coverage/confidence；
4. condition encoder 读取所有 global/local evidence；
5. 采样 source residual $r_0$；
6. 从 $p(z)$ 采样一次 sequence-level $z$；
7. 用 $v_\theta(r_t,t,c,z)$ 完整积分到 $\widehat R$；
8. 输出 $\widehat C=C^G+\widehat R$；
9. 重复不同 $(r_0,z)$ 产生少量 samples；
10. 用 deployable independent scorer 或下游任务选择/排序。

第 10 步不能使用 GT。若没有可靠 scorer，可以保留多样本交给人工或下游，但不能宣称
系统已自动找到正确轨迹。

---

## 9. Generator 和 selector 的边界

V-RFM 回答：

> 在有效修复目标确实多模态时，如何生成多条不被均值压扁的 coherent trajectories？

它不回答：

> 哪条生成轨迹最符合真实相机运动？

后一个问题需要：几何 consistency energy、图像重投影、track/depth/point consistency、
VGGT confidence、下游重建质量或学习到的 scorer。scorer 必须与 target generator 分开，
否则会形成“生成器自己给自己判对”的循环论证。

如果 V0 主要是 `SELECTOR_PROBLEM`，直接训练 selector 更简单；若内部平均安全，
deterministic fusion 更高效。V-RFM 不是默认答案。

---

## 10. Baselines 与消融

### 10.1 必须比较的模型

| 模型 | 回答的问题 |
|---|---|
| Raw global VGGT | 不修复时的基准 |
| Deterministic mean/weighted fusion | 简单平均是否已经足够 |
| Deterministic Transformer refiner | 强确定性模型能否利用全部 local evidence |
| Ordinary conditional RF | 多步连续生成但没有 velocity latent 是否足够 |
| V-RFM | 显式多模态 velocity 是否有额外价值 |
| V-RFM without camera features | VGGT camera/image features 是否必要 |

### 10.2 关键消融

- sequence-level $z$ vs per-window/per-frame $z$；
- all local trajectories vs pre-averaged local input；
- fixed standard-normal prior vs future conditional prior；
- unique GT residual targets vs validated multi-target set；
- no alignment confidence；
- no independent scorer；
- latent dimensions；
- solver steps 与少步积分性能。

主要评价不是“样本越多越好”，而是：best-of-$K$、scorer-selected、coverage、invalid
rate、trajectory smoothness、ATE/RTE、mode collapse 和计算成本。

---

## 11. 失败模式

| 失败 | 表现 | 优先处理 |
|---|---|---|
| 候选标签有错 | 多样本多，但大多无效 | 修 evaluator/target builder |
| 只有唯一 full-sequence target | latent 不被使用 | 回到 deterministic refiner |
| Posterior collapse | 不同 $z$ 产生相同轨迹 | 检查条件泄漏、capacity、KL schedule |
| Prior-posterior gap | 训练重建好，推理样本差 | 改 prior matching，不先加模型尺寸 |
| Temporal mode switching | 轨迹在窗口边界跳变 | 强制 sequence-level $z$ 与全序列网络 |
| Diversity = noise | 样本不同但几何都差 | 独立 scorer 与平滑/合法性约束 |
| Scorer 同源偏差 | 生成器偏爱 evaluator 漏洞 | 增加独立几何/下游评价 |

---

## 12. 进入条件与当前状态

正式实现前必须满足：

1. V0 result card 为 `GO_VRFM`；
2. thresholds、manifest、run/artifact hashes 已冻结；
3. 多个 pairwise 有效方向能组成多个 sequence-level coherent targets；
4. target builder 与 scorer 的角色和信息访问已分开；
5. deterministic Transformer 与 ordinary RF baselines 已定义；
6. GT 不进入推理 condition 或 deployable scorer。

当前状态：

> `protocol in progress; no scientific conclusion yet`

因此本文是一个条件方法设计，不表示 V-RFM 已被选定。若前置实验得到 selector 或
deterministic 结论，本文保留为被否决方案的设计记录。

---

## 参考文献

- Wang et al. *VGGT: Visual Geometry Grounded Transformer*. CVPR, 2025.
- Liu, Gong, and Liu. *Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow*. ICLR, 2023.
- Lipman et al. *Flow Matching for Generative Modeling*. ICLR, 2023.
- Guo and Schwing. *Variational Rectified Flow Matching*. ICML, 2025.
