# 从相机轨迹多解到修复速度歧义

## VGGT 长短上下文修复的理论基础

> 文档角色：三文档中的理论基础
>
> 分支：`codex/camera_solution_space_docs_reframe`
>
> 状态：`protocol in progress; no scientific conclusion yet`
>
> 日期：2026-08-23

---

## 0. 先说结论

我们现在其实面对两个不同的问题。

第一个是传统的**物理解空间问题**：同一组图像和几何约束全部固定以后，是否仍有
多条物理上不同、评价上都成立的相机轨迹？这个问题需要处理 gauge、可辨识性、
子水平集和连通性，数学上很严格。

第二个是更贴近当前 VGGT 修复方案的**修复速度歧义问题**：一条 500 帧序列先由
VGGT 整体预测，再切成九个长度 100、stride 50 的短窗口。对于相邻窗口共享的
50 帧，左右两个短窗口可能给出两份不同的修复建议。我们要问的是：

> 两份建议是不是都能改善长序列？如果都有效，它们能不能安全平均？

Variational Rectified Flow Matching（V-RFM）主要对应第二个问题。它不要求我们
先证明世界中存在两个断开的物理真解；只要同一个数据—时间状态上确实存在多个
有效、互相冲突、而且平均后更差的目标速度，就有直接理由研究 latent-conditioned
velocity。

但这里有一条不能越过的边界：

> 多个短窗口产生多个输出，只说明我们有多个**候选**。只有通过统一、冻结、
> 不参与候选生成的评价后，它们才可能成为多个**有效修复方向**。

因此当前最关键的工作不是立刻训练 V-RFM，而是先完成 `V0：多短窗修复速度歧义
诊断`。本报告建立概念和证据边界；详细实验协议见配套前置实验文档。

---

## 1. 从 VGGT 的实际场景开始

### 1.1 一条 500 帧轨迹和九个短窗口

设 VGGT 对完整 500 帧序列给出一条长序列预测。记第 $i$ 帧的 global camera
center 为

$$
c_i^G\in\mathbb R^3,\qquad i=0,\ldots,499.
$$

再用长度 100、stride 50 切出九个短窗口：

```text
W0:   0–99
W1:  50–149
W2: 100–199
...
W8: 400–499
```

相邻窗口总有 50 帧重叠。例如 $W_0$ 和 $W_1$ 都预测了第 50–99 帧。由于两个
窗口看到的上下文不同，它们的局部坐标系也不同，不能把 raw camera centers 直接
相减。先只用预测值把局部结果对齐到 global prediction 的 gauge，得到
$\widetilde c_i^{(k)}$。第 $k$ 个窗口在覆盖帧上的候选残差定义为

$$
d_k(i)=\widetilde c_i^{(k)}-c_i^G.
$$

于是一个 overlap 上会有左窗口建议 $d_L$ 和右窗口建议 $d_R$。它们就是我们
准备检验的两个候选修复方向。

### 1.2 为什么长短帧关系目前确实很弱

长窗口和短窗口只共享部分图像，却改变了很多东西：

- 进入模型的上下文长度不同；
- attention 能看到的远距离关系不同；
- 每次预测的坐标 gauge 不同；
- 长序列的累积漂移和短窗口的局部稳定性不同；
- 边界帧在两个窗口中的相对位置不同；
- 可见性、遮挡与有效视差也可能不同。

所以“100 帧预测”和“500 帧预测”不是同一条件下对同一未知量的重复求解。
仅仅看到二者不一样，不能证明固定观测下有多个物理解。

但这并不让短窗口失去价值。短窗口仍然可以作为一个**候选修复生成器**：它们从
不同局部上下文出发，对同一段 global trajectory 提出多个修改方向。我们当前要
验证的是这些修改方向的结构，而不是把它们直接升级为物理真解。

### 1.3 两条问题线不能互相替代

| 问题线 | 固定什么 | 研究对象 | 可以支持的结论 |
|---|---|---|---|
| 物理解空间 | 完整观测、传感器假设、几何约束、评价能量 | 去 gauge 后的可接受相机轨迹集合 | 唯一性、弱可辨识方向、连通分量、能垒 |
| 修复速度分布 | noisy/global state、时间、全部可部署条件与候选构造协议 | 对 camera residual 的条件速度分布 | 确定性融合、selector、连续冗余或速度多模态 |

可能出现四种组合：物理解近似唯一，但修复算法有多个有效路径；物理解存在连续
弱方向，但当前修复目标仍近似单峰；二者都有歧义；二者都没有。因而我们不能用
任何一条线替另一条线下结论。

---

## 2. 六种“看起来不一样”必须分开

“多解”这个词太容易把不同现象混在一起。下面六类现象在 VGGT 中都有可能出现，
但它们的含义完全不同。

| 现象 | 通俗解释 | VGGT 中的例子 | 不能直接推出什么 |
|---|---|---|---|
| 物理多解 | 同一完整观测真的允许不同三维相机运动 | 退化运动或严重遮挡下，两条去 gauge 后仍远离的轨迹都满足独立几何评价 | 不能由 long/short 输出不同直接推出 |
| Gauge 冗余 | 同一物理轨迹换了全局坐标系 | 整条轨迹一起旋转、平移，单目时还可能整体缩放 | 不是新的物理解，也不是模型 mode |
| 局部弱约束 | 某些方向改一点，误差变化很慢 | 前向小视差时，某段深度/平移方向难以辨认 | 不等于有离散分支 |
| 优化 basin | 求解器从不同初值走到不同停点 | BA 或轨迹优化的多个局部极小 | 不等于可行集道路不连通 |
| 算法输出差异 | 输入或执行协议变了，预测随之变化 | 100 帧与 500 帧预测、两个不同短窗口预测 | 不等于两个输出都有效 |
| 概率/速度多模态 | 同一已固定条件下，合理目标分布有多个峰或多个方向 | 同一 overlap 上 $d_L,d_R$ 都有效，但中间平均方向明显更差 | 不等于多个最终物理真解 |

最容易犯的错误是从第五行直接跳到第一行或第六行。前置实验的职责，就是在统一
评价下把“输出差异”逐步筛成“有效候选”，再判断它属于 selector、连续冗余还是
速度多模态。

---

## 3. 物理解空间：它仍然重要，但不是 V-RFM 的唯一门票

### 3.1 “固定观测”不只是固定图片文件名

完整条件记为 $C$。它至少应包含：

- 图像像素、frame identity、顺序和长度；
- resize、crop、归一化、颜色空间和分辨率；
- 内参、畸变、时间戳和 rolling-shutter 假设；
- 动态物体、天空、遮挡和可见性 mask；
- 已知尺度、重力、地图、首尾锚点等外部约束；
- 所有 residual、权重、robust loss、阈值和随机源。

少固定其中一项，研究对象就换了。long-vs-short 最明显地改变了条件中的序列长度
和上下文，因此不属于固定 $C$ 的重复采样。

### 3.2 Gauge 是“换尺子和坐标系”，不是新世界

Gauge 指同一物理重建采用不同全局坐标表示。例如整条相机轨迹和场景一起平移、
旋转，在单目未知尺度时再整体缩放，图像投影仍可能不变。研究多解时必须先把这些
表示差异除掉。

具体群不能机械写死：

- 有真实 metric depth 或尺度锚点时，尺度通常不可自由变化；
- 纯单目、无尺度锚点时，常见 gauge 包含 $\mathrm{Sim}(3)$；
- 已固定首帧位姿、重力或地图时，剩余 gauge 会更小。

所以我们研究的是商空间中的轨迹距离，而不是 raw pose vector 的欧氏距离。

### 3.3 四个不同集合

设轨迹为 $T$，可能还包括内参 $K$ 和场景/nuisance variables $Z$。冻结评价为
$E(C,T,K,Z)$。若只关心相机变量，可定义 profile energy

$$
\bar E(C,T,K)=\inf_Z E(C,T,K,Z).
$$

需要区分：

1. 精确观测 fiber：所有完全产生同一理想观测的状态；
2. 全局最优集：$\operatorname*{arg\,min}\bar E$；
3. 相对近优集：$\bar E\le E^*+\epsilon$；
4. 绝对可接受集：$\bar E\le h$。

实验中的“有效候选”通常属于第四类。它由预注册阈值 $h$ 决定，不应在看完结果后
调到刚好包住喜欢的样本。

### 3.4 用四个量描述，不做错误的三选一

正容差集合一般不是一条细线，而会在可接受点周围形成有厚度的区域。因此
“唯一 / 连续低维 / 离散分支”不是严谨的互斥分类。更可靠的是同时报告：

- $D_h$：去 gauge 后可接受集合的直径；
- $\beta_0$：道路连通分量数；
- $\mathcal H_{h,\delta}$：远候选之间必须跨过的 minimax 高度；
- $k_{\mathrm{weak}}$：局部弱可辨识方向数。

两个断开的分量内部仍可能各有连续弱方向；一个连通集合也可能直径很大。详细定义
和技术条件放在附录。

### 3.5 为什么这条线仍值得保留

物理解空间研究能回答更深的问题：VGGT 的误差来自不可辨识性，还是模型本身没有
利用足够的几何信息？它还能为独立 scorer、阈值和失败案例提供严格依据。但它是
一条长期研究线，不应再被写成“只有先认证断开物理解，才能使用 latent”的硬门槛。

---

## 4. 修复速度歧义：当前最直接的前置问题

### 4.1 多解可能出现在哪里

在当前方案中，最自然的多解位置不是“500 帧最终轨迹有几个世界”，而是：

> 给定同一个 global trajectory 状态、同一个 flow time、同一组完整条件证据，
> 目标 camera-center residual velocity 是否有多个有效方向？

这里的“完整条件证据”很重要。如果模型只看到左窗口或只看到右窗口，两个输出不同
只是条件不同。方法设计中应把 global trajectory、全部 local trajectories、coverage、
alignment confidence 和 camera features 一起作为 condition，再研究在这一共同条件下
目标 residual distribution 是否需要 latent。

短窗口输出只是构造训练目标候选的一种方式。最终的多模态性必须由冻结评价、数据
分布与配对协议共同定义，不能由候选生成器自己宣布。

### 4.2 为什么“两个端点都好”还不够

假设 $d_L$ 和 $d_R$ 各自修复后都比 global prediction 好。仍要看它们之间：

$$
d(\alpha)=(1-\alpha)d_L+\alpha d_R,
\qquad \alpha\in[0,1].
$$

如果所有内部插值都好，那么两个端点可能只是同一片连续冗余中的不同位置。此时
确定性平均、加权融合或普通回归可能已经足够。

如果两个端点都好，但 $\alpha=0.5$ 的平均方向或一段内部插值明显变差，说明“一条
确定性平均速度”可能落在不想要的位置。这才是 V-RFM 最直接的动机。

### 4.3 四类实验结果对应四种工程决策

| 冻结结果 | 含义 | 下一步 |
|---|---|---|
| `NOT_SUPPORTED` | 没有稳定的双端有效、方向分离事件 | 不做生成式修复，先改候选或评价 |
| `SELECTOR_PROBLEM` | 通常只有一个候选有效 | 做 scorer/selector，而不是多模态 generator |
| `CONTINUOUS_REDUNDANCY` | 双端和内部平均都有效 | 优先确定性融合、均值或低维连续参数化 |
| `MULTIMODAL_VELOCITY_SUPPORTED` | 双端有效且分离，平均/内部显著更差，跨场景稳定 | 进入 V-RFM 方法验证 |

最后一行支持的是“修复速度值得 latent 建模”。它仍不自动证明多个物理真解。

---

## 5. MSE、Diffusion、Rectified Flow 与 V-RFM

### 5.1 MSE 为什么会平均

若完全相同的输入 $u$ 对应多个目标速度 $v$，确定性网络 $f_\theta(u)$ 用 MSE
训练时，函数空间最优解是条件均值：

$$
f^*(u)=\mathbb E[v\mid u].
$$

“平均”不是说网络把两个训练样本逐像素抄在一起，而是说在相同条件上，平方误差
最优的单一输出位于目标条件分布的均值。如果两个有效方向相反，均值可能接近零；
如果中间方向恰好不好，单一 MSE 输出就会形成修复折中。

### 5.2 Diffusion 能不能处理离散分支

能。Diffusion/score model 并不只擅长连续冗余，也可以表达多峰甚至看起来分离的
数据分布。把“Diffusion 不能处理离散分支”当成一般结论是不准确的。

真正要区分的是三件事：

1. **表达能力**：模型族能否表示多峰分布；
2. **训练信号**：数据里是否真的覆盖了各个有效模式；
3. **我们研究的对象**：最终样本分布，还是 data–time space 中同一点的速度分布。

Diffusion 当然可以作为候选方法。但 V-RFM 与这里的问题更直接对齐：我们恰好怀疑
在相同 data–time location 上存在多个 target velocities，而普通确定性速度场会把它们
平均。选择 V-RFM 不是因为 Diffusion “做不到”，而是因为 V-RFM 把速度歧义本身
放进模型定义里。

还要注意严格拓扑措辞：连续 ODE、随机 SDE、有限时间密度和有限采样的 support
性质并不相同。“样本看起来分成两簇”既不是断开 support 的证明，也不是物理解空间
断开的证明。

### 5.3 普通 Rectified Flow 的冲突

Rectified Flow 用耦合样本 $(x_0,x_1)$ 构造直线路径

$$
x_t=(1-t)x_0+t x_1,
$$

其目标速度是 $x_1-x_0$。当不同 coupling 在同一 $(x_t,t)$ 处给出不同目标速度时，
普通确定性速度场 $v_\theta(x_t,t)$ 在 MSE 下学习这些速度的条件均值。Guo 与 Schwing
将其称为 data–time space 中的 multi-modal/ambiguous velocity 问题。

映射到 VGGT：$x_1$ 不是“整条物理真轨迹的任意一个答案”，而应先被操作化为经过
冻结协议验收的 target camera-center residual；condition 还必须包含全部可部署的
global/local evidence。否则我们只是把标签噪声或错误窗口塞进生成模型。

### 5.4 V-RFM 改了什么

V-RFM 引入连续 latent $z$，令

$$
p(v\mid x_t,t,z)=\mathcal N\!\left(v;
v_\theta(x_t,t,z),I\right).
$$

对 $z$ 积分后，$p(v\mid x_t,t)$ 成为混合分布，因此相同 data–time location 可以
有不同 latent-conditioned velocity。训练时用 recognition model
$q_\phi(z\mid x_0,x_1,x_t,t)$ 近似不可解 posterior；推理时从 prior $p(z)$ 采样，
然后在整次 ODE integration 中保持该 $z$。这允许轨迹在观测空间相交，但仍由 latent
区分其速度身份。

在 VGGT 修复中，我们计划使用**序列级 latent**：一条 500 帧轨迹只采样一次，
所有帧和窗口共享。这样可以避免模型每帧或每个 overlap 随机切换修复策略。

### 5.5 V-RFM 不负责什么

V-RFM 不会自动知道哪个短窗口是对的。它也不会把错误候选变成有效候选。它负责的
是：在训练目标确实包含多个有效速度方向时，不必把它们压成一个均值。

候选有效性和最终样本选择仍需独立 scorer、几何能量、置信度或下游任务。如果 V0
主要得到 `SELECTOR_PROBLEM`，合理答案是先做 selector，而不是强行训练 V-RFM。

---

## 6. V0：多短窗修复速度歧义诊断

V0 位于完整解空间拓扑研究和生成模型训练之前。它只回答一个窄问题：

> 多个重叠短窗口是否稳定地产生多个有效、分离、不可安全平均的 camera repair
> velocities？

### 6.1 必须冻结的对象

- scene 和 500 帧 frame identity；
- 九个 100 帧窗口及 overlap；
- local-to-global prediction-only alignment；
- 每个 scene 只拟合一次的 global-to-GT Sim(3)；
- 候选、均值和内部插值共用的评价指标；
- calibration split 上预注册的阈值和灰区；
- scene-level bootstrap 与结果标签规则。

GT 只用于 privileged offline diagnosis，不能进入候选产生或 prediction-only alignment。
否则会发生信息泄漏：实验可能证明“看过答案以后能选对”，而不是图像证据本身支持
多个修复方向。

### 6.2 最小插值检查

至少评价

$$
\alpha\in\{0,0.25,0.5,0.75,1\}.
$$

端点回答“两个候选各自是否有效”；中间点回答“它们能否安全融合”。同时报告方向
cosine、归一化 RMS separation、逐帧一致率、alignment residual、translation error
和 relative translation error，不能只看一个漂亮的二维投影。

### 6.3 负控制

至少包含：self-pair、纯 gauge copy、随机错误窗口、残差取反、小扰动和退化对齐。
这些控制分别检验管线能否识别零差异、坐标伪影、明显错误方向、方向敏感性、局部
稳定性和对齐失败。

### 6.4 GO 门槛

只有以下事件在多个 scene 上稳定复现，才进入正式 V-RFM 实现：

1. $d_L$ 与 $d_R$ 的对齐可靠；
2. 两个端点都通过冻结有效性门槛；
3. 两个方向按预注册 separation 指标显著分开；
4. 平均或预注册内部插值显著变差；
5. scene-level bootstrap 置信区间不依赖少数特殊场景。

前置实验未结束前，本报告不声明以上条件已经满足。

---

## 7. 证据等级与安全结论语言

### 7.1 证据是不对称的

- 找到一个远距离有效对，可以反证“所有有效解都近似唯一”；
- 找到一条完整低能路径，可以证明这一候选对在同一道路分量；
- 没找到远解，不能证明唯一；
- 路径优化失败，不能证明断开；
- 找到两个模型输出，不能证明两个输出都有效；
- 找到两个有效端点，也不能跳过内部插值直接称速度多模态。

### 7.2 推荐用语

| 证据 | 可以写 | 不应写 |
|---|---|---|
| long/short 输出不同 | “上下文敏感，产生候选修复差异” | “发现多个物理解” |
| 一个候选好、一个差 | “呈现 selector problem” | “存在双模态有效速度” |
| 双端和内部都好 | “存在连续可融合冗余” | “平均必然失败” |
| 双端好、内部差、跨场景稳定 | “支持多模态修复速度” | “证明多个物理真轨迹” |
| 路径搜索失败 | “尚未找到低能路径” | “可行集已经断开” |

---

## 8. 研究路线

```text
Stage 0  概念、术语与证据边界
    ↓
V0       多短窗修复速度歧义诊断（当前关键前置实验）
    ├─ NOT_SUPPORTED / CONTINUOUS_REDUNDANCY → 确定性融合
    ├─ SELECTOR_PROBLEM                     → scorer / selector
    └─ MULTIMODAL_VELOCITY_SUPPORTED        → V-RFM 方法验证

并行长期线：
Stage 1  独立几何能量与合成控制
    ↓
Stage 2  固定实例的局部弱可辨识性
    ↓
Stage 3  全局候选、路径与 minimax 能垒
    ↓
Stage 4  跨实例 prevalence 与最终模型选择
```

这条路线把“是否需要 V-RFM”的直接证据提前到 V0，同时保留完整物理解空间研究。
两条线相互校准，但不互相绑架。

---

## 9. 当前结论

目前最稳妥的判断是：

1. 现有 long-vs-short 关系很弱，不能证明固定观测下的物理多解；
2. 多个 overlap short windows 的确提供了一个自然的多候选来源；
3. 候选是否构成多模态修复速度，取决于双端有效性、方向分离和内部插值；
4. Diffusion 可以表示多峰分布，V-RFM 的特殊价值在于直接建模 data–time space 的
   多模态速度，而不是“只有它能处理离散分支”；
5. 若 V0 显示平均安全，就先做确定性融合；若只有一个候选有效，就先做 selector；
   只有稳定出现“端点好、平均差”，才进入 V-RFM。

这也是三份文档之间的逻辑顺序：理论基础定义什么算证据，前置实验决定证据属于哪
一类，方法设计只在对应门槛成立时被激活。

---

# 数学附录

## 附录 A：状态、观测映射与 profile energy

设完整状态为

$$
x=(T,K,Z)\in\mathcal M,
$$

其中 $T\in SE(3)^N$ 是相机轨迹，$K$ 是内参，$Z$ 汇总场景几何、深度、对应关系、
动态 mask 等 nuisance variables。理想观测映射记为

$$
F:\mathcal M\rightarrow\mathcal Y.
$$

给定观测 $y$，exact fiber 是 $F^{-1}(y)$。若存在 gauge 群 $G$ 作用于状态空间且
$F(g\cdot x)=F(x)$，物理对象应写为 $F^{-1}(y)/G$。

实验能量一般写成

$$
E(C,T,K,Z)
=\sum_r w_r\,\rho_r\!\left(e_r(C,T,K,Z)\right)+R(T,K,Z),
$$

并通过

$$
\bar E(C,T,K)=\inf_Z E(C,T,K,Z)
$$

消去 nuisance variables。若不同候选使用不同的 $Z$ 优化策略，就不再是同一个
profile energy，候选间能量不可直接比较。

## 附录 B：Gauge、商空间与距离

选择一个与传感器协议相符的 gauge 群 $G$。轨迹等价类记为 $[T]$，商空间距离可写为

$$
d_Q([T_1],[T_2])
=\inf_{g\in G} d_{\mathcal T}(T_1,g\cdot T_2).
$$

$d_{\mathcal T}$ 应在合法的 $SE(3)$ 表示上定义，并明确 rotation/translation 的权重、
时间采样和尺度处理。四元数 $q$ 与 $-q$、Euler angle 周期、raw quaternion 未归一化
都是表示问题，不应被聚类成物理 component。VGGT 的 `pose_enc_list` 是同一次确定性
Camera Head 的迭代状态，也不是四个 posterior samples。

## 附录 C：Exact fiber 的局部维数

若 $F$ 在 $x^*$ 附近光滑且 Jacobian 秩局部恒为 $r$，constant-rank theorem 给出

$$
\dim F^{-1}(F(x^*))=\dim\mathcal M-r
$$

的局部结论。进一步除去局部自由作用的 gauge 维数后，才得到物理 exact fiber 的
局部维数。

限制条件不能省略：秩必须在邻域稳定；奇点处的 null space 维数可能突然变化；一阶
null direction 不保证能积分成长距离可行曲线；若硬约束把状态空间切成带边界或分层
集合，普通流形版本需替换为相应的约束分析。

## 附录 D：为什么正容差集合通常是“厚”的

令

$$
\mathcal S_h=\{q\in\mathcal Q:\bar E(q)\le h\}.
$$

若 $\bar E$ 连续且存在严格可行点 $q_0$ 满足 $\bar E(q_0)<h$，则由连续性，$q_0$
附近存在一个开邻域仍满足 $\bar E<h$。因此 $\mathcal S_h$ 在局部含有开集，通常是
满维厚集合，而不是低维流形。

若物理域在所选 gauge 固定后是闭的，$\bar E$ 下半连续，且其子水平集有界，则
$\mathcal S_h$ 紧；连续能量在紧集上取得最小值。这些条件需要实验逐项验证，不能只
因为优化器停止就声称全局最优存在并已找到。

## 附录 E：直径、道路分量、minimax 高度与弱方向

### E.1 商空间直径

$$
D_h=\sup_{q_1,q_2\in\mathcal S_h}d_Q(q_1,q_2).
$$

小 $D_h$ 支持“所有可接受轨迹都很近”，大 $D_h$ 反证近似唯一，但不说明集合是否
连通。

### E.2 道路连通分量

$\beta_0(\mathcal S_h)$ 是道路连通分量数。有限采样的聚类数不是 $\beta_0$；采样
空洞可能只是漏采，也可能被低能窄桥连接。

### E.3 Minimax 高度

两候选 $q_a,q_b$ 间定义

$$
H(q_a,q_b)
=\inf_{\gamma(0)=q_a,\gamma(1)=q_b}
\max_{s\in[0,1]}\bar E(\gamma(s)).
$$

若找到一条完整路径且其最大能量不超过 $h$，即可证明这对候选在 $\mathcal S_h$
中同分量。反过来，路径搜索失败只给出一个数值上界缺失，不能作为断开证明；声称
断开需要可靠 barrier 下界或全局拓扑证书。

### E.4 局部弱可辨识维数

在去 gauge 的切空间中计算 generalized Hessian 或 Jacobian spectrum，按预注册阈值
$\tau$ 统计小曲率方向：

$$
k_{\mathrm{weak}}(\tau)
=\#\{j:\lambda_j\le\tau\}.
$$

它描述局部弱方向，不等于全局连续解维数。需要 perturb–refit、continuation 和邻域
秩稳定性检查，才能判断一阶弱方向能否积成更长路径。

## 附录 F：最小反例

1. **Gauge copies**：两条 raw 轨迹相距很远，但对齐后完全一致；输出双簇并非物理
   双解。
2. **连通香蕉形集合**：有限采样在两端形成两簇，中间窄桥未采到；聚类数大于真实
   道路分量数。
3. **双 basin、单分量**：优化器被参数化或曲率困住，但存在合法低能路径连接两个
   停点。
4. **双端有效、均值无效**：这是速度多模态的候选证据，但仍可能来自错误配对或
   label noise，必须跨 scene 和负控制复现。
5. **一个端点有效、一个无效**：输出差异明显，却只是 selector problem。
6. **双端和均值都有效**：存在连续可融合冗余，不需要为了“有多个输出”强上 latent。

---

## 参考文献

- Wang et al. *VGGT: Visual Geometry Grounded Transformer*. CVPR, 2025.
- Hartley and Zisserman. *Multiple View Geometry in Computer Vision*. 2nd ed., 2004.
- Triggs et al. *Bundle Adjustment—A Modern Synthesis*. 2000.
- Lipman et al. *Flow Matching for Generative Modeling*. ICLR, 2023.
- Liu, Gong, and Liu. *Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow*. ICLR, 2023.
- Guo and Schwing. *Variational Rectified Flow Matching*. ICML, 2025.
- Ho et al. *Denoising Diffusion Probabilistic Models*. NeurIPS, 2020.
- Song et al. *Score-Based Generative Modeling through Stochastic Differential Equations*. ICLR, 2021.
- Lee. *Introduction to Smooth Manifolds*. 2nd ed., 2013.
- Edelsbrunner and Harer. *Computational Topology: An Introduction*. 2010.
