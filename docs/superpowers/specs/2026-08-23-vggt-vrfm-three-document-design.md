# VGGT 多短窗修复与 V-RFM 三文档设计

> 日期：2026-08-23  
> 文档分支：`codex/camera_solution_space_docs_reframe`  
> 理论基线：`codex/camera_solution_space_01_theory_foundation@cc1d8ac`  
> 状态：已批准的文档架构；尚未改写或生成正式 PDF

## 1. 目标

将现有“固定观测下相机轨迹解空间”理论材料，重新组织成三份边界清楚、面向不同问题的 PDF：

1. **理论基础 PDF**：解释什么才算相机轨迹歧义，什么只是 gauge、预测误差或算法输出差异。
2. **前置实验 PDF**：详细检验多个重叠短窗口是否为同一长序列提供多个有效且不可安全平均的修复方向。
3. **方法设计 PDF**：用通俗方式说明，如果前置实验成立，如何把 Variational Rectified Flow Matching 用于 VGGT 相机轨迹修复。

三份文档依次回答：

```text
什么算真正的歧义？
        ↓
VGGT 中是否存在需要建模的修复速度歧义？
        ↓
如果存在，V-RFM 如何利用这种歧义？
```

## 2. 目标读者与语言要求

默认读者了解 VGGT、相机位姿或三维视觉，但不要求了解 Flow Matching、拓扑学、变分推断或微分几何。

所有文档遵循以下顺序：

1. 先给出 VGGT 的具体场景；
2. 再解释直觉；
3. 最后给出必要公式；
4. 严格证明、技术条件和反例放入附录。

全文固定使用以下术语：

- `长序列预测`：一次完整长上下文 VGGT camera prediction；
- `短窗口预测`：从长序列切出的重叠局部窗口预测；
- `修复方向`：短窗口相对于长序列 camera center 的候选残差；
- `有效候选`：通过冻结评价协议的修复候选；
- `平均修复`：多个候选残差的确定性均值或加权均值；
- `序列级 latent`：一次完整轨迹生成只采样一次并保持不变的 V-RFM 隐变量。

抽象术语首次出现时必须紧跟普通语言解释。例如，`gauge` 首次出现时解释为“同一物理轨迹采用不同全局坐标系后的表示差异”。不得只写 `fiber`、`quotient`、`minimax barrier` 或 `posterior mode` 而不给 VGGT 对应例子。

三份文档统一使用以下贯穿示例：

```text
一条 500 帧长序列
→ 九个长度 100、stride 50 的短窗口
→ 相邻窗口在 50 帧重叠区给出两份修复建议
→ 判断它们是一致、只有一个正确、连续可融合，还是形成多个有效方向
```

## 3. 三份文档的职责边界

### 3.1 理论基础 PDF

建议标题：

> **从相机轨迹多解到修复速度歧义：VGGT 长短上下文修复的理论基础**

这份文档保留现有理论工作的严谨部分，但不再把“固定 RGB-D 观测下的解空间拓扑”当作 V-RFM 唯一的前置问题。

正文职责：

- 区分物理多解、gauge 冗余、局部弱约束、优化 basin、算法输出多样性和概率多模态；
- 解释为什么 100 帧与 500 帧结果不同不能证明同一观测存在多个物理解；
- 同时解释为什么多个短窗口仍可能产生多个候选修复速度；
- 区分“最终轨迹多模态”与“同一数据—时间状态上的速度多模态”；
- 解释普通 Rectified Flow 的条件均值速度和 V-RFM 的 latent-conditioned velocity；
- 给出进入前置实验和方法设计的证据边界。

现有理论 PDF 必须修改：

1. **执行摘要**：并列写出“物理解空间”和“修复速度分布”两条问题线，并声明 V-RFM 主要对应后者。
2. **旧证据审计**：保留“long-vs-short 不能证明固定观测物理多解”，补充“它可以生成候选修复方向”。
3. **MSE / Diffusion / Flow Matching 章节**：新增 V-RFM，解释普通 RF 为什么平均速度、V-RFM 为什么允许不同 `z` 下的轨迹相交。
4. **模型门槛**：删除“只有认证断开分量才值得引入 latent”的过强要求。改为：多个有效速度方向且平均方向造成损失，已经构成 V-RFM 的直接动机。
5. **研究路线**：在完整解空间拓扑研究之前加入 `V0：多短窗修复速度歧义诊断`。
6. **结论语言**：不得把速度多模态写成多个物理真解，也不得把多个短窗口输出直接写成有效多解。

正文只保留理解上述关系需要的符号。现有 constant-rank、子水平集满维性、紧性、minimax 下界和严格连通性讨论移入数学附录，不删除。

### 3.2 前置实验 PDF

建议标题：

> **多个短上下文 VGGT 预测是否产生多模态相机修复速度？**  
> *Do Overlapping Short-Context VGGT Predictions Induce Multimodal Camera-Repair Velocities?*

这是三份文档中最详细的一份，承担可执行、可复现的实验协议和结果报告。它不介绍完整 V-RFM 网络。

正文目录：

1. **问题动机**：用 500/100 帧例子解释同一重叠帧为何有两个修复建议。
2. **需要排除的误解**：多个输出不等于多个有效方向；短窗误差不等于多模态。
3. **数据与已有证据**：列出实际 branch、run ID、artifact、场景和 frame identity。
4. **实验单位**：`(scene, left window, right window, shared 50 frames)`。
5. **候选构造**：只用预测值将两个短窗口对齐到 global prediction gauge，得到 `d_L` 和 `d_R`。
6. **固定评价**：global-to-GT Sim(3) 每场景只拟合一次，随后冻结给全部候选；GT 只做 privileged offline diagnosis。
7. **插值检查**：评价 `d(alpha)=(1-alpha)d_L+alpha d_R`，至少使用 `alpha={0,0.25,0.5,0.75,1}`。
8. **方向与有效性指标**：cosine、归一化 RMS separation、逐帧一致率、alignment residual、translation error、relative translation error。
9. **四类结果**：`NOT_SUPPORTED`、`SELECTOR_PROBLEM`、`CONTINUOUS_REDUNDANCY`、`MULTIMODAL_VELOCITY_SUPPORTED`。
10. **负控制**：self-pair、gauge copy、随机错误窗口、残差取反、小扰动和退化对齐。
11. **统计协议**：calibration 冻结阈值，scene-level bootstrap，已查看的旧 40 scenes 只称 development evaluation。
12. **GO / NO-GO**：只有双端有效、方向分离且平均/内部插值变差的事件跨场景稳定，才进入 V-RFM。
13. **结果与失败案例**：每个结论配一条真实轨迹和一幅重叠窗口示意图。
14. **复现信息**：代码 commit、输入 manifest、split、run ID、命令和输出 hash。

该 PDF 跟随正在执行的前置实验分支更新。结果出来前使用明确的 `protocol in progress; no scientific conclusion yet` 状态，不预填正面结论。

### 3.3 方法设计 PDF

建议标题：

> **用 Variational Rectified Flow 融合 VGGT 多短窗修复方向**

这是 6–10 页的概念设计与讲解文档。它服务于组会、合作者沟通和方法早期评审，不承担代码级实现合同。

正文目录：

1. **实际问题**：500 帧整体预测漂移，多个 100 帧窗口对重叠区域给出不同建议。
2. **为什么不能提前平均**：用“向左、向右、平均后不动”的一维例子解释。
3. **普通 Rectified Flow**：同一 `(x_t,t,condition)` 只有一个确定性速度，MSE 学条件均值。
4. **V-RFM 的变化**：加入 `v_theta(x_t,t,condition,z)`，让 `z` 区分修复方向。
5. **VGGT 条件输入**：global trajectory、全部 local trajectories、覆盖关系、alignment confidence、camera features；不得先融合成单一 local input。
6. **输出**：仅生成 camera-center residual；rotation 和 FoV 第一版保持 global 输出。
7. **序列级 latent**：一条长序列只采样一次，避免窗口或帧之间切换修复策略。
8. **训练与推理**：训练 posterior 可以看目标残差，推理只从 prior 采样；说明 posterior collapse 风险。
9. **候选选择问题**：V-RFM 负责生成，不自动判断哪个短窗口正确；需要独立 scorer、几何能量或下游选择。
10. **基线与消融**：deterministic fusion、deterministic Transformer、普通 conditional RF、V-RFM、no-camera-feature。
11. **进入条件**：前置实验为 `GO` 才正式实现；否则优先解决 selector 或确定性融合。

方法 PDF 只使用一个核心损失公式和一个训练/推理流程图。网络宽度、层数、优化器和数据工程细节不进入正文，只在未来实现附录中记录。

## 4. 三份文档的相互引用

- 理论 PDF 引用前置实验 PDF 作为 V0 的操作化协议，不引用尚未产生的正面结果。
- 前置实验 PDF 引用理论 PDF 中的术语边界和结论规则，但在正文重复必要的通俗定义。
- 方法 PDF 引用前置实验的最终决策卡；结果未完成时写成条件设计，不暗示方法必要性已经成立。
- 三份 PDF 共用一份文献库，至少加入 Guo & Schwing 的 ICML 2025 V-RFM 论文以及 Rectified Flow / Flow Matching 的原始文献。

## 5. 图形设计

三份文档统一视觉语言，但图的职责不同：

- 理论 PDF：一张“六种不同歧义”的关系图；
- 前置实验 PDF：一张真实 500/100 overlap 图、一张 residual interpolation 曲线、一张四类决策图；
- 方法 PDF：一张普通 RF 与 V-RFM 对比图、一张完整 VGGT-V-RFM 训练/推理流程图。

所有图先展示具体轨迹或窗口，再展示抽象变量。颜色固定：global 为深蓝，left/right short windows 为橙/绿，平均修复为灰色，V-RFM samples 为紫色系。图注必须能脱离正文独立理解。

## 6. 文件布局

正式源文件建议放置为：

```text
doc/camera_solution_space_01_theory_foundation/
  camera_trajectory_solution_space.md
  camera_trajectory_solution_space.tex

doc/camera_velocity_ambiguity_preexperiment/
  camera_velocity_ambiguity_preexperiment.md
  camera_velocity_ambiguity_preexperiment.tex

doc/variational_rectified_camera_refiner/
  variational_rectified_camera_refiner_method.md
  variational_rectified_camera_refiner_method.tex

doc/references/
  camera_refiner_references.bib
```

生成 PDF 写入 ignored 的 `output/pdf/`，不直接提交编译缓存。是否提交 PDF 本体沿用当前理论分支做法；若提交，必须同时提交对应 source commit 和可复现构建命令。

## 7. 构建与视觉验证

实施阶段必须使用 PDF skill 的 render-and-verify 流程：

1. 从 Markdown/LaTeX 生成 PDF；
2. 用 Poppler 将全部页面渲染为 PNG；
3. 检查公式、中文字体、表格、图注、分页、交叉引用和超链接；
4. 对每份 PDF 至少检查第一页、目录页、所有含图页面和最后一页；
5. 修复所有溢出、孤行、空白页和不可读字号后再交付。

内容验证包括：

- 三份文档不互相矛盾；
- 没有把多个窗口输出写成多个有效解；
- 没有把 V-RFM 写成窗口选择器；
- 没有声称前置实验尚未得到的结果；
- 所有 VGGT 数字、分支和 run ID 都有 provenance；
- 方法 PDF 的每个抽象符号都有 VGGT 对应物。

## 8. 分支与并行工作边界

- 本文档重构在 `codex/camera_solution_space_docs_reframe` 独立完成。
- 正在执行的前置实验分支拥有实验代码、配置、运行结果和详细协议的事实来源。
- 文档分支不得修改前置实验代码或正在生成的 artifact。
- 合并前只从前置实验分支读取已提交、带 provenance 的事实；不复制未提交草稿或推测性结果。
- 理论 PDF 和方法 PDF 可先完成无结果版本；前置实验 PDF 的结果章节等待实验分支给出冻结结果卡。

## 9. 非目标

本轮不做以下工作：

- 实现或训练 V-RFM；
- 重跑 VGGT 或下载新数据；
- 把现有 8 帧 RGB-D 解空间实验包装成 V-RFM 证据；
- 承诺 V-RFM 一定优于 deterministic refiner；
- 为了讲解简单而删除必要的证据边界。

## 10. 验收标准

文档重构完成时应满足：

1. 不熟悉 Flow Matching 的 VGGT 研究者能在方法 PDF 前三页理解问题与核心方法；
2. 前置实验 PDF 足够详细，使另一位研究者无需口头补充即可复现协议；
3. 理论 PDF 保留严谨边界，同时正文不要求读者先掌握拓扑学；
4. 三份文档对“多解”“速度多模态”“有效候选”和“物理真解”的用词一致；
5. 每份 PDF 都通过完整页面渲染检查；
6. 前置实验未结束时，任何文档都不提前声称 V-RFM 的必要性。
