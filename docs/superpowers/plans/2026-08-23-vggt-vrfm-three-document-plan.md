# VGGT 多短窗修复与 V-RFM 三文档 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task, and use `pdf:pdf` for the build/render/visual-verification tasks.

**Goal:** 将当前单份“固定观测相机解空间”材料重构为三份互相衔接、语言通俗且证据边界严格的 PDF：理论基础、前置实验协议、V-RFM 方法设计。

**Architecture:** Markdown 是内容评审主稿，LaTeX 是正式排版源；三份文档共享一套 LaTeX 视觉样式和一份 BibTeX 文献库。一个跨平台 Python 构建入口负责 XeLaTeX/BibTeX 编译和 Poppler 页面渲染，一组轻量 `unittest` 负责文件布局、关键术语、状态声明、交叉引用和“不得提前下结论”的内容契约。前置实验分支仍是实验事实的唯一来源，本分支只读取已经提交且带 provenance 的信息。

**Tech Stack:** Chinese Markdown、XeLaTeX/ctex、TikZ、BibTeX/natbib、Python 3.10+ `unittest`/`subprocess`、Poppler (`pdfinfo`/`pdftoppm`/`pdftotext`)

**Spec:** `docs/superpowers/specs/2026-08-23-vggt-vrfm-three-document-design.md`

## Global Constraints

- 只在 `codex/camera_solution_space_docs_reframe` 和其独立 worktree 中修改文档；不修改正在运行的前置实验代码、配置或 artifact。
- H20 上 `/home/ubuntu/yjh/vggt` 是实验事实来源。只可通过只读命令检查已提交 commit、manifest、run ID 和结果卡；不得把未提交内容写进文档，也不得发起实质性 H20 → 本地传输。
- 实验未冻结前，三份文档统一写明 `protocol in progress; no scientific conclusion yet`，不得暗示已经观察到多模态修复速度或 V-RFM 已有必要性。
- “多个短窗口输出”“多个有效修复方向”“修复速度多模态”“多个物理真解”始终是四个不同命题。
- 统一示例固定为 500 帧长序列、长度 100/stride 50 的九个短窗口、相邻窗口共享 50 帧。
- 统一颜色：global 深蓝、left/right short windows 橙/绿、平均修复灰、V-RFM samples 紫。
- 正文先讲 VGGT 场景和直觉，再给公式；严谨拓扑和微分几何内容保留在理论附录，不删除。
- 生成的 PDF、LaTeX 缓存和渲染 PNG 只进入 ignored 的 `output/` 或 `tmp/`，不提交。
- 所有新增引用先用论文或正式会议页面核验作者、标题、年份和 venue；技术论断优先引用原始论文。

---

## Task 1: 用失败测试冻结三文档内容契约

**Files:**

- Create: `tests/documentation/__init__.py`
- Create: `tests/documentation/test_vggt_vrfm_documents.py`

**Step 1: 写出当前必然失败的布局与语义测试**

测试至少定义以下常量和检查：

```python
DOCS = {
    "theory": ROOT / "doc/camera_solution_space_01_theory_foundation/camera_trajectory_solution_space.md",
    "experiment": ROOT / "doc/camera_velocity_ambiguity_preexperiment/camera_velocity_ambiguity_preexperiment.md",
    "method": ROOT / "doc/variational_rectified_camera_refiner/variational_rectified_camera_refiner_method.md",
}
STATUS = "protocol in progress; no scientific conclusion yet"
DECISIONS = (
    "NOT_SUPPORTED",
    "SELECTOR_PROBLEM",
    "CONTINUOUS_REDUNDANCY",
    "MULTIMODAL_VELOCITY_SUPPORTED",
)
```

增加以下测试：

- 三份 `.md`、三份 `.tex`、共享 `.bib` 和共享 `.sty` 均存在；
- 理论文档同时包含“物理解空间”“修复速度歧义”“Variational Rectified Flow”和 `V0`；
- 前置实验包含状态声明、500/100/50 贯穿示例、`d_L`/`d_R`、固定 global-to-GT Sim(3)、五个 `alpha` 和四类决策标签；
- 方法文档包含 sequence-level latent、prior/posterior、camera-center residual、selector/scorer 边界和四个主要 baseline；
- 三个 `.tex` 均加载共享 style 和共享 bibliography；
- 源文件中不得出现“多个短窗口证明多个物理真解”“实验已经证明 V-RFM 必要”“V-RFM 一定优于”等提前结论。

测试只约束稳定事实和安全边界，不要求逐字复制整段正文，以免未来正常编辑造成脆弱失败。

**Step 2: 运行测试，确认失败原因正是缺少新文档**

Run:

```powershell
python -m unittest tests.documentation.test_vggt_vrfm_documents -v
```

Expected: `FAIL`，明确列出缺少 experiment/method `.md`、`.tex`、共享 `.bib` 或 `.sty`；不得因 import 或路径错误中断。

**Step 3: 提交内容契约**

```powershell
git add tests/documentation
git commit -m "test: define VGGT V-RFM document contract"
```

---

## Task 2: 建立共享文献库与 LaTeX 视觉层

**Files:**

- Create: `doc/references/camera_refiner_references.bib`
- Create: `doc/references/vggt_vrfm_report.sty`
- Modify: `doc/camera_solution_space_01_theory_foundation/camera_trajectory_solution_space.tex`
- Delete: `doc/camera_solution_space_01_theory_foundation/references.bib`

**Step 1: 核验并汇总原始文献**

将现有 VGGT、多视几何、Diffusion、Score、Flow Matching 引用移入共享 `.bib`，再从正式论文页面核验并加入：

- Rectified Flow 原始/正式论文；
- Guo & Schwing 的 Variational Rectified Flow Matching 正式条目；
- 如正文实际使用，再加入变分推断或条件生成的原始来源。

不要从博客或二手综述复制 BibTeX。删除旧局部 `.bib` 前，逐项确认 key 已在共享库中。

**Step 2: 提取公共排版定义**

在 `vggt_vrfm_report.sty` 中集中定义：

```tex
\definecolor{GlobalBlue}{HTML}{17365D}
\definecolor{LeftOrange}{HTML}{D97904}
\definecolor{RightGreen}{HTML}{3B7D44}
\definecolor{MeanGray}{HTML}{6B7280}
\definecolor{VRFMPurple}{HTML}{7E57C2}
```

同时集中设置页面边距、标题/页眉、列表间距、超链接颜色、`statusbox`、`keybox`、`warningbox` 和 TikZ 公共库。保留各文档自己的 `pdftitle`、页眉副标题和数学宏。

**Step 3: 让现有理论 TeX 使用共享资源**

把重复的 package/颜色/box 定义替换为：

```tex
\usepackage{doc/references/vggt_vrfm_report}
...
\bibliography{doc/references/camera_refiner_references}
```

此步只做基础设施迁移，不改理论正文。

**Step 4: 先编译旧理论文档，排除样式迁移回归**

Run:

```powershell
xelatex -interaction=nonstopmode -halt-on-error -output-directory=tmp/pdfs doc/camera_solution_space_01_theory_foundation/camera_trajectory_solution_space.tex
bibtex tmp/pdfs/camera_trajectory_solution_space
xelatex -interaction=nonstopmode -halt-on-error -output-directory=tmp/pdfs doc/camera_solution_space_01_theory_foundation/camera_trajectory_solution_space.tex
xelatex -interaction=nonstopmode -halt-on-error -output-directory=tmp/pdfs doc/camera_solution_space_01_theory_foundation/camera_trajectory_solution_space.tex
```

Expected: exit code 0；log 中无 `Citation ... undefined`、`Reference ... undefined`。

**Step 5: 提交共享资源**

```powershell
git add doc/references doc/camera_solution_space_01_theory_foundation
git commit -m "docs: share report style and camera refiner references"
```

---

## Task 3: 增加可复现的三文档构建与文本检查入口

**Files:**

- Create: `scripts/docs/build_vggt_vrfm_pdfs.py`
- Create: `scripts/docs/check_vggt_vrfm_pdf_text.py`
- Modify: `tests/documentation/test_vggt_vrfm_documents.py`

**Step 1: 先为构建 manifest 增加失败测试**

测试导入 `DOCUMENTS`，确认三个条目的 TeX 输入和 PDF 输出分别为：

```text
camera_trajectory_solution_space.tex -> camera_trajectory_solution_space_theory.pdf
camera_velocity_ambiguity_preexperiment.tex -> camera_velocity_ambiguity_preexperiment.pdf
variational_rectified_camera_refiner_method.tex -> variational_rectified_camera_refiner_method.pdf
```

Run: `python -m unittest tests.documentation.test_vggt_vrfm_documents -v`

Expected: `ERROR` 或 `FAIL`，因为构建模块尚不存在。

**Step 2: 实现跨平台构建脚本**

脚本应：

1. 从仓库根目录解析所有路径，不依赖当前 shell 目录；
2. 为每份文档运行 XeLaTeX → BibTeX → XeLaTeX ×2；
3. 编译缓存写入 `tmp/pdfs/<document-key>/`；
4. PDF 复制到 `output/pdf/`；
5. `--render` 时调用 `pdftoppm -png -r 144`，输出到 `tmp/pdf-renders/<document-key>/`；
6. `--document theory|experiment|method|all` 支持单文档和全量构建；
7. 任一子命令非零时原样失败，并打印正在构建的文档 key，不吞掉 TeX 错误。

**Step 3: 实现 PDF 文本安全检查**

`check_vggt_vrfm_pdf_text.py` 使用 `pdftotext` 提取三份 PDF，检查：

- 文件可提取且不是空白 PDF；
- 状态声明在前置实验和方法文档中可见；
- 不出现 `??`、`Citation ... undefined` 的可见残留；
- 四类决策标签在前置实验 PDF 中完整出现；
- 方法 PDF 中能搜到 `sequence-level latent` 或中文等价词。

**Step 4: 运行单元测试**

Run:

```powershell
python -m unittest tests.documentation.test_vggt_vrfm_documents -v
```

Expected: manifest 测试通过；文档存在性测试仍因 experiment/method 源缺失而失败。

**Step 5: 提交构建工具**

```powershell
git add scripts/docs tests/documentation
git commit -m "build: add reproducible research PDF pipeline"
```

---

## Task 4: 重写理论基础 Markdown 的正文层级

**Files:**

- Modify: `doc/camera_solution_space_01_theory_foundation/camera_trajectory_solution_space.md`

**Step 1: 替换标题、状态与执行摘要**

标题改为：

```text
从相机轨迹多解到修复速度歧义：VGGT 长短上下文修复的理论基础
```

摘要并列给出两条问题线：

- 物理解空间：固定完整观测和评价后，去 gauge 的可接受相机轨迹集合是什么形状；
- 修复速度分布：给定 long/global prediction 与全部 local-window evidence，是否存在多个有效且不可安全平均的修复方向。

明确 V-RFM 直接对应第二条线，第一条线提供概念边界但不是必须先完成的拓扑认证。

**Step 2: 用 500/100/50 示例重写正文入口**

在任何抽象集合定义前，先定义：

```math
d_k(i)=\widetilde c^{(k)}_i-c^G_i,
```

其中 `c^G_i` 是 500 帧 global camera center，`\widetilde c^{(k)}_i` 是只用预测值对齐到 global gauge 的第 `k` 个 100 帧窗口结果。解释相邻窗口在共享 50 帧上为什么会给出 `d_L`、`d_R` 两个候选，但此时还不能称它们“都正确”。

**Step 3: 把六种易混概念做成正文核心表**

依次解释并给 VGGT 例子：物理多解、gauge 冗余、局部弱约束、优化 basin、算法输出差异、概率/速度多模态。每行必须包含“它是什么”“VGGT 中的例子”“它不能推出什么”。

**Step 4: 重写 MSE / RF / V-RFM 解释**

正文只保留一条核心逻辑：在完全相同的 `(x_t,t,condition)` 下，用 MSE 训练的确定性速度场倾向条件均值；若有效目标速度有多个互相冲突的方向，均值可能落在差的内部区域。V-RFM 用序列级 `z` 条件化速度，使相同可见状态可对应不同 latent-conditioned velocity。

同时明确：

- Diffusion 既能表示连续也能表示离散/多峰分布，并非“不能处理离散分支”；
- 本项目选择 V-RFM 的理由不是 Diffusion 的表达能力不足，而是更直接地研究速度歧义；
- 多个窗口是 candidate generator，不自动提供真实 posterior 样本；
- V-RFM 是 generator，不是窗口正确性 selector。

**Step 5: 插入 V0，并改写模型门槛**

在原 Stage 1 前加入：

```text
V0：多短窗修复速度歧义诊断
```

门槛改为：如果多个候选在冻结评价下都有效、方向显著分离，且其均值或内部插值显著更差，就足以直接研究 latent-conditioned velocity；不要求先证明物理解空间有认证的断开分量。

**Step 6: 将严谨数学内容移入附录但不删减**

正文保留必要的 gauge、有效候选、条件速度概念；把 constant-rank、正容差子水平集满维性、闭性/紧性、minimax 下界与连通性证书整理为附录 A–D。保持原引用和论证限定词。

**Step 7: 运行 Markdown 内容契约**

Run: `python -m unittest tests.documentation.test_vggt_vrfm_documents -v`

Expected: theory 相关断言通过；experiment/method 仍因文件缺失失败。

**Step 8: 提交理论主稿**

```powershell
git add doc/camera_solution_space_01_theory_foundation/camera_trajectory_solution_space.md
git commit -m "docs: reframe theory around VGGT repair velocities"
```

---

## Task 5: 同步理论 LaTeX 并完成理论 PDF

**Files:**

- Modify: `doc/camera_solution_space_01_theory_foundation/camera_trajectory_solution_space.tex`
- Modify: `doc/camera_solution_space_01_theory_foundation/README.md`

**Step 1: 让 LaTeX 章节顺序与 Markdown 一致**

每个 Markdown 一级/二级核心章节在 TeX 中有对应 section/subsection。TeX 可为分页拆段，但不得改变结论或偷偷增加实验事实。

**Step 2: 绘制“六种不同歧义”关系图**

用 TikZ 先画具体的 global/left/right 轨迹，再连到六类概念卡片。图注明确“窗口输出差异只位于算法输出层；通过有效性与插值检验后，才可能支持修复速度多模态”。

**Step 3: 更新理论 README**

README 改写为：该目录是三文档中的第一份；列出新标题、共享资源路径、单文档构建命令和当前证据状态。删除已经过时的父分支/旧阶段描述。

**Step 4: 构建并检查理论 PDF**

Run:

```powershell
python scripts/docs/build_vggt_vrfm_pdfs.py --document theory --render
python scripts/docs/check_vggt_vrfm_pdf_text.py --document theory
```

Expected: 两条命令 exit code 0；生成 `output/pdf/camera_trajectory_solution_space_theory.pdf` 和逐页 PNG。

**Step 5: 用 PDF skill 做视觉检查**

至少检查封面、目录、“六种歧义”图页、V-RFM 解释页、V0 路线页和最后一页。修复 overfull box、中文字体缺失、孤行、图注跨页、空白页和不可读字号后，重新完整构建。

**Step 6: 提交理论排版稿**

```powershell
git add doc/camera_solution_space_01_theory_foundation
git commit -m "docs: typeset VGGT repair velocity theory report"
```

---

## Task 6: 编写详细前置实验 Markdown

**Files:**

- Create: `doc/camera_velocity_ambiguity_preexperiment/README.md`
- Create: `doc/camera_velocity_ambiguity_preexperiment/camera_velocity_ambiguity_preexperiment.md`

**Step 1: 只读核对实验 provenance**

在 H20 上执行只读检查：

```bash
ssh h20 "cd /home/ubuntu/yjh/vggt && git branch --show-current && git rev-parse HEAD && git status --short"
```

只把已提交且能由 commit 定位的 branch、commit、manifest、run ID 和 artifact hash 写入“已冻结事实”表。若结果卡尚未提交，则保留明确占位符，并写状态声明；不得抄入 shell 临时输出、未提交结果或口头推测。

**Step 2: 写清实验单位和候选构造**

正文依次定义：

- 一条 500 帧 scene；
- 九个 `(start, start+99)` 窗口，start 为 `0,50,...,400`；
- 实验单位 `(scene,left window,right window,shared 50 frames)`；
- 只用预测值估计 local-to-global 对齐；
- 共享帧上的 `d_L`、`d_R`；
- GT 不参与候选产生与 local-to-global 对齐。

**Step 3: 写清冻结评价与泄漏边界**

说明每个 scene 的 global-to-GT Sim(3) 只拟合一次并冻结，随后同一个变换用于 global、两个端点、均值和全部插值点。GT 只用于 privileged offline diagnosis；正式部署不能依赖这一 scorer。

**Step 4: 写出指标与插值协议**

固定插值网格：

```math
d(\alpha)=(1-\alpha)d_L+\alpha d_R,
\quad \alpha\in\{0,0.25,0.5,0.75,1\}.
```

逐项解释 cosine、归一化 RMS separation、逐帧方向一致率、alignment residual、translation error 和 relative translation error。所有阈值标为“由 calibration split 冻结”，结果前不得事后填阈值。

**Step 5: 把四类决策写成互斥的操作规则**

- `NOT_SUPPORTED`：候选不稳定、不分离，或没有双端有效证据；
- `SELECTOR_PROBLEM`：一个端点有效而另一个无效，主要任务是选对候选；
- `CONTINUOUS_REDUNDANCY`：双端有效且内部插值/平均也保持有效，没有坏的内部区域；
- `MULTIMODAL_VELOCITY_SUPPORTED`：双端有效、方向分离、平均或预注册内部插值显著变差，并且该事件在 scene-level bootstrap 下稳定。

说明最后一类只支持“修复速度分布值得 latent 建模”，不支持“存在多个物理真轨迹”。

**Step 6: 补齐负控制、统计和停止规则**

覆盖 self-pair、gauge copy、随机错误窗口、残差取反、小扰动、退化对齐；明确 calibration/evaluation 分离、旧 40 scenes 仅作 development evaluation、scene-level bootstrap、失败样本也必须报告。

**Step 7: 创建结果卡和复现模板**

结果章节在实验结束前只包含：

```text
protocol in progress; no scientific conclusion yet
```

并预留固定字段：branch、commit、input manifest hash、split、run ID、command、artifact path、artifact hash、样本数、四类比例、bootstrap CI、代表成功/失败案例。README 写明这些字段只能从冻结结果卡更新。

**Step 8: 运行内容契约并提交**

Run: `python -m unittest tests.documentation.test_vggt_vrfm_documents -v`

Expected: experiment Markdown 检查通过；method 文件相关检查仍失败。

```powershell
git add doc/camera_velocity_ambiguity_preexperiment
git commit -m "docs: specify VGGT repair velocity pre-experiment"
```

---

## Task 7: 排版前置实验 PDF 与三张解释图

**Files:**

- Create: `doc/camera_velocity_ambiguity_preexperiment/camera_velocity_ambiguity_preexperiment.tex`

**Step 1: 同步 Markdown 的完整实验结构**

TeX 必须包含动机、误解排除、provenance、实验单位、候选构造、固定评价、插值、指标、四类决策、控制、统计、GO/NO-GO、结果状态、失败案例模板和复现信息。

**Step 2: 绘制 500/100 overlap 图**

用真实协议帧号绘制 500 帧 global bar、九个 100 帧窗口和相邻 50 帧 overlap。颜色遵守全局约定，图注独立说明为什么共享帧会收到两份候选修复。

**Step 3: 绘制 residual interpolation 阅读图**

画 `alpha` 横轴、冻结误差指标纵轴、两端点和均值位置。实验结果未冻结时，曲线必须醒目标注“示意图，不是实验结果”，只解释三种可能形态，不能伪装成观测数据。

**Step 4: 绘制四类决策流程图**

流程顺序固定为：对齐是否可靠 → 双端是否有效 → 方向是否分离 → 内部插值是否变差 → 跨场景是否稳定。四个叶节点使用完整英文标签，并在下方给中文解释。

**Step 5: 构建、文本检查和视觉检查**

Run:

```powershell
python scripts/docs/build_vggt_vrfm_pdfs.py --document experiment --render
python scripts/docs/check_vggt_vrfm_pdf_text.py --document experiment
```

Expected: exit code 0，状态声明和四个决策标签可从 PDF 文本中提取。

用 PDF skill 检查封面、目录、三张图、长表格、结果卡模板和最后一页；特别检查横向表格、代码 commit/hash 字段和中英文长标签是否越界。

**Step 6: 提交前置实验排版稿**

```powershell
git add doc/camera_velocity_ambiguity_preexperiment
git commit -m "docs: typeset repair velocity pre-experiment report"
```

---

## Task 8: 编写通俗的 V-RFM 方法 Markdown

**Files:**

- Create: `doc/variational_rectified_camera_refiner/README.md`
- Create: `doc/variational_rectified_camera_refiner/variational_rectified_camera_refiner_method.md`

**Step 1: 用一维例子开场**

先讲“一个候选向左修、一个候选向右修、直接平均可能变成不动”，再映射到 500 帧 global trajectory 和两个重叠 100 帧 local predictions。明确这只是解释平均风险，真正进入方法必须通过前置实验 GO 门槛。

**Step 2: 通俗解释普通 RF 与 V-RFM**

普通 RF：相同 noisy trajectory state、time 和 condition 只能输出一个确定性速度，MSE 对冲突目标学习条件均值。

V-RFM：增加 `z` 后，同一个可见状态可在不同 `z` 下输出不同修复方向；轨迹在观测空间相交时仍可由 latent 身份区分。避免把 `z` 讲成离散窗口编号或手工分支标签。

**Step 3: 冻结 VGGT 方法接口**

输入列为：global camera centers、所有对齐后的 local camera centers、coverage mask、alignment confidence、可选 camera features。不得先把全部 local predictions 平均成单一输入。

第一版输出只生成 camera-center residual；rotation 和 FoV 沿用 global VGGT。说明这样做是降低验证变量，不是理论上永远不修 rotation/FoV。

**Step 4: 定义序列级 latent 与训练/推理差别**

- 一条完整长序列只采样一次 `z`，全 500 帧共享；
- 训练 posterior 可看目标 residual；
- 推理 prior 只能看可部署 condition；
- 解释 KL 权重、posterior collapse、prior-posterior gap 和 temporal mode switching 风险。

正文只保留一个经过 V-RFM 原论文核验的教学性目标公式；其余实现细节进入“未来实现附录”。

**Step 5: 明确 generator/selector 边界**

写明 V-RFM 生成多条 coherent repair samples，但不能自动判定哪条物理正确。部署仍需独立 scorer、几何能量、置信度或下游任务选择；若前置实验是 `SELECTOR_PROBLEM`，优先做 selector 而不是 V-RFM。

**Step 6: 写出 baseline、消融和进入条件**

至少包括 deterministic mean fusion、deterministic Transformer fusion、ordinary conditional RF、V-RFM、V-RFM without camera features；将 `GO`、`NO-GO/selector`、`NO-GO/deterministic` 三种路线写成表格。

**Step 7: 运行内容契约并提交**

Run: `python -m unittest tests.documentation.test_vggt_vrfm_documents -v`

Expected: 所有 Markdown 内容断言通过；只剩 method TeX/构建相关缺失项（如有）。

```powershell
git add doc/variational_rectified_camera_refiner
git commit -m "docs: explain variational rectified camera refinement"
```

---

## Task 9: 排版 6–10 页方法 PDF 与两张核心图

**Files:**

- Create: `doc/variational_rectified_camera_refiner/variational_rectified_camera_refiner_method.tex`

**Step 1: 将方法主稿压缩到讲解型结构**

正文目标 6–10 页：问题与例子、为什么不能先平均、RF、V-RFM、VGGT 输入输出、训练/推理、selector 边界、baseline 与进入条件。网络宽度、层数、优化器和数据加载放到简短“未来实现附录”，不扩写成代码合同。

**Step 2: 绘制普通 RF 与 V-RFM 对比图**

左侧用灰色平均箭头表示 deterministic conditional velocity；右侧用紫色不同深浅表示不同 `z` 下的 coherent velocity。图注强调“latent-conditioned velocities 可以不同”而非“V-RFM 自动发现正确窗口”。

**Step 3: 绘制完整 VGGT–V-RFM 流程图**

图中必须出现：500-frame global VGGT → nine local VGGT windows → prediction-only alignment/coverage → condition encoder → sequence-level `z` → camera-center residual samples → frozen/independent scorer。训练 posterior 与推理 prior 用虚实线区分。

**Step 4: 构建并验证页数与文本**

Run:

```powershell
python scripts/docs/build_vggt_vrfm_pdfs.py --document method --render
python scripts/docs/check_vggt_vrfm_pdf_text.py --document method
pdfinfo output/pdf/variational_rectified_camera_refiner_method.pdf
```

Expected: exit code 0；`Pages:` 在 6–10 范围内；文本可提取且状态声明可见。

**Step 5: 用 PDF skill 做视觉检查**

检查全部页面，重点核对前 3 页是否让不熟悉 Flow Matching 的 VGGT 研究者理解问题、两张流程图是否可独立阅读、唯一核心公式是否未溢出、紫色样本在打印/缩放后仍可分辨。

**Step 6: 提交方法排版稿**

```powershell
git add doc/variational_rectified_camera_refiner
git commit -m "docs: typeset variational camera refiner concept"
```

---

## Task 10: 增加三文档导航与一致性说明

**Files:**

- Create: `doc/VGGT_VRFM_DOCUMENT_SET.md`
- Modify: `doc/camera_solution_space_01_theory_foundation/README.md`
- Modify: `doc/camera_velocity_ambiguity_preexperiment/README.md`
- Modify: `doc/variational_rectified_camera_refiner/README.md`

**Step 1: 创建总导航页**

用一张表列出三份文档的“回答问题、当前状态、Markdown、TeX、PDF 输出、是否依赖实验结果”。再用三行箭头说明：概念边界 → V0 证据 → 条件方法设计。

**Step 2: 为每个目录写准确的单文档入口**

每个 README 包含：目标读者、非目标、共享术语、构建命令、输出路径、状态声明和上/下游文档链接。不要复制正文或写会快速过时的实验数字。

**Step 3: 运行链接和状态检查**

扩展 `test_vggt_vrfm_documents.py`，确认导航页引用的六个 source 路径都存在，三个 README 的构建 key 分别正确。

Run: `python -m unittest tests.documentation.test_vggt_vrfm_documents -v`

Expected: all pass。

**Step 4: 提交导航**

```powershell
git add doc/VGGT_VRFM_DOCUMENT_SET.md doc/*/README.md tests/documentation
git commit -m "docs: add VGGT V-RFM document navigation"
```

---

## Task 11: 全量构建、逐页渲染与交叉文档审计

**Files:**

- Modify as needed: all files under the three document directories
- Modify as needed: `doc/references/vggt_vrfm_report.sty`
- Modify as needed: `tests/documentation/test_vggt_vrfm_documents.py`

**Step 1: 跑完整自动检查**

Run:

```powershell
python -m unittest discover -s tests -v
git diff --check
python scripts/docs/build_vggt_vrfm_pdfs.py --document all --render
python scripts/docs/check_vggt_vrfm_pdf_text.py --document all
```

Expected: 全部 exit code 0；现有测试保持通过；三份 PDF 和全部页面 PNG 均生成。

**Step 2: 检查 LaTeX 日志**

搜索三个 build log 中的 `Undefined`、`Overfull`、`Underfull`、`multiply defined` 和字体替代。对真实版式问题逐项修正；不能仅通过隐藏 warning 让检查变绿。

**Step 3: 用 PDF skill 逐页检查**

先做每份 PDF 的 contact sheet，再检查原始分辨率页面。逐页核对：

- 中文字形、数学符号和英文标签；
- 目录页码、交叉引用、citation 和超链接；
- 表格/代码字段不越界；
- 图注脱离正文可理解；
- 无空白页、孤立标题、单行尾段和过小字号；
- 全局颜色和术语一致。

任何修改后重新运行 Step 1，而不是只重编单页。

**Step 4: 做人工语义审计**

逐文档搜索并人工阅读“多解”“多模态”“有效”“平均”“posterior”“selector”“已证明”等词的上下文，确认：

- long-vs-short 只产生 candidate，不证明固定观测物理多解；
- 双端有效且平均变差才是 V-RFM 的直接动机；
- V-RFM 不被描述为窗口选择器；
- 方法设计仍由前置实验 GO/NO-GO 控制；
- 数学附录与通俗正文无矛盾。

**Step 5: 提交视觉与一致性修复**

```powershell
git add doc scripts/docs tests/documentation
git commit -m "docs: verify VGGT V-RFM report set"
```

---

## Task 12: 冻结无结果版本并推送远端

**Files:**

- Verify only: entire worktree

**Step 1: 做提交前最终验证**

使用 `superpowers:verification-before-completion`，重新执行：

```powershell
git status --short
git diff --check HEAD
python -m unittest discover -s tests -v
python scripts/docs/build_vggt_vrfm_pdfs.py --document all --render
python scripts/docs/check_vggt_vrfm_pdf_text.py --document all
git status --ignored --short output tmp
```

Expected:

- tracked worktree clean；
- 所有测试和三份 PDF 构建通过；
- PDF、render PNG 和 TeX 缓存只显示为 ignored；
- 不存在尚未提交的实验事实或生成物。

**Step 2: 核对提交历史和范围**

```powershell
git log --oneline origin/codex/camera_solution_space_01_theory_foundation..HEAD
git diff --stat origin/codex/camera_solution_space_01_theory_foundation...HEAD
```

确认变化只包含 spec、plan、文档源、共享样式/文献、文档测试和构建工具。

**Step 3: 推送独立文档分支**

```powershell
git push -u origin codex/camera_solution_space_docs_reframe
```

Expected: 远端创建/更新同名分支；不改写前置实验分支历史。

---

## Gated Follow-up: 实验结果冻结后回填结果卡

此部分不阻塞“无结果版本”交付，仅在前置实验分支提交正式 result card 后执行。

1. 只读核验 result card commit、manifest、run ID、artifact hash 和统计输出；
2. 将结果逐字段填入前置实验 Markdown/TeX，不重新解释或挑选样本；
3. 按真实决策更新方法 PDF 的 GO/NO-GO 状态，理论 PDF 只引用结论等级；
4. 每个主结论加入一条真实轨迹和一个失败案例；
5. 重跑 Task 11 的全部自动与视觉验证；
6. 单独提交 `docs: report frozen repair velocity experiment results`，保留 source commit provenance。
