# P1/P2/P3 可执行训练协议

本文件把“按数据集训练”和“跨任务共享”拆成三个不同实验，避免同一份结果同时声称内部性能、通用模型和外部泛化。

## 1. 训练前冻结的四张表

正式训练前必须由数据清单生成并人工核对以下表，任何表发生变化都产生新的数据快照哈希：

1. `dataset_manifest`：subject、session、task、language、source、endpoint、label provenance、license 和可用模态。
2. `endpoint_harmonization`：原始标签到研究终点的映射及临床含义；映射来自数据文档，不从文件名猜测。
3. `split_manifest`：外层 train/test、内层 fold、subject/source group 和近重复组。
4. `pooling_eligibility`：每个数据集是否可进入 P2，以及排除原因。

共享训练的纳入条件同时包括：横断面终点、标签语义能够统一、subject-level 标签可信、许可允许合并、角色/任务元数据足以路由、shortcut 负控未失败。纵向进展、连续分数、公开人物压力测试和标签含义不兼容的数据不得为了扩大样本量进入同一 P2 head。

## 2. P1：数据集内三条件基准

每个数据集按其官方终点和官方 split 独立运行 B1/B2/B3。没有官方 split 时，先按 subject/source group 固定外层 split，再开始任何参考计算或特征选择。

- B1 使用预注册传统指标和正则化模型。
- B2 接收与 B3 相同病例的原始可用模态和任务说明，但不接收状态、监督先验、Skill 或证据工具。
- B3 使用完整 8.27 链路。

P1 回答“在同一数据协议下 B3 是否优于传统模型和直接大模型”，不回答跨数据集泛化。

## 3. P2：共享认知状态主模型

P2 的共享单位不是数据集或原始模态，而是任务条件化的认知状态观测。每个 task instance 形成：

`state severity + observability mask + reliability + trajectory summary + task family + language + capped auxiliary embeddings`

### 3.1 模型结构

1. **共享状态骨架**：S01–S14 的定义固定，缺失或不适用状态由 observability mask 置零，不用缺失值伪装正常。
2. **任务条件化融合**：学习任务族的状态系数，但使用层级收缩约束其靠近全局系数，防止小任务产生任意权重。
3. **语言适配**：指标参考、分词/解析器和概率校准按语言拟合；数据不足时只学习截距/温度，不学习整套语言特异疾病机制。
4. **辅助表示**：冻结音频/文本编码器；训练折内降维，贡献受上限约束，不能进入医生机制解释。
5. **终点头**：P2 只使用一个预注册且语义统一的主分类终点。其他终点进入 P3 独立 head。

首选实现是低容量 masked hierarchical gate，而不是端到端训练大型 Transformer。候选复杂模型只有在内层验证同时改善判别、校准和跨 fold 稳定性时才保留；否则使用正则化线性/广义加性版本。这一选择由有效 subject 数和预注册模型选择规则决定，不由单次测试结果决定。

### 3.2 损失与选择

主损失是终点适配的交叉熵；同时记录 Brier/ECE，但不在没有预注册的情况下临时调权重追逐测试集。每个数据集在 minibatch 或样本权重中等权进入共享目标，避免大数据源完全支配。状态门控使用稀疏和层级收缩；辅助表示总贡献有上限。损失权重和上限只在内层 CV 网格中选择，并保存所有候选结果。

### 3.3 嵌套 OOF 顺序

对每个 outer fold：

1. 冻结 outer test subjects。
2. 在 outer train 内拟合预处理、参考、任务/语言适配器和模块 A。
3. 通过 inner folds 为每个 outer-train subject 产生未见过该 subject 的 OOF prior。
4. Agent 读取 OOF prior 和 OOF evidence package，生成训练轨迹。
5. 只用这些 OOF 轨迹训练模块 B。
6. 在全部 outer train 上重拟合模块 A 和 B。
7. outer test 每个病例只运行一次冻结链路。

## 4. P3：泛化和特殊终点

- 留一数据集、留一任务和留一语言验证只在标签语义相同的终点家族内进行。
- 纵向进展按 subject 成组并保留成对访视，输入包括 visit-specific StateCards 和配对变化。
- 序数终点使用序数 head；多分类输出完整类别 logit 修正向量；连续终点使用数值及区间修正。模块 B 不再用一个标量假装覆盖所有终点。
- 公开视频等非标准数据只作为压力测试，不回流调整主模型。

## 5. 两个监督模块与 Agent

模块 A 学习统计先验及任务条件化状态贡献。单一诊断 Agent 使用版本化 Skill 和只读工具检查证据、反证、任务冲突与混杂，提出有限候选修正。确定性验证器先审查候选；模块 B 再学习修正是否可采纳及校准。Agent 不是自由诊断器，也不是报告润色器；它只能在预注册边界内修正可审计的认知状态假设。

## 6. 发布门槛

正式训练前必须满足：primary references 核验完成、P2 pooling 表冻结、split/leakage 审计通过、所有 conditional metric 有验证凭证、P1/P2/P3 终点和统计方法预注册。任何一项缺失，只允许生成开发运行，不允许发布性能结论。

