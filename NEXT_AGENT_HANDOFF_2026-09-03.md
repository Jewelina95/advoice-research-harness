# ADvoice 项目交接文档

更新日期：2026-09-03  
当前 GitHub 代码仓库：`https://github.com/Jewelina95/advoice-research-harness`（目前为 Private）  
当前演示提交：`6a85a11 feat: publish runnable ADvoice research demo`  
主要本地工程：`/Users/wenshaoyue/Desktop/research/ad general/AD voice/9.2/`  
可发布代码仓库：`/Users/wenshaoyue/Desktop/research/ad general/AD voice/9.2/github_release/advoice-research-harness/`

本文是下一位 Agent 的工作入口。它记录已经实现的方法、已经验证的结果、尚未成立的结论、用户反复确认的设计约束，以及下一轮应从哪里继续。不要只依据历史对话重建项目；先检查本文列出的代码、配置、产物和测试。

## 0. 当前可交付演示

本地演示地址：`http://127.0.0.1:8777/`。标准启动方式：

```bash
git clone https://github.com/Jewelina95/advoice-research-harness.git
cd advoice-research-harness
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
make demo
```

演示页面只呈现 ADvoice 当前系统，不展示开发阶段的 B1/B2/B3 对照，也不展示 SpeechCARE 对照。页面分成两个可独立理解的部分：

1. **单条录音演示。** 页面提供临床访谈、图片描述、结构化认知任务和自然讲话四种任务通道。点击“Run selected recording”后，本地服务调用 `POST /api/run-case/{case_id}`，重新执行任务路由、音频和文本测量、`MetricEvidence` 构建、`StateCard` 聚合及证据链接检查。结果以医生审查界面呈现，包含临床摘要、认知状态、原始指标证据、片段到指标再到状态的回溯图，以及可播放的来源片段。
2. **单一数据集演示。** 页面展示 ADReSS 2020 的一项归档受试者独立测试结果，共 27 人。当前页面报告 Accuracy 0.8519、Balanced Accuracy 0.8489、Macro F1 0.8500、Macro AUROC 0.9780、Macro AUPRC 0.9799，并展示混淆矩阵、HC/AD 类别表现和证据完整性审计。该结果来自 8.27 归档产物，属于开发阶段结果，不是新的 9.2 锁定复跑，也不是确认性临床验证。

公开仓库中的四条 WAV 是确定性生成的合成音频，不是患者录音。这样可以让审稿人直接运行代码而不违反数据许可和隐私要求。合成案例会真实执行公开证据提取链，但不会调用在线 GPT，也不会伪造疾病类别或概率。授权研究者可通过 `demo/local_cases.json` 在本地挂载受限患者样例；原始音频仍留在 Git 之外。

演示相关代码位置：

| 功能 | 文件 |
| --- | --- |
| 浏览器页面 | `demo/web/index.html`、`demo/web/app.js`、`demo/web/styles.css` |
| 本地运行接口与音频服务 | `demo/server.py` |
| 单条录音证据处理 | `src/advoice/demo.py` |
| ADReSS 队列展示数据 | `demo/output/adress_2020_cohort_summary.json` |
| 模型配置 | `configs/models/default.yaml` |
| Agent 配置 | `configs/agents/default.yaml` |
| 正式训练与融合 | `src/advoice/condition_c.py` |
| 证据和状态 | `src/advoice/evidence.py`、`src/advoice/states.py` |
| 医生报告 | `src/advoice/diagnostic_agent_report.py` |

当前演示已经通过 133 项本地测试、桌面和移动浏览器检查以及 GitHub Reproducibility checks。GitHub Pages 发布任务被跳过，因为仓库目前是 Private；可直接访问的版本仍是本地服务。

## 1. 项目要解决什么

ADvoice 研究的是：能否仅利用语音、转录、说话人角色和任务信息，对认知受损风险进行跨任务、跨语言、跨数据来源的研究性筛查，并输出医生能够审查和回溯的证据。

项目不是把所有声学和文本特征直接送进分类器，也不是让大语言模型自由生成诊断。当前方法把原始测量转换成具有医学含义、适用范围和质量边界的证据，再将证据组织成可修正的认知状态。监督模型形成统计先验，单一诊断 Agent 审查证据、反证、任务适用性和混杂，最后由验证集冻结的规则决定 Agent 的证据是否可以修正先验。

系统的临床边界是“研究性认知筛查和转诊支持”，不是阿尔茨海默病确诊工具，也不能仅凭语音确认病理类型或疾病分期。

## 2. 当前核心研究主张

当前最重要的研究概念不是“学习几个模态权重”，而是建立一个任务条件化、证据约束、可回退的认知状态模型：

1. 同一语音现象在图片描述、结构化访谈、认知任务和自然讲话中的含义不同，因此必须先做任务、语言、模态和角色路由。
2. 原始指标只有在记录了参考范围、异常方向、可靠度、混杂、任务范围和片段定位后，才可以成为可审查的 `MetricEvidence`。
3. 多个相关指标不能重复投票，应先在同一临床构念内形成 `StateCard`；总体状态和任务特异状态可以并存，但重复视图必须去重。
4. 片段轨迹用于保留局部变化。医生报告不展示复杂模型矩阵，而是展示关键片段、对应文本、状态变化和可播放音频。
5. 诊断 Agent 维护并修正认知状态和证据链，不进行无约束自由推理。这是本项目对 Cognition-of-Thought 思想的实际借鉴。
6. 质量控制只降低可靠度或触发回退，不能直接增加疾病风险。
7. 报告生成发生在预测锁定以后；报告 Agent 不能重新计算或改变类别概率。

## 3. 与相关工作的关系

### SpeechCARE

参考论文：`https://www.nature.com/articles/s41746-025-02026-x`

SpeechCARE 使用音频表示、文本表示和人口学信息，通过自适应门控学习病例级模态权重。它的优势是端到端表示学习和动态融合。ADvoice 借鉴了共享多语言表示、任务信息进入融合以及动态门控候选，但没有复制其系统。

ADvoice 的差异是把融合单位从“模态”下沉到“任务条件化证据和认知状态”，并增加类型化证据 ID、反证、混杂、报告权限、片段回溯、确定性回退以及医生报告契约。当前实验只能证明认知表示有独立增益，不能证明完整 ADvoice 9.2 已超过 SpeechCARE。

### 多证据临床决策系统

相关工作：`https://www.nature.com/articles/s41591-026-04601-5` 和 `https://www.nature.com/articles/s41746-026-03048-9`

本项目吸收的原则是：不同来源证据要有明确身份、适用范围和冲突处理；模型输出要和支持证据、反证及不确定性绑定；无效工具调用或证据不足必须回退。不要把这些论文写成已经验证本项目的方法，它们只是设计依据。

### Cognition-of-Thought

相关论文：`https://arxiv.org/abs/2509.23441`

本项目使用的是“认知状态维护和修正”的结构思想，不是复现社会推理任务，也不是暴露模型隐藏思维链。可公开的对象只有结构化证据、状态更新、引用 ID、回退原因和锁定结论。

## 4. 版本演进

| 阶段 | 主要变化 | 保留下来的设计 |
| --- | --- | --- |
| 7.16 | 整理指标体系、临床状态、数据通道和医生报告 | 指标证据身份、状态层、Layer A/Layer B 评估结构 |
| 7.30 | 引入训练式融合，与 SpeechCARE 对照 | 音频、文本、状态和分支融合；同协议比较意识 |
| system 8.13 | 将数据、训练、评估和报告组织成可重复工程 | 配置驱动、阶段缓存、固定报告入口 |
| 8.21 | 单一诊断 Agent 加两个监督模块 | Agent 不只写报告，而是参与证据审查和受限预测修正 |
| 8.27 | 引入认知状态维护、技能包、回退和分层审计 | Cognition-of-Thought 风格的状态更新；Evidence/State/Trace 契约 |
| 9.2 | 去除概率锚定、修复证据 ID、重写门控和病例路由 | 当前正式方法协议和 PREPARE 审计 |
| 9.3 公开演示 | 四通道合成案例、单页浏览器演示、GitHub CI | 不公开受限数据；演示与正式训练明确分离 |

历史目录只能用于理解设计和展示风格。不得把 8.27 的产物重命名为 9.2 结果。

## 5. 现行 9.2 完整链路

### 5.1 数据路由

每个病例先确定：数据集、任务类型、语言、可用模态、说话人角色、患者有效语音范围和质量状态。路由不能读取疾病标签。

访谈数据必须区分患者和采访者；图片描述保留任务内容边界；多任务数据保留每个任务的独立身份；自然讲话只使用该任务确实可观察的状态。

### 5.2 输入标准化

音频处理包括统一采样率、低通、质量测量、任务感知切分和片段索引。当前深音频配置使用 16 kHz、5 秒窗口、25% 重叠、每位受试者最多 30 秒，并保留片段注意力候选。

文本处理包括受试者语音隔离、ASR 或原始转录规范化、语言识别、任务文本分区和说话人角色处理。ASR 默认使用 Whisper Large V3 Turbo 的 MLX 实现。

### 5.3 指标与 `MetricEvidence`

原始指标来自声学、语音行为、语言、对话和任务表现。不是所有指标都可以写入医生报告。

每个 `MetricEvidence` 至少包含：

- `value`：当前病例的原始测量值。
- `reference`：只用训练折建立的同任务或共享参考。
- `direction`：相对参考偏高或偏低，以及该方向是否与认知受损假设一致。
- `reliability`：当前病例中该证据受音频、转录、角色和样本量影响后的可信程度。
- `components`：该证据由哪些原始指标组成。
- `confounds`：设备、噪声、时长、任务提示、ASR、角色覆盖等可能污染因素。
- `task_scope`：证据适用于哪个任务或任务族。
- `segment_ids`：可以回放和查看文本的原始片段。
- `inference_permission`：是否允许进入模型推理。
- `report_permission`：是否允许进入医生报告。

“方向”不是疾病概率。例如静音比例升高只能表示相对训练参考存在更多非发声时间；是否支持认知受损还要结合任务、可靠度和其他证据。“可靠度”也不是医学真实性概率，它是该病例中测量是否足以被使用的门控量。

核心实现：`src/advoice/evidence.py`。

### 5.4 `StateCard`

`StateCard` 将同一认知构念内的相关指标聚合，避免同义指标重复投票。状态保留支持证据、反证、不可观察原因、可靠度、任务范围和片段轨迹。

当前状态包括停顿与流畅性、输出效率、发声连续性、韵律、词汇提取、词汇多样性、语义连贯性、信息密度、任务表现和互动负担等。只有当前通道可观察且算法已经实现的状态才可以启用。

总体状态和任务特异状态同时存在。若可靠的任务特异参考可用，应替代相同测量的总体参考，而不是让二者重复投票。

核心实现：`src/advoice/states.py`。

### 5.5 监督先验

当前条件 C 使用多分支监督学习形成折外先验：

- 文本分支：`Alibaba-NLP/gte-multilingual-base` 表示及文本统计。
- 音频分支：`utter-project/mHuBERT-147` 表示、片段表示和声学统计。
- 状态分支：共享与任务特异 `StateCard`。
- 序列分支：片段级轨迹和任务内变化。
- 质量分支：只作为门控和残差化变量，不作为疾病证据。

当前预训练编码器主要用于冻结表示提取，疾病标签训练发生在浅层分类器、分支选择、片段注意力、堆叠和门控部分。不要把它写成已完成的端到端全参数微调。

训练采用受试者级五折交叉验证和折外预测。训练折内建立参考、标准化、缺失值处理、质量残差化、分支选择、校准和融合参数。测试标签不得进入任何选择过程。

核心实现：`src/advoice/models.py`、`src/advoice/condition_c.py`、`src/advoice/dynamic_gate.py`、`src/advoice/sequence_expert.py`、`src/advoice/deep_embeddings.py`、`src/advoice/deep_audio_embeddings.py`。

### 5.6 单一诊断 Agent

正式条件 C 的 Agent 配置位于 `configs/agents/default.yaml`，当前 provider 为 OpenAI API，模型配置为 `gpt-5.6-luna`。

Agent 首轮接收的是盲化结构化证据包，不是完整原始病历，也看不到监督概率、预测类别或真实标签。证据包包含：任务和语言背景、MetricEvidence、StateCard、关键片段、反证、质量限制、可用证据原型和允许调用的工具。

Agent 输出：

- HC 与认知受损的 0 至 4 级离散证据等级。
- 在证据足够时输出 MCI 与 AD 的 0 至 4 级分期证据；不足时必须返回 `undetermined`。
- 支持和反对结论的类型化证据 ID。
- 对状态的“保留、降权、失效、不可观察”更新。
- 质量限制、冲突和回退理由。

Agent 不直接输出最终概率。系统在开发集内把离散证据等级校准为证据似然，再由冻结修正强度进行有限修正。

技能包位于 `skills/ad_evidence_diagnostic/`，包括医学范围、证据层级、状态知识、任务可观察性、混杂与鉴别、认知回退、报告权限、标签泄漏政策和报告契约。技能包是 Agent 的约束知识，不是已完成的模型微调。

核心实现：`src/advoice/cognitive_agent.py`、`src/advoice/diagnostic_agent.py`、`src/advoice/agent_runtime.py`。

### 5.7 验证冻结的融合与回退

Agent 的证据能否改变先验由以下条件共同决定：证据覆盖、可靠度、混杂程度、任务和语言路由、证据 ID 有效性、分期可判定性以及开发集上的净增益。

修正强度候选为 `0.0, 0.125, 0.25, 0.5, 0.75, 1.0, 1.5`。非零强度必须在开发病例上使 Macro F1 至少提高 0.01，同时 Macro AUROC 的下降不超过 0.001。否则强度自动归零。

非法 ID、未审查反证、不可观察状态、严重质量限制、校准样本不足或开发门禁失败时，系统回退到监督先验。

三分类融合分成“HC/认知受损”和“受损条件下 MCI/AD”两个层级。中性 Agent 证据必须严格保持先验不变，不能因为类别数量产生偏向受损的数学偏置。

核心实现：`src/advoice/cognitive_agent.py`、`src/advoice/condition_c.py`。

### 5.8 锁定结论和报告

最终概率和类别先锁定，再生成医生报告和患者可读版本。报告只能引用 `report_permission=true` 的证据；质量控制、MFCC、设备噪声和音量不能被写成疾病机制。

医生报告应包括：筛查结论和边界、主要认知发现、每项发现对应的状态和原始证据、关键可播放片段、反证、采集质量和解释限制、建议的临床复核和随访。报告不是训练日志，也不向医生展示内部提示词或复杂轨迹矩阵。

核心实现：`src/advoice/diagnostic_agent_report.py`、`src/advoice/report_agent.py`、`skills/ad_evidence_diagnostic/REPORT_CONTRACT.md`。

## 6. 数据通道和数据集

| 数据通道 | 当前数据集 | 主要可观察内容 |
| --- | --- | --- |
| 临床访谈 | IAEAV | 患者/采访者角色、回答启动、互动负担、患者说话占比、语言与停顿 |
| 图片描述 | ADReSS 2020、ADReSSo diagnosis | 停顿、输出效率、词汇提取、句法、内容单元、图片任务相关性 |
| 结构化/多任务认知任务 | PROCESS-2、PREPARE、DementiaBank Pitt | 任务特异停顿、词语流畅性、回忆遗漏/侵入、任务表现和跨任务状态 |
| 自发或自然讲话 | TAUKADIAL、DementiaNet Public Figures、NCMMSC2021_AD | 多语言自然语音、发声连续性、韵律、词汇和叙事状态；只启用任务允许的证据 |
| 纵向进展 | ADReSSo progression | 随访差异、输出效率、停顿和发声连续性变化 |

配置中的十项独立任务：`IAEAV`、`ADReSS_2020`、`ADReSSo_2021_diagnosis`、`ADReSSo_2021_progression`、`PROCESS_2`、`PREPARE_DrivenData`、`TAUKADIAL`、`DementiaBank_Pitt`、`DementiaNet_PublicFigures`、`NCMMSC2021_AD`。

不同数据集必须独立训练和评估，不能把来源不同的病例随意混合后报告一个总准确率。跨数据集实验必须明确定义训练域、外部测试域、标签映射和任务可观察性。

`NCMMSC2021_AD` 只使用长录音。六秒短片和无标签测试轨道由配置排除。

原始受限音频、转录、临床标签和身份信息不能提交到 GitHub。公开仓库只包含确定性生成的合成音频。

## 7. 三个实验条件

| 条件 | 定义 | 用途 |
| --- | --- | --- |
| B1 | 传统机器学习基线 | 检查手工声学、语言和结构化特征的传统预测能力 |
| B2 | 直接转录文本 Agent | 检查不提供结构化证据时，通用模型直接读文本的表现和报告风险 |
| B3 / Ours | 监督先验 + 认知证据图 + 盲化诊断 Agent + 冻结融合 | 当前主方法 |

补充历史条件 `Legacy_C` 只用于解释 8.13 版本：可靠性感知融合加报告型 Agent。它不是当前主结果。

必须保留相同编码器的析因实验：

- E0：相同编码器加普通模态门控。
- E1：E0 加 MetricEvidence、StateCard 和任务路由，不启用 Agent 修正。
- E2：E1 加盲化 Agent、病例路由和验证集冻结修正。
- E3：向 Agent 显示监督先验，只用于审计锚定效应，不能作为主结果。

E1-E0 证明认知证据框架的贡献；E2-E1 才能证明 Agent 的预测净增益。不要把二者合并归因。

## 8. 两层评估协议

### Layer A：预测和校准

必须报告 accuracy、balanced accuracy、Macro/Weighted F1、MCC、Macro/Weighted AUROC、Macro AUPRC、Brier、log loss、ECE、错误率、阈值附近不确定率、高置信错误率、Macro FNR/FPR，以及每类敏感度和特异度。适用时补充微平均 AUROC、AUPRC 和 F1 以对齐 SpeechCARE。

### Layer B：框架特有验证

必须报告 MetricEvidence 完整性、StateCard 完整性、分支贡献回溯、报告权限执行、片段证据忠实度、临床报告 25 分量表、概念干预、仅概念与原始状态消融、Agent 修正正确数、破坏原正确数、净修正数、回退率、证据 ID 有效率和负对照。

负对照包括 QC-only 和去除时长/音量后的比较。QC-only 表现过强通常意味着设备、采集流程或数据集来源泄漏，不是医学成功。

正式比较需要受试者独立切分、训练折内预处理、外折预测、多随机种子和置信区间。发布论文前至少运行十个随机种子，并使用新的站点外、时间外或未触碰外部队列做确认性验证。

## 9. 当前已验证结果

### PREPARE 官方测试回顾性结果

病例数 `n=412`，其中 HC 229、MCI 51、AD 132。

| 方法 | Accuracy | Macro F1 | Micro AUROC | Macro AUROC | Micro AUPRC |
| --- | ---: | ---: | ---: | ---: | ---: |
| B1 | 0.5534 | 0.5149 | 0.7616 | 0.7278 | 0.5767 |
| B2 | 0.4272 | 0.3770 | 0.6280 | 0.5836 | 0.4948 |
| B3 / ADvoice 9.2 | 0.6723 | 0.6183 | 0.8462 | 0.8131 | 0.7011 |
| SpeechCARE 论文十次均值 | 0.7211 | 未报告同口径值 | 0.8683 | 未报告同口径值 | 0.7473 |

当前可以成立的结论：

- B3 明显高于内部 B1 和 B2。
- 相同冻结编码器的析因实验中，认知证据框架使 Accuracy 提高 0.0752，Macro AUROC 提高 0.0601。
- 相对 SpeechCARE 公开检查点，ADvoice 的 MCI recall 为 0.686，对方为 0.333；这是早期受损敏感性优势，不是总体性能优势。
- 类型化证据 ID 有效率约为 95.39%。
- 报告权限执行和质量证据分离审计为 100%。
- 结构化报告量表为 18.5/25，直接 LLM 为 11.25/25，当前配对报告病例只有 4 例，因此只能作为初步质量结果。

当前不能成立的结论：

- 独立 ADvoice 9.2 没有超过 SpeechCARE 的 Accuracy、Micro AUROC 和 Micro AUPRC。
- Agent-on 相对 Agent-off 的 Macro AUROC 约为 -0.0008，Accuracy 变化为 0；Agent 分类净增益尚未建立。
- PREPARE 官方测试集在历史开发中被反复查看，当前比较是回顾性开发测试，不能标为未触碰的确认性外部验证。
- 其余九项配置任务尚未按照 9.2 协议重新运行。它们只有 8.27 历史产物。

## 10. 当前失败模式及已完成修复

### 失败模式

1. Agent 可见的任务语义证据不足。停顿和输出效率已经进入证据包，但内容单元、语义流畅性聚类与切换、记忆遗漏/侵入/重复、任务完成度和叙事连贯性仍不完整。
2. Agent 的原始 HC/受损证据判断区分能力偏弱，三分类证据似然接近随机，不能通过增大修正强度解决。
3. 早期实现虽然记录 `state_updates`，但最终概率主要读取 `evidence_scores`，状态“保留、降权、失效、不可观察”还没有完整变成可执行的重新计算链。
4. 旧的层级融合把三分类中性意见错误地解释为更支持“受损”；该数学偏置已经修复。
5. 现有 Agent 修正主要处理 HC 与受损，MCI 与 AD 内部边界仍依赖监督先验，限制了对分期错误的修正。
6. SpeechCARE 端到端微调 mHuBERT、mGTE、自注意力和门控；当前 ADvoice 主要使用冻结表示和浅层融合，上游表示能力仍有差距。
7. 西班牙语和任务不平衡仍明显；小语言或小任务上的高分不能支撑总体优效。

### 已完成修复

- Agent 首轮不再看到监督概率和类别。
- 状态、指标、片段和 QC 使用统一类型化 ID。
- 覆盖率改为状态、任务和分支覆盖，不受增加重复指标影响。
- 增加任务族、语言、模态、质量和预测-证据冲突路由。
- Agent 改为输出离散证据等级，不直接写概率。
- 修正强度只在开发折外病例上学习，不通过门禁则归零。
- 中性 Agent 证据保持监督先验不变。
- 分期证据不足时返回中性，不默认 MCI。
- 取消冲突病例的修正放大。
- 推理权限和医生报告权限已经分离。
- 开发门禁失败时，正式测试不调用没有预测作用的 Agent。

探索性重放中，中性证据偏置修复使 Accuracy 从 0.6723 增到 0.6796，Macro F1 从 0.6189 增到 0.6246；由于原修正强度是在旧公式下选择，这不是正式冻结结果。

## 11. 下一轮方法优先级

下一位 Agent 不应继续通过提示词微调或测试集调参追求超过 SpeechCARE。正确顺序如下：

1. 补齐任务语义证据工具，并为每项工具增加训练折参考、任务范围、片段 ID、反证和报告权限。
2. 让 `state_updates` 真正触发状态重新聚合和两个监督头的重新计算，而不是只写入日志。
3. 将分类稳定拆成 HC/受损头和受损条件下 MCI/AD 头，分别训练、校准和审计。
4. 在相同编码器和相同切分下完成 E0/E1/E2，对 Agent 增益做干净隔离。
5. 在开发集运行多随机种子；只有 E2-E1 的置信区间稳定为正，才在正式测试启用非零 Agent 修正。
6. 再评估端到端或参数高效的多语言编码器适配。必须与认知框架增益分开报告，避免把更强编码器的贡献归给 Agent。
7. PREPARE 方法冻结后，使用新的外部队列做确认性验证；之后再按数据集逐一运行其余九项任务。
8. 多语言处理要建立语言特异分词、词典和任务参照，但共享状态定义和证据 schema。不能只修中文而忽略西班牙语。
9. 新数据、指标、状态或 Agent 契约变化都必须使缓存失效并自动重跑相应 Layer A/Layer B 报告。

## 12. 公开演示当前状态

本地演示地址：`http://127.0.0.1:8777/`

公开演示包含四个确定性合成案例：

1. Clinical interview
2. Picture description
3. Structured cognitive task
4. Natural speech

单页界面先解释完整架构，再让用户切换四类案例。每个案例展示音频、转录、MetricEvidence、StateCard、片段轨迹、trace map 和 Agent 风格报告结果。点击片段会定位音频并显示对应文本。

公开演示不会加载训练后的临床权重，也不会调用在线 GPT。当前页面中的报告是基于证据契约生成的确定性 `offline_preview`，用于展示接口和追溯流程，不能被描述为真实病例诊断或实时 Agent 结果。

公开仓库不包含原始患者音频。四个 WAV 是 `demo/generate_sample.py` 生成的音调和静音，不是人声，也不是正常人参考。

仓库目前是 private，因此 GitHub Pages 工作流按配置跳过。CI 已通过，最近一次通过的工作流运行是 `https://github.com/Jewelina95/advoice-research-harness/actions/runs/33754669262`。不要未经用户确认修改仓库可见性。

## 13. 代码和配置入口

| 内容 | 路径 |
| --- | --- |
| 项目入口和运行编排 | `src/advoice/cli.py`、`src/advoice/pipeline.py` |
| 数据加载与适配 | `src/advoice/data.py`、`configs/datasets/*.yaml` |
| 特征提取 | `src/advoice/features.py`、`src/advoice/transcripts.py`、`src/advoice/asr.py` |
| MetricEvidence | `src/advoice/evidence.py` |
| StateCard 和片段证据 | `src/advoice/states.py` |
| B1/B3 监督模型 | `src/advoice/models.py`、`src/advoice/condition_c.py` |
| 深文本和音频表示 | `src/advoice/deep_embeddings.py`、`src/advoice/deep_audio_embeddings.py` |
| 动态门控和序列专家 | `src/advoice/dynamic_gate.py`、`src/advoice/sequence_expert.py` |
| 诊断 Agent | `src/advoice/cognitive_agent.py`、`src/advoice/diagnostic_agent.py` |
| Agent 运行时 | `src/advoice/agent_runtime.py` |
| B2 直接 Agent | `src/advoice/direct_agent.py` |
| 医生报告 | `src/advoice/diagnostic_agent_report.py`、`src/advoice/report_agent.py` |
| 评估 | `src/advoice/evaluation.py`、`src/advoice/failure_analysis.py` |
| 报告生成 | `src/advoice/reporting.py`、`src/advoice/aggregate_reporting.py` |
| 模型配置 | `configs/models/default.yaml` |
| Agent 配置 | `configs/agents/default.yaml` |
| 评估配置 | `configs/evaluation/default.yaml` |
| AD 医学技能包 | `skills/ad_evidence_diagnostic/` |
| JSON 契约 | `schemas/` |
| 公开演示 API | `demo/server.py` |
| 公开演示逻辑 | `src/advoice/demo.py` |
| 公开演示前端 | `demo/web/index.html`、`demo/web/styles.css`、`demo/web/app.js` |
| 方法协议 | `docs/METHOD_PROTOCOL_9_2.md` |
| 验证边界 | `docs/VALIDATION_STATUS.md` |
| SpeechCARE 差距审计 | `docs/PREPARE_SPEECHCARE_GAP_ROOT_CAUSE_2026-09-03.md` |

## 14. 运行命令

在可发布仓库根目录运行：

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
make test
make demo
```

演示默认打开 `http://127.0.0.1:8765/`。当前会话使用的本地服务运行在 `8777`。

运行单一数据集：

```bash
make validate DATASET=PREPARE_DrivenData
make full DATASET=PREPARE_DrivenData
make evaluate DATASET=PREPARE_DrivenData
make report DATASET=PREPARE_DrivenData
```

完整 Agent 运行需要有效的 OpenAI API 配置，并会产生调用成本。快速工程检查使用：

```bash
make quick DATASET=PREPARE_DrivenData
```

`make all-full` 先执行 PREPARE release gate。当前 gate 未通过，因此它不会继续把其他数据集跑成正式 9.2 结果。这是有意的防护，不是程序卡死。

## 15. 工程纪律

- 数据集必须由配置注册，不在代码中散落硬编码路径。
- 每个运行必须记录配置哈希、数据清单哈希、代码版本、随机种子、模型版本和阶段缓存。
- 数据变更、metric/state schema 变更、模型配置变更、Agent 契约变更和评估配置变更要分别触发正确的缓存失效。
- 所有参考范围、标准化、特征选择、类别阈值和校准参数只在训练折拟合。
- 受试者不能跨训练、开发和测试集合。
- 不因目标是超过基线而筛选有利种子、修改测试阈值或隐藏失败数据。
- 公共代码、README、网页和图应使用英文；内部交接和用户汇报可以使用中文。医学技能中的中文提示词是当前实验配置的一部分，不能为了表面统一语言而静默修改。
- 图表模板要固定；新增数据只更新数据层，不反复重写视觉结构。
- 医生报告必须使用自然、临床可读的表达，不展示内部训练术语、提示词或开发目标。
- 每次更改后运行测试，并同步更新验证状态；不能只更新 HTML 而不更新产生它的数据。

## 16. 用户已经反复确认的要求

1. 保留 metric -> MetricEvidence -> StateCard -> segment trajectory -> 分支/状态融合 -> Agent -> 锁定报告的完整可追溯链。
2. `trajectory` 保留并做成可交互片段定位；CN 正常参考是独立虚线，不是把正常组编码成数值 0。
3. 不同状态使用不同支持指标，不能所有状态显示相同的一段证据。
4. 每个证据必须说明原始值、参考、方向、可靠度、混杂、任务和片段来源。
5. 医生报告要有筛查结论、主要发现、原始证据、反证、采集限制、临床复核和未来建议。
6. Agent 应参与证据审查和有限预测修正，不应退化成单纯把冻结概率改写成报告。
7. 只使用一个诊断 Agent，外围由两个小型监督模块和确定性验证器支持；不要堆叠多个角色 Agent。
8. 不同数据集独立训练和评估；多任务数据不能在状态层提前平均掉任务差异。
9. 最终报告保留 Layer A 和 Layer B 两层，并同时给总图、细节图和数据值。
10. 可以以超过 SpeechCARE 为研究目标，但不能伪造优效。未超过时必须定位失败模式并设计新的未触碰验证。
11. GitHub 只放核心代码、配置、合成案例和可复现说明，不放受限原始音频或标签。
12. 公开网页必须在同一页介绍方法、四个数据通道案例、Agent 输出和一个数据集评估结果。

## 17. 下一位 Agent 开始工作的检查顺序

1. 阅读本文件、`README.md`、`docs/METHOD_PROTOCOL_9_2.md`、`docs/VALIDATION_STATUS.md` 和差距审计文档。
2. 检查 `git status`，不要覆盖用户未提交的改动。
3. 运行 `make test`，确认当前测试基线。
4. 检查 `configs/datasets/registry.yaml`、`configs/models/default.yaml` 和 `configs/agents/default.yaml`，确认本次改动属于数据、证据、训练、Agent 还是报告层。
5. 先写清楚假设、数据边界、缓存失效范围和对应 Layer A/Layer B 指标，再改代码。
6. 若修改预测链，必须完成 E0/E1/E2 隔离；若只修改网页，不得重写或重新解释冻结实验结果。
7. 若跑 PREPARE，明确标为回顾性开发评估；若要发表优效结论，换用新的锁定外部队列。
8. 完成后更新 `docs/VALIDATION_STATUS.md`、报告、测试和本交接文档中的状态。

## 18. 当前最简状态结论

当前仓库已经具备完整的配置驱动研究管线、类型化证据和状态契约、受约束单一诊断 Agent、确定性回退、两层评估、四通道公开演示和 GitHub CI。认知证据框架相对相同编码器基线的增益已经成立；诊断 Agent 的病例分类净增益和相对 SpeechCARE 的总体优效尚未成立。下一轮真正需要增加的是任务语义证据和可执行状态更新，并在干净的多随机种子和新外部队列上验证，而不是继续调整展示文本或用测试集追逐更高数字。
