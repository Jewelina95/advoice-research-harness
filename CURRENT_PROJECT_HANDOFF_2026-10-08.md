# ADvoice 当前项目交接文档

更新时间：2026-10-08（Europe/London）

本文面向接手代码、实验和论文工作的下一个 Agent。它描述当前可验证状态，不把历史结果、计划、软件测试和真实临床性能混在一起。

> 名称说明：本仓库名称是 **ADvoice research harness**。仓库目前没有 AWS 部署、Terraform、CloudFormation 或云端训练基础设施。如果“整个 AWS 项目”是指云服务，则该部分尚不存在；本文按 ADvoice 项目整理。

## 1. 先读这一页：当前真实状态

| 项目 | 当前状态 |
| --- | --- |
| GitHub 仓库 | `https://github.com/Jewelina95/advoice-research-harness.git` |
| GitHub `main` | `88a887c` (`handle channel-unobservable states in evaluation`) |
| 最新本地候选分支 | `feat/evidence-state-pilot-v1` |
| 候选实现检查点 | `256199a6628c60ce2c03ce6fdaea993fe0b9410d`（本交接文件提交前） |
| 与 `origin/main` 的差异 | 候选分支 **ahead 30 / behind 8**；尚未合并或推送为 GitHub 主线 |
| 全量软件测试 | `1298 passed, 3 skipped` |
| Pilot DAG 干运行 | 13 个阶段全部生成 `dry_run` 工件；没有真实 API 调用 |
| 真实付费 Agent pilot | **没有运行** |
| 新版 Agent 净增益 | **尚未由真实数据证明** |
| SpeechCARE 超越状态 | **没有超越**；PREPARE 历史独立结果仍低于论文均值 |
| 论文状态 | 本地 Overleaf 上传包和 PDF 已生成；没有成功上传到 Overleaf |

关键判断：代码已经具备更严格的 evidence/state/Agent pilot 合同、OOF 来源约束、运行时预算和报告门控，但它仍是一个待复审的候选分支，不是已经发布并证实优于 SpeechCARE 的最终系统。

## 2. 项目目标

输入是一段患者语音及其可用转录、任务信息和说话角色。系统从三个层面分析：

1. 声学与韵律：停顿、发声连续性、语速、音高和辅助声学表示。
2. 语言与认知任务：词汇提取、信息密度、句法、内容覆盖、任务完成情况。
3. 对话与交互：患者/访谈者角色、轮次、响应延迟、修正与中断。

系统输出研究性认知筛查结果、HC/MCI/AD 或数据集定义的目标类别、证据充分性、主要 StateCard、支持与反证、局部轨迹、限制和复核建议。它不能依据语音确诊 Alzheimer 病理，不能替代量表、神经系统检查、影像或生物标志物。

核心研究问题不是“LLM 能否写一份报告”，而是：在相同输入、编码器、患者和分割下，结构化 MetricEvidence、StateCard、可回放状态修订和受约束 Agent 是否能相对匹配监督基线产生可重复的净增益。

## 3. 当前存在两条不同的 Agent 路径

### 3.1 维护中的校准单 Agent 基准路径

这是论文和受控比较应优先使用的路径：

```text
数据路由
  -> MetricEvidence
  -> shared/task-specific StateCards
  -> subject-isolated supervised base
  -> prior-blind Agent assessment
  -> executor-authorized evidence/state replay
  -> OOF joint calibration
  -> locked prediction
  -> clinician-facing report
```

Agent 首轮不读取监督概率或真值。Agent 返回有序类别支持、证据 ID、状态判断和有限操作。监督基础、Agent 判断和状态回放通过开发集 OOF 联合校准，而不是手工设置权重。

### 3.2 完全 Agent-led 实验路径

入口是 `advoice agent-led`。Agent 可以调用证据工具、形成假设、修订 evidence workspace 并作出独立研究判断。该路径与校准基准分开；其结果不能混入 B/J-A/J-S/J-AS 的受控比较。

完全 Agent-led 路径当前仍存在边界：修订后 A/B 顾问模型不会自动重新执行，旧顾问输出会失效；没有独立校准时，Agent ordinal scores 不是概率；provider 错误或 abstention 不会静默替换为监督预测。

## 4. 端到端架构

### Layer 0：数据和身份边界

- 按 dataset、subject/source identity、task、language、modality 和 speaker role 建立记录。
- 同一患者或同一来源的录音必须在同一 partition。
- 精确 raw recording hash 冲突会阻断受影响 cohort。
- 排除记录也参与重复/碰撞检查，防止“先排除再漏检”。
- 真值标签使用单独 typed label reader，只允许训练和 scorer 读取。

主要实现：

- `src/advoice/pilot/data.py`
- `src/advoice/pilot/labels.py`
- `src/advoice/pilot/contracts.py`
- `configs/datasets/*.yaml`
- `configs/channels/*.yaml`

### Layer 1：基础处理与表征

- 音频、转录、ASR 来源、任务和说话角色分别保留。
- 深度声学和文本 embedding 是 model-only 表示，不能直接写成疾病机制。
- 手工指标、deep embeddings、质量指标和来源信息不能混成一个无类型向量。
- 当前历史系统包含 mHuBERT、multilingual text embedding、声学、语言、对话和任务特征；具体启用项由配置和可观察性决定。

主要实现：

- `src/advoice/features.py`
- `src/advoice/text_representation.py`
- `src/advoice/audio_embeddings.py`
- `src/advoice/routing.py`
- `src/advoice/pipeline.py`

### Layer 2：MetricEvidence

每个指标对象保存：

- raw value 和 unit；
- abnormal direction；
- fold-local reference scope；
- reliability components；
- missingness/availability；
- confound tags；
- task、role、asset 和 segment provenance；
- inference permission 与 report permission；
- 是否已经被监督模型使用。

质量变量可以影响可靠性或触发限制，但不能作为疾病证据。设备、音量、MFCC 或 embedding 维度即使具有预测力，也不能自动写成临床机制。

主要实现：

- `src/advoice/evidence.py`
- `schemas/metric_evidence_v2.schema.json`
- `configs/metrics/`
- `skills/ad_evidence_diagnostic/EVIDENCE_REGISTRY.csv`

### Layer 3：StateCard 和任务可观察性

- 指标先按 correlation family 分组，避免同源或公式相关指标重复投票。
- 使用训练折内健康参考构建方向化标准分数。
- shared state 与 task-specific residual state 分开保存。
- missing 或 task-unobservable 不得当作正常值。
- StateCard 保存支持证据、反证、confounds、coverage、confidence 和 segment provenance。

主要实现：

- `src/advoice/states.py`
- `src/advoice/state_graph.py`
- `configs/states/`
- `configs/observability/`
- `skills/ad_evidence_diagnostic/STATE_KNOWLEDGE.md`
- `skills/ad_evidence_diagnostic/TASK_OBSERVABILITY.md`

### Layer 4：监督基础和 OOF 学习

监督模块负责从 text/audio/state/segment 等分支学习疾病标签关联。当前 pilot 要求：

- 患者/来源分组的 folds；
- 每折单独拟合 reference、imputation、normalization、feature selection 和 model；
- 完整 development OOF 覆盖后才能拟合 calibrator；
- final refit 与 OOF artifact、dataset、channel、task、class order 和 source identity 绑定；
- holdout label 不能进入拟合；
- 不允许跨数据集借用 calibrator。

HC/MCI/AD 使用两个头：

- impairment head：HC versus impaired；
- conditional stage head：MCI versus AD（这不是 Alzheimer 病理分期）。

主要实现：

- `src/advoice/condition_c.py`：历史 Condition C 主训练路径；
- `src/advoice/pilot/learning.py`：新版 pilot fold、OOF、receipt、calibrator 和 matched baseline；
- `configs/models/default.yaml`。

### Layer 5：prior-blind Agent 和 evidence replay

Agent 输入包含 sanitized transcript、MetricEvidence、StateCard、counterevidence、任务/角色和质量限制，但不包含监督概率和真值。

Agent 可以：

- 输出 HC/MCI/AD 或 route classes 的 ordinal evidence support；
- 引用 evidence/state IDs；
- 请求 deterministic remeasurement；
- 请求 source role/span correction；
- 标记 unsupported interpretation；
- 标记 confound。

Agent 不能：

- 创建不存在的测量、参考范围或病史；
- 直接改 raw value；
- 直接写入最终概率；
- 使用质量控制项作为疾病证据；
- 使用过期或其他病例的 evidence ID。

Executor 授权操作后生成 child snapshot，撤销受影响对象并重新计算。有效修改后必须进行 fresh v1 assessment，不能沿用 v0。最多三项操作和两轮 semantic assessment。

主要实现：

- `src/advoice/pilot/runtime.py`
- `src/advoice/authority_review_runtime.py`
- `src/advoice/conditional_authority.py`
- `skills/ad_evidence_diagnostic/`
- `schemas/agent_decision.schema.json`
- `schemas/cognitive_trace.schema.json`

### Layer 6：联合校准和五个实验臂

五个臂是：

| Arm | 含义 |
| --- | --- |
| `B_raw` | 未校准监督基础 |
| `B` | OOF 校准后的监督基础 |
| `J-A` | 基础 + Agent 独立类别证据 |
| `J-S` | 基础 + state replay log-odds delta |
| `J-AS` | 基础 + 当前 Agent 判断 + state replay delta |

联合模型特征顺序固定为：`base_log_odds, agent_contrast, replay_log_odds_delta, intercept`。非截距系数非负，L2 prior 是 `[1, 0, 0, 0]`。这使无稳定增量的 Agent/replay 系数可以回到零，而不是因为监督模型不确定就自动放大 Agent。

Matched baseline 在与 J-AS 完全相同的有效病例上重新评估 B，避免通过缺失病例改变分母制造 Agent 增益。

主要实现：

- `src/advoice/pilot/learning.py`
- `configs/pilot/evidence_state_v1.yaml`
- `src/advoice/authority_joint_fusion.py`
- `configs/calibration/authority_registry.json`

### Layer 7：锁定、评分和报告

- 预测锁绑定 code/config/data/model/calibrator/snapshot hashes。
- predicted class 从概率和固定 class order 派生，不能信任相互矛盾的字符串字段。
- 报告只读取 locked aggregate，不重新训练、不重算指标、不删除失败病例。
- Agent/report failure、fallback、abstention、token 和费用保留在分母中。
- Layer A 评估预测；Layer B 评估证据链和系统行为。

主要实现：

- `src/advoice/pilot/reporting.py`
- `src/advoice/pilot/runner.py`
- `templates/pilot/evaluation_report.html.j2`
- `templates/pilot/system_report.html.j2`
- `src/advoice/diagnostic_agent_report.py`

## 5. Training 与 post-training 的边界

本项目当前没有对 GPT/Luna/Sol/Astra 做模型权重微调，也没有 LLM post-training。

实际训练的是：

1. 监督基础分支和 Condition C；
2. fold-local preprocessing/reference；
3. OOF base calibration；
4. Agent/replay joint calibrator；
5. 数据集特异的筛查头和条件 stage head。

Agent 目前使用 versioned skill、prompt、工具和运行时规则进行 inference-time reasoning。更换 Agent 模型不是 post-training；prompt 更新也不是 post-training。未来若积累医生审阅的 evidence action、修订和最终判断数据，才可能做 SFT/preference optimization，但必须与当前实验分开版本化。

## 6. 数据集与 channel

仓库配置了十个任务配置，并不等于十个完全独立的临床 cohort。

| 数据集配置 | 主要 channel / task | 当前定位 |
| --- | --- | --- |
| `ADReSS_2020` | English picture description | 独立监督 pilot；历史 internal holdout |
| `ADReSSo_2021_diagnosis` | audio-only picture description | 历史评估 |
| `ADReSSo_2021_progression` | longitudinal progression audio | 压力/失败模式；不是普通诊断标签 |
| `DementiaBank_Pitt` | TalkBank neuropsychological multitask | 需防来源/质量捷径 |
| `DementiaNet_PublicFigures` | public natural speech | 人物隔离工程测试，样本很小 |
| `IAEAV` | Spanish clinical interview | acquisition/interviewer confounding 压力集 |
| `NCMMSC2021_AD` | Mandarin long picture description | 只使用长录音；排除 6 秒轨道 |
| `PREPARE_DrivenData` | multilingual structured tasks | SpeechCARE 主要比较数据集 |
| `PROCESS_2` | structured multitask | 历史三分类评估 |
| `TAUKADIAL` | Mandarin-English spontaneous speech | 跨语言/ASR 压力测试 |

Pilot 计划中的受控规模：

- PREPARE：120 development/training + 30 frozen internal evaluation；不触碰 412 official test 做模型选择。
- ADReSS 2020：65 + 16；排除已经查看过的历史 27 holdout。
- NCMMSC：96 + 24；长录音；排除 6 秒轨道和官方 53。
- IAEAV：14，压力测试，不新拟合跨域 calibrator。
- DementiaNet：6，人为隔离工程测试，不作临床效能结论。

原始音频、患者转录和受限标签不在 Git 仓库中。`references/speechcare/` 有上游公开表格和预测，但公开可见不等于自动允许再次发布。

## 7. Evaluation 合同

### Layer A：预测和校准

包括 accuracy、balanced accuracy、macro/weighted F1、MCC、macro/micro/weighted AUROC、AUPRC、Brier、log loss、ECE、校准参数、高置信错误、阈值行为、每类 sensitivity/specificity，以及 paired bootstrap/McNemar。

新版 pilot 还要求：

- `J-*` 与 exact matched B 比较；
- HC/MCI/AD 每类 help/harm；
- 缺失/失败/fallback 不从分母消失；
- 至少报告模型、prompt、skill、dataset 和 cost identities。

### Layer B：证据和系统完整性

包括：

- MetricEvidence/StateCard completeness；
- report permission；
- evidence-to-segment faithfulness；
- snapshot/hash/receipt continuity；
- state intervention 和重算；
- fallback、abstention 和 API failure；
- token/cost accounting；
- report structure audit。

Layer B 不是医生验证，也不是临床解释有效性的替代物。

## 8. 当前实证结果

### 8.1 PREPARE 与 SpeechCARE

历史独立 ADvoice PREPARE：

- accuracy / micro F1：`0.672330`
- micro AUROC：`0.846162`

仓库采用的 SpeechCARE 论文均值：

- accuracy / micro F1：`0.7211`
- micro AUROC：`0.8683`

当前独立系统没有超过 SpeechCARE。accuracy 差约 4.88 个百分点。

历史 released-prediction + cognition extension 的 accuracy 是 `0.735437`，但它使用 SpeechCARE 发布输出，不是独立训练的 ADvoice，不能作为本系统超越 SpeechCARE 的证据。

2026-09 受控 PREPARE 候选在 official test 的一次 retrospective 结果：

- accuracy：`0.665049`
- macro F1：`0.558180`
- macro AUROC：`0.804681`
- micro AUROC：`0.846914`

该候选已明确不推广。official test 已被多次查看，不能再作为新的确认性验证集。

### 8.2 四个 processed-input 工程 pilot

| Pilot | Test n | Historical accuracy | Current accuracy | Current macro AUROC |
| --- | ---: | ---: | ---: | ---: |
| ADReSS2020 internal holdout | 27 | 0.851852 | 0.925926 | 0.978022 |
| ADReSSo progression | 12 | 0.750000 | 0.750000 | 0.555556 |
| NCMMSC long | 53 | 0.811321 | 0.773585 | 0.921855 |
| PublicFigures | 6 | 0.666667 | 0.666667 | 0.500000 |

这些运行禁用了 live Agent。它们说明工程可运行和不同 failure modes，不能证明新版 Agent 增益。NCMMSC accuracy 下降必须保留，不能只报告 ADReSS 的提升。

### 8.3 新版 evidence-state Agent pilot

尚无真实 provider/患者 pilot 结果。当前只有软件、组合矩阵和 dry-run 验证。因此：

- Agent 增益未成立；
- “更强模型带来更大增益”未验证；
- “超过 SpeechCARE”未验证；
- 不得把 `1298 passed` 写成临床性能。

## 9. 本轮已经完成的工程修复

### 数据与身份

- 所有 inventory 记录在排除前做 recording collision 检查。
- 只阻断受影响 cohort，不污染其他数据集。
- source identity、raw hash、dataset/task/class/channel 进入工件绑定。

### Learning 与 calibration

- 完整 development OOF membership 强制检查。
- base、Agent assessment、trace、replay、fusion 和 probability 都有 sealed receipts。
- 阻止 final-refit prediction 进入 calibrator。
- 阻止重命名训练身份伪装 holdout。
- calibrator 绑定 dataset/version/channel/task/classes。
- matched-fit baseline API 已实现。

### Agent runtime

- prior-blind packet 和 transcript quarantine。
- executor registry/skill/prompt/snapshot 内容哈希。
- semantic call 与 transport attempt 分开记账。
- global、subject 和 comparison-cell 预算。
- 预留 worst-case token/cost；零预算等于零调用。
- 缺失 usage/cost telemetry 立即 fail closed。
- timeout 必须完成取消和 terminal acknowledgement 后才能 retry。
- snapshot/skill/registry/packet integrity drift 会停止队列。

### Runner 与报告

- 无真实 adapter 时 stage 为 `blocked`，不能假装成功。
- 非评分阶段不读取或哈希 sealed labels。
- dry-run 输出不能作为真实运行依赖。
- resume 校验 code/config/input hashes。
- score/render 有 schema/hash locks。
- 空 aggregate 不能标记为 complete。
- 概率派生 predicted class；修复 zero-F1 和 per-class help/harm。

## 10. 验证记录

在候选实现检查点上：

```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
1298 passed, 3 skipped
```

Pilot DAG 干运行：

```bash
PYTHONPATH=src python scripts/run_evidence_state_pilot.py \
  --config configs/pilot/evidence_state_v1.yaml \
  --run-dir .local/handoff-dry-run-2026-10-08 \
  --dry-run --provider fake
```

13 个阶段均为 `dry_run`：

```text
inventory -> split -> canary -> fit-oof -> assess-oof -> fit-fusion
-> fit-final -> predict-holdout -> stress -> model-comparison -> score -> render
```

没有真实 API、训练或患者推断。

## 11. 当前阻断项

1. 候选分支没有合并 `origin/main` 的 8 个提交，也没有推送；需要先做整合 PR。
2. `configs/pilot/evidence_state_v1.yaml` 是 disabled scaffold：manifests、routes、provider、model、pricing、skill/prompt hashes 和正预算都未解析。
3. `configs/calibration/authority_registry.json` 没有一个经过独立审查并满足条件的非零 authority artifact。
4. 新版真实 provider adapter 和生产 learning adapter 尚未在冻结 pilot 中通过最终独立 review。
5. PREPARE official test 已被反复查看，只能做 retrospective comparison。
6. Agent 增益和模型能力 scaling 尚无真实结果。
7. StateCard 的医学内容效度没有医生验证。
8. 自动 report rubric 不是医生评分。
9. IAEAV、Pitt、NCMMSC 存在 acquisition/quality shortcut 风险。
10. `docs/VALIDATION_STATUS.md` 仍写旧的 325 tests，已过时；README 也把 GitHub main 描述成维护框架，但最新 pilot 尚未在 main。
11. 当前工作树有未跟踪的 LaTeX 编译产物和上传 ZIP，不应不加审查地提交。

## 12. Git 与 GitHub 交接

### 远程主线

```text
origin/main = 88a887c
```

主线包含本候选分支缺少的 8 个历史 audit/governance 提交。部分内容在候选分支上存在不同 SHA 的同类提交，因此整合时可能出现内容重复，不能直接无脑 merge。

### 当前候选

```text
branch = feat/evidence-state-pilot-v1
implementation checkpoint = 256199a6628c60ce2c03ce6fdaea993fe0b9410d
ahead/behind origin/main = 30/8
```

下一位 Agent 应：

1. 新建备份分支或 worktree。
2. 比较 `origin/main..feat/evidence-state-pilot-v1` 的重复 audit/report commits。
3. 优先保留 pilot contracts/data/learning/runtime/runner/reporting 修复。
4. 将 `origin/main` 的 channel-unobservable 和 governance 更新显式整合。
5. 跑全量测试和 dry-run。
6. 独立 review 后创建 PR；不要直接 force-push main。

## 13. 仓库目录地图

| 路径 | 内容 |
| --- | --- |
| `src/advoice/` | 主 pipeline、features、states、models、evaluation、reports |
| `src/advoice/pilot/` | 新版 evidence-state pilot contracts/data/learning/runtime/runner/reporting |
| `configs/` | dataset/channel/model/Agent/evaluation/pilot 配置 |
| `schemas/` | evidence、Agent、trace、decision、report JSON contracts |
| `skills/ad_evidence_diagnostic/` | AD scope、state knowledge、observability、confounds、permissions |
| `tests/` | 单元、泄漏、运行时、校准、runner、报告和回归测试 |
| `scripts/` | 明确的实验、审计、冻结和 pilot 入口 |
| `demo/` | 合成四通道 demo 和本地 web server |
| `reports/` | 历史/会议报告；不等于新实验结果 |
| `paper/overleaf_upload_2026-09-24/` | 本地论文和图 |
| `.local/` | 不入 Git 的 dry-run、私有工件和临时输出 |

## 14. 常用命令

安装：

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

测试：

```bash
make test
```

公开合成 demo：

```bash
make demo
# http://127.0.0.1:8765
```

管理型实验配置检查与运行：

```bash
make experiment-check CONFIG=/absolute/path/to/private/experiment.yaml
make experiment CONFIG=/absolute/path/to/private/experiment.yaml
```

单数据集：

```bash
make validate DATASET=ADReSS_2020
make full DATASET=ADReSS_2020
make evaluate DATASET=ADReSS_2020
make report DATASET=ADReSS_2020
```

完全 Agent-led 合成例子：

```bash
advoice agent-led \
  --workspaces demo/agent_led/synthetic_workspace.jsonl \
  --labels HC AD \
  --provider openai_api \
  --output-dir .local/agent-led-demo
```

Evidence-state pilot 只读干运行：

```bash
PYTHONPATH=src python scripts/run_evidence_state_pilot.py \
  --config configs/pilot/evidence_state_v1.yaml \
  --run-dir .local/pilot-dry \
  --dry-run --provider fake
```

没有 T7 ACCEPT、resolved config 和正预算时，不得加 `--allow-paid`。

## 15. 当前可查看报告和论文

- 老师汇报 HTML：`reports/meeting_ready_2026-09-24/index.html`
- 中文口语稿：`reports/meeting_ready_2026-09-24/oral_report_zh.md`
- 五臂历史验证：`reports/five_arm_validation_2026-09-17/`
- 多数据集历史分析：`reports/latest_evidence_agent_multidataset_2026-09-17/`
- Post-Agent 历史审计：`reports/post_agent_evaluation_2026-09-17/`
- 论文源码：`paper/overleaf_upload_2026-09-24/main.tex`
- 论文修订说明：`paper/overleaf_upload_2026-09-24/REVISION_STATUS.md`

论文明确区分历史 evaluation 与新版待运行 pilot。不要把历史 Layer A/B 图当作新版 Agent 增益。

## 16. 下一位 Agent 的具体执行顺序

### Phase A：代码整合，不跑真实数据

1. 整合 `origin/main` 和 `feat/evidence-state-pilot-v1`。
2. 更新 README、VALIDATION_STATUS、SYSTEM_REVIEW 的测试数和真实状态。
3. 跑 `1298+` 全量测试；新增主线测试也必须通过。
4. 运行 pilot dry-run，检查 13 个 stage hashes 和 resume。
5. 再做一次独立代码审查，重点检查 labels、OOF receipts、budget、score denominator 和 render lock。

### Phase B：冻结真实 pilot 配置

1. 解析 PREPARE、ADReSS、NCMMSC manifests 和 source identities。
2. 只使用长 NCMMSC；阻断无法解决的 collision。
3. 写定 provider/model/date/pricing/skill/prompt/extractor/state hashes。
4. 写定正的 global/subject/cell token 和 USD ceilings。
5. 注册生产 adapters，禁止 synthetic adapter 自我声明可信。
6. T7 review 得到 ACCEPT 后才运行 canary。

### Phase C：受控 pilot

1. 先 24-call canary，验证 telemetry、cache、取消和报告。
2. 分别训练 PREPARE、ADReSS、NCMMSC；不混库训练。
3. 完整 development OOF 后拟合 B/J-A/J-S/J-AS。
4. 冻结 calibrators 后一次预测 internal holdout。
5. IAEAV/DementiaNet 只做 stress，不拟合跨域 calibrator。
6. 报告 exact matched Agent gain、每类 help/harm、fallback 和成本。

### Phase D：科学结论

只有满足以下条件才能写 Agent 增益：

- J-A/J-AS 相对 matched B 有稳定配对增益；
- 增益不是普通 calibration 或 missing-row denominator 造成；
- HC/MCI/AD 不通过牺牲单类换总体分数；
- evidence/state 修改真实改变预测；
- 所有失败、fallback 和成本进入分母；
- 更强模型比较使用相同 cohort、skill、policy 和预算。

SpeechCARE 超越只能在同协议、同 cohort、同 endpoint、冻结后的 retrospective 比较中描述；确认性结论需要新的外部 cohort。

## 17. 禁止事项

- 不用 official test 反复选阈值、prompt、skill 或权重。
- 不把 SpeechCARE released predictions + cognition 写成独立 ADvoice。
- 不把软件测试写成临床有效性。
- 不把 embedding、设备或质量变量写成疾病机制。
- 不把 unavailable state 当正常值。
- 不删除失败 Agent 调用或 abstention 来提高 accuracy。
- 不把 Agent ordinal support 直接解释为概率。
- 不把私有录音、转录、标签、API 日志或密钥提交 Git。
- 不再复制新的日期系统文件夹；统一在 GitHub 仓库分支和隔离 workspace 工作。

## 18. 建议先读的文件

1. `README.md`
2. `docs/CALIBRATED_SINGLE_AGENT_FRAMEWORK.md`
3. `docs/AGENT_LED_ARCHITECTURE.md`
4. `docs/SYSTEM_REVIEW.md`
5. `docs/RESEARCH_WORKSPACE.md`
6. `docs/superpowers/plans/2026-09-24-agent-pilot/CONTRACTS.md`
7. `docs/superpowers/plans/2026-09-24-agent-pilot/START_HERE.md`
8. `configs/pilot/evidence_state_v1.yaml`
9. `src/advoice/pilot/contracts.py`
10. `src/advoice/pilot/data.py`
11. `src/advoice/pilot/learning.py`
12. `src/advoice/pilot/runtime.py`
13. `src/advoice/pilot/runner.py`
14. `src/advoice/pilot/reporting.py`
15. `tests/test_pilot_integration.py`

## 19. 最终交接判断

项目已经从“监督模型给概率、Agent 写报告”推进到“监督基础 + prior-blind Agent + typed evidence replay + OOF 联合校准 + locked report”的可测试架构。Trace evidence 的软件链路已经较完整：source segment、MetricEvidence、StateCard、assessment、operation、child snapshot、prediction 和 report 可以绑定并审计。

尚未完成的是最关键的实证环节：真实 Agent 是否相对 matched supervised baseline 有净增益，以及这种增益是否能在不同 channel、语言和数据集上稳定复现。下一位 Agent 的首要任务不是继续美化报告，而是先合并代码主线、冻结真实 pilot、通过独立 gate，并产生第一份合法的 J-A/J-S/J-AS 对照结果。
