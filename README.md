# ADvoice 9.2 research harness

[![Reproducibility checks](https://github.com/Jewelina95/advoice-research-harness/actions/workflows/ci.yml/badge.svg)](https://github.com/Jewelina95/advoice-research-harness/actions/workflows/ci.yml)

可复现的语音认知筛查研究工程。系统按数据集和受试者独立切分，比较三个条件：传统机器学习 `B1`、直接诊断大模型 `B2`、证据约束单一诊断 Agent `B3/Ours`。输出包括逐数据集报告、全数据集方法报告、Layer A 医学预测评估、Layer B 框架特异验证，以及医生版和患者/家属版病例说明。

本系统用于研究性认知风险筛查和转诊支持，不用于确诊阿尔茨海默病病理，也不输出未经验证的疾病阶段。

方法与数据流见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。当前完成范围、未完成数据集和 PREPARE 门禁状态见 [docs/VALIDATION_STATUS.md](docs/VALIDATION_STATUS.md)。

## Pipeline

1. 数据路由：识别数据集、任务、语言、模态和说话人角色。
2. 数据处理：患者语音隔离、ASR/转录规范化、任务感知切分、声学和语言指标抽取。
3. MetricEvidence：为每个指标保存值、训练折参考、方向、可靠度、缺失、混杂、任务、片段和报告权限。
4. StateCard：在训练折内部校准指标，生成总体及任务特异认知状态；同一状态的多个任务视图共享一票，避免重复投票。
5. 监督预测链：融合文本、言语行为、辅助声学和片段表示，产生交叉验证外的统计先验；它不进入 Agent 的首轮证据判断。
6. Agent 校准集：从训练数据的折外预测和折内参考值构造无标签泄漏的盲化证据工作区。
7. 单一诊断 Agent：只读取 MetricEvidence、StateCard、任务与片段轨迹、反证、质量限制和医学 Skill，输出 0–4 级类别证据判断及可验证证据 ID，不输出风险概率。
8. 验证集冻结融合：在保留开发标签的病例上选择一个共享修正强度；测试时以证据覆盖率、可靠度、混杂和病例路由控制对数意见池修正，接口错误自动回退到监督先验。
9. 输出：锁定预测后分别生成医生版和患者/家属版报告；报告不能修改预测，并必须通过证据角色、回溯、隐私和过度诊断校验。
10. 评估：Layer A 报告区分度、分类、校准和筛查操作点；Layer B 报告盲化、ID 合法性、修正增量、证据完整性、回溯、消融、干预和负控。

## Data

默认运行十个独立任务：`IAEAV`、`ADReSS_2020`、`ADReSSo_2021_diagnosis`、`ADReSSo_2021_progression`、`PROCESS_2`、`PREPARE_DrivenData`、`TAUKADIAL`、`DementiaBank_Pitt`、`DementiaNet_PublicFigures` 和 `NCMMSC2021_AD`。

`NCMMSC2021_AD` 只纳入 `AD_dataset_long` 的长录音。`AD_dataset_6s` 和无标签测试集在配置中明确排除；原始文件保留用于数据审计，不进入训练或测试。

原始数据不提交到 Git。`data/raw` 通过本地链接或配置连接到受控数据目录。

仓库中的公开语音案例由程序合成，不含患者信息。完整的数据边界和申请要求见 [DATA_AVAILABILITY.md](DATA_AVAILABILITY.md)。

## Public demo

公开演示用于验证“音频与转录 → 指标证据 → 认知状态 → 片段回溯”的工程链路。它使用合成语音，不加载临床诊断权重，也不生成疾病概率。

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
make demo
```

随后打开 `http://127.0.0.1:8765`。网页支持直接运行仓库样例，也支持在本地上传 WAV 和转录文本。接口和可复现边界见 [docs/PUBLIC_REPRODUCIBILITY.md](docs/PUBLIC_REPRODUCIBILITY.md)。

## Run

```bash
git clone https://github.com/Jewelina95/advoice-research-harness.git
cd advoice-research-harness
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
make test
make all-full
```

也可以双击 `RUN_ALL_FULL.command`。`all-full` 现在先完整运行 PREPARE，并检查 Micro AUROC、Micro F1 和 Micro AUPRC 是否全部超过 SpeechCARE 论文均值；任一门禁失败即停止，不运行其他数据集。该门禁只是回顾性工程门禁，不等于确认性优效证明。

只重建本轮 PREPARE 审计报告：

```bash
make prepare-audit
```

运行采用内容哈希缓存；源代码、配置或输入改变时，只重算受影响的阶段。任何数据集失败都会写入 `reports/batch_run_status.json`，生成可用的部分汇总后以非零状态退出，不再静默标记为全量成功。

## Outputs

- `reports/latest/index.html`：最新入口。
- `reports/latest/system_report.html`：数据、指标、状态、融合、Agent 和报告链路。
- `reports/latest/evaluation_report.html`：B1/B2/B3、Layer A、Layer B、负控和失败模式。
- `reports/latest/evaluation_oral_presentation_zh.md`：中文口语汇报。
- `reports/latest/prepare_9_2_method_audit.html`：七项 Agent 改造、相同编码器隔离和 SpeechCARE 门禁审计。
- `reports/datasets/<dataset>/latest/`：逐数据集报告和病例示例。
- `runs/<run_id>/`：不可变运行快照、配置哈希、源码清单和产物。

SpeechCARE 数值只在同一数据集、同一终点和兼容协议下作描述性对照。开发过程中反复查看过的测试集不能再被称为未触碰外部验证；任何性能提升必须由受试者级配对区间和预先定义的评估支持。
