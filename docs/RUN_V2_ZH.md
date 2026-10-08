# v2 实验运行说明（PREPARE，三家模型）

分支：`refactor/provider-neutral-harness`。仓库：`https://github.com/Jewelina95/advoice-research-harness`。

## 0. 准备

```bash
git clone https://github.com/Jewelina95/advoice-research-harness.git
cd advoice-research-harness
git checkout refactor/provider-neutral-harness
python3 -m pip install -e ".[dev]" mlx-whisper   # mlx-whisper 只在 Apple Silicon 上需要
```

输入数据：9.2 的 PREPARE 缓存目录（特征、状态、embedding、ASR 转录、manifest），下文记为 `$ART`：

```bash
export ART="/Users/wenshaoyue/Desktop/research/ad general/AD voice/9.2/artifacts/PREPARE_DrivenData"
```

所有输出写到 `.local/v2/`（不进 Git）。

## 1. 模型与认证

| 模型 | provider 参数 | 模型参数 | 认证 |
| --- | --- | --- | --- |
| GPT-5.5 | `openai_api` | `gpt-5.5-2026-04-23` | 环境变量 `OPENAI_API_KEY` |
| Claude Opus 5.5（API） | `anthropic_api` | `claude-opus-5-5` | 环境变量 `ANTHROPIC_API_KEY` |
| Claude Opus 5.5（本机 CLI，无需 API key） | `claude_cli` | `claude-opus-5-5` | 先在终端运行一次 `claude` 并输入 `/login` |
| DeepSeek | `deepseek_api` | 按 DeepSeek 当前模型名 | 环境变量 `DEEPSEEK_API_KEY` |

三家使用同一份提示、同一份输出 schema、同一推理强度（`ADVOICE_AGENT_EFFORT`，默认 `low`）。拒答、超时、schema 不合格一律记为失败，不自动换模型。

## 2. 步骤

### 2.1 Whisper 编码器特征（一次，约 1–2 小时，M2 Max）

```bash
python3 scripts/v2/whisper_encoder_embeddings.py --manifest "$ART/manifest.csv" --out-dir .local/v2/whisper
```

输出 `whisper_l8/16/24/32.npz`（第 8/16/24/32 层的 mean‖std）。

### 2.2 LLM 臂（每个模型各跑两次）

- `--arm B`：纯 Agent，只看转录，直接给 HC/MCI/AD 概率。
- `--arm C`：框架测量，按知识库测内容单元、偏题、空泛词、语义错误、找词困难、连贯、句法、ASR 质量，以及独立序数判断。

```bash
# GPT（另一个窗口也可以跑这两条）
python3 scripts/v2/agent_measure.py --artifacts "$ART" --arm C --provider openai_api --model gpt-5.5-2026-04-23 --out-dir .local/v2/agent --workers 6
python3 scripts/v2/agent_measure.py --artifacts "$ART" --arm B --provider openai_api --model gpt-5.5-2026-04-23 --out-dir .local/v2/agent --workers 6

# Claude（本机 CLI）
python3 scripts/v2/agent_measure.py --artifacts "$ART" --arm C --provider claude_cli --model claude-opus-5-5 --out-dir .local/v2/agent --workers 4
python3 scripts/v2/agent_measure.py --artifacts "$ART" --arm B --provider claude_cli --model claude-opus-5-5 --out-dir .local/v2/agent --workers 4

# DeepSeek
python3 scripts/v2/agent_measure.py --artifacts "$ART" --arm C --provider deepseek_api --model <deepseek-model> --out-dir .local/v2/agent --workers 6
python3 scripts/v2/agent_measure.py --artifacts "$ART" --arm B --provider deepseek_api --model <deepseek-model> --out-dir .local/v2/agent --workers 6
```

- 每 12 例一批；每批按请求内容哈希缓存，中断后重跑同一命令只补跑缺失的批，不重复付费。
- 先加 `--limit 12` 做冒烟测试。
- 每批的 token、耗时、费用写在 `batch_*.json.calls.jsonl`；结果在 `<arm>_<provider>_<model>/results.csv`。
- 发送内容只有化名 ID、任务类型和转录；不含标签、人口学和路径。PREPARE 数据使用协议是否允许外发需确认。
- GPT-5.5 全量参考用量：C 臂约 32 万输入 / 22 万输出 token；B 臂约 25 万 / 6.5 万。

### 2.3 评估

```bash
G=.local/v2/agent
python3 scripts/v2/prepare_v2.py --artifacts "$ART" --out .local/v2/result.json --seeds 3 \
  --extra whisper24=.local/v2/whisper/whisper_l24.npz \
  --agent-c gptC=$G/C_openai_api_gpt-5.5-2026-04-23/results.csv claudeC=$G/C_claude_cli_claude-opus-5-5/results.csv \
  --agent-b gptB=$G/B_openai_api_gpt-5.5-2026-04-23/results.csv claudeB=$G/B_claude_cli_claude-opus-5-5/results.csv
```

没有的模型去掉对应参数即可。各臂含义：

| 臂 | 组成 |
| --- | --- |
| `A_acoustic` | 手工声学特征 |
| `A_plus_text` | 声学 + 文本指标 |
| `E_embeddings` / `E_plus_demo` | 冻结 mHuBERT + 文本 embedding（+ 年龄、性别、语言、任务） |
| `S_framework` / `S_framework_z` | 声学 + 文本 + 状态 + embedding + 人口学（+ 按任务×语言×年龄分层的健康参考 z） |
| `S_framework_z+whisper24` | 再加 Whisper 编码器分支 |
| `B_<x>_direct` | 纯 Agent 概率直接评估 |
| `C_<x>C` | 框架 + Agent 测量 |
| `S+<x>B_judgment` | 框架 + 纯 Agent 判断作为堆叠候选 |

每个分支在训练集上 5 折 OOF 选正则，再用多项逻辑回归堆叠；官方 412 例测试集只用于评估。PREPARE 官方测试集历史上已被多次查看，结果只能作为回顾性比较。

## 3. 当前结果（2026-10-08，1 个种子，GPT-5.5）

| 臂 | Accuracy | macro-F1 | micro AUC | log-loss | 召回 HC/MCI/AD |
| --- | --- | --- | --- | --- | --- |
| B 纯 GPT | 0.556 | 0.419 | 0.698 | 1.068 | 0.81 / 0.25 / 0.23 |
| A 声学 | 0.583 | 0.399 | 0.764 | 0.892 | 0.91 / 0.14 / 0.18 |
| A + 文本 | 0.631 | 0.507 | 0.815 | 0.805 | 0.91 / 0.29 / 0.27 |
| 冻结 embedding + 人口学 | 0.675 | 0.593 | 0.857 | 0.708 | 0.87 / 0.43 / 0.43 |
| S 框架 + 分层 z | 0.680 | 0.570 | 0.860 | 0.703 | 0.89 / 0.29 / 0.46 |
| C = S + GPT 测量 | 0.682 | 0.573 | 0.860 | 0.703 | 0.89 / 0.29 / 0.47 |
| S + GPT 判断 | 0.687 | 0.591 | 0.860 | 0.703 | 0.89 / 0.35 / 0.46 |
| SpeechCARE 论文均值 | 0.721 | — | 0.868 | — | — |

结论见 `V2_PLAN_ZH.md` 的更新，以及会话中的总结。
