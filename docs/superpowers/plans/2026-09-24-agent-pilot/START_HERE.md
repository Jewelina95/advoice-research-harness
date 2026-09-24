# Evidence-State Agent Pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and test a bounded, evidence-traceable Agent contribution against a matched supervised baseline across representative speech channels.
**Architecture:** Reuse existing measurement, evidence, state and model components. Add a small pilot orchestration layer with fold-safe fitting, a blind bounded Agent, replay, and regularized joint calibration. Keep the existing default production route until acceptance.
**Tech Stack:** Existing Python 3.11+, numpy/pandas/scikit-learn/scipy, pytest, Jinja2/matplotlib and configured model provider.

## 本轮交付范围

这是可派发的工程任务包，不是重新讲解研究思路，也不是已完成实现的声明。本轮只提交任务规范；实现和真实 API 运行由下面的任务执行。主入口为本文件，调度依据为 dispatch.yaml，所有 worker 必须遵守 CONTRACTS.md 和各自任务文件。

仓库：/Users/wenshaoyue/Desktop/research/ad general/AD voice/9.2/github_release/advoice-research-harness
原始数据：/Users/wenshaoyue/Desktop/research/ad general/AD voice/raw data/
历史工件：/Users/wenshaoyue/Desktop/research/ad general/AD voice/8.27/artifacts/

这是对 2026-09-23 两份计划的工程细化。算法原则不变；本任务包对执行顺序、文件归属和调用预算的规定优先。旧 TRAINING_PROTOCOL_P1_P2_P3.md 中向 Agent 展示基础概率或池化训练的安排不适用于本 pilot。

## 谁来做，什么时候做

| 顺序 | Agent / 编程模型 | 工作 | 交付与进入下一步条件 |
|---|---|---|---|
| T0 | 当前协调者 / Astra，high | 审核已有未提交修复；冻结接口和基线检查点 | checkpoint SHA、contracts、基线测试日志 |
| T1 | Data / Terra，high | 本地数据映射、患者去重、分层划分、缓存来源 | manifest、排除清单、泄漏测试 |
| T2 | Learning / Sol，high | 折内训练、OOF、两头融合及匹配基线校准 | 可复算模型工件、无泄漏训练测试 |
| T3 | Runtime / Sol，high | evidence 权限、盲判、状态修订重算、调用限额 | 冲突场景测试、版本有效性及调用账本 |
| T4 | QA / Terra，high | 独立构造端到端和组合测试 | 离线验收报告，不修改预测算法 |
| T5 | Presentation / Luna，medium | 固定 Layer A/B 汇报与交接渲染 | 离线 HTML；不得重算或选择性删结果 |
| T6 | Runner / Terra，high | 串联阶段、断点续跑、canary 与指定 pilot | 真正预测工件、usage、报告；不自行调算法 |
| T7 | Reviewer / Astra，high | 实现前放行、运行后科学结论复核 | ACCEPT / REVISE / BLOCK，逐项证据 |

不是同时开八个窗口。T0 完成后只并行 T1、T2、T3；集成后 T4/T5 并行；再 T6A 实现运行器、T7P 离线审查、T6B 执行数据、T7R 审核结果。T6A/T6B 属于同一 Runner 角色，T7P/T7R 属于同一 Reviewer 角色，但在调度中是四个独立节点，防止审核之前就开始付费运行。最多三名代码 worker。编程 Agent 的型号与病例推断模型分开配置，不因 Luna 写报告就让 Luna 自动承担医学推断。

## 协调者执行清单

- [ ] 读取 dispatch.yaml、CONTRACTS.md、任务文件和当前 git diff；不要重新读取整个历史对话。
- [ ] T0 生成 ACCEPTED_BASELINE.md。当前参考 HEAD 为 041f7e7261605395a328d3b6132addc1cb3c1410，不能忽略其后的未提交改动。
- [ ] 从已审查且包含必要未提交修复的 checkpoint 建立独立 worktree。不得直接从旧 HEAD 派发；不得自动 stash/reset 或一次性 add 全仓库。
- [ ] 对每个 worker 只传通用前缀、任务文件、contracts、checkpoint SHA。fork_context=false；不携带整段会话。
- [ ] T1/T2/T3 只能写 dispatch 中的文件；范围不够就返回变更请求，由 T0 决定，不能跨越所有权。
- [ ] 按 T1、T2、T3 顺序集成，冲突只由 T0 处理。保存每任务 commit SHA 与测试日志。
- [ ] T4/T5 完成后集成，T6 实现并离线验证运行器。T7 检查同一个候选 SHA。
- [ ] T7 放行后，T6 从 canary 开始，仅按阶段运行；不得跳过门禁直接跑九库。
- [ ] 完成三个独立训练 pilot 后做两个压力集，再做预注册模型对照；普通误判不是中途换样理由。
- [ ] 每轮失败回到对应文件 owner 修复；只重跑失效的阶段。禁止为了得高分反复改 holdout。
- [ ] T7 完成结果审核后再更新 GitHub 框架与不含私有病例的示例；原始音频、转录、标签、token、API 日志留本地。

## 数据执行规模

| 数据集 | 训练/开发 | 冻结内部评估 | 执行方式 |
|---|---:|---:|---|
| PREPARE | 120 | 30 | 从历史训练池 1622 人中锁定 150；不触碰 412 官方测试病例 |
| ADReSS 2020 | 65 | 16 | 81 人历史训练池；排除旧 27 holdout |
| NCMMSC2021_AD | 96 | 24 | 120 人官方训练池；长录音；排除整个 6 秒轨道及官方 53 |
| IAEAV | 不新训 | 14 | 采集批次隔离压力测试，不冒称新融合器已跨域校准 |
| DementiaNet | 不新训 | 6 | 人物隔离工程测试，不作临床效能结论 |

前三库分别训练、不混库。80% 内做五折 OOF，校准器不读取训练内预测；20% 不参与调 prompt、skill、公式。人数约束服从身份去重，偏差必须在 API 调用前写明。测试标签只能由冻结预测后的评分步骤读取。

## 派发消息模板

“在 checkpoint <实际 SHA> 上执行 <任务文件>。只读 CONTRACTS.md 和该任务列出的依赖。你只能修改 dispatch.yaml 中你的 write_files；不要启动付费 API 或其他任务。逐项完成 checkbox，运行 acceptance_commands，提交 WORKER_RESULT.md 指定内容。遇到接口冲突、来源不明或测试标签泄漏停止并返回证据，不自行扩大范围。”

T6B 的真实运行是例外，只在 T7P gate=ACCEPT、预算与模型 ID 已解析后开启。数据运行授权来自用户既有授权，不需要再问一次；不满足技术门禁则停止并报告原因。dispatch 的 write_files 是源码权限，artifact_write_paths 是本地交付物权限；checkpoint 与合并属于 T0 独有操作，不授权其他 worker 修改共享文件。

## 什么叫基本定下来

固定输入/输出接口、证据与状态职责、患者划分、两个小模块的训练边界、Agent 两轮上限、追溯和验收标准。第一版性能是假设，不是保证。允许模块升级，但必须改变版本、失效正确缓存、重走对应验收。不是“代码一次写完就必然超过 SpeechCARE”。

本 pilot 的主问题：相同输入、训练人数、编码器和划分下 J-AS 是否比 B 有净增益。SpeechCARE 同协议准确率竞争留到完整 development/官方评估阶段；此处不能用小样本高分替代。未达到性能目标时保留结果并定位原因，不把方案写成已成功。
