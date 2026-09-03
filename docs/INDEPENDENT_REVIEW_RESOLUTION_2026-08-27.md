# 8.27 方案独立审查与修复记录

## 审查边界

8.20/8.21 只作为评估结果报告的信息组织和图表呈现参考，不作为 8.27 代码、参数、模型结果或中间产物的实现基线。本轮没有运行数据处理、模型训练、Agent 批处理或性能评估。

## 审查方法

由独立审查 Agent 对方案、Skill、Schema、fixtures 和静态验证器进行五轮只读攻击。审查不只读文档，还实际修改内存中的样例构造反例，检查错误是否会被拒绝。

## 已关闭的主要缺口

1. P1 数据集内基准、P2 共享认知状态主模型和 P3 泛化/特殊终点不再混用。
2. 8.27 不继承旧代码；旧报告只提供呈现参考。
3. subject/session/task/segment 引用被锁定，禁止跨任务拼接证据。
4. `metric_id -> state_id` 由注册表锁定，禁止把词汇指标挂到停顿状态。
5. 普通分类、纵向配对分类、序数分类和连续回归使用不同终点契约和修正边界。
6. 真实 JSON Schema Draft 2020-12 校验替代了只检查 required 字段的伪校验。
7. 病例包到医生报告共享 run/config/data/Skill provenance，trace ID 和模块输入输出必须逐级解析。
8. 训练折参考必须解析到 trusted artifact registry，校验实际文件 SHA-256、fold、population 和状态。
9. conditional 报告权限必须具有任务、语言和方法一致的可信验证产物，不能由病例包自称通过。
10. 显式诊断披露片段不能进入临床指标、模型辅助特征、embedding、StateCard 或 Agent 预测上下文，只能用于 QC/audit。
11. CooT 风格回退会检查证据/状态存在性、回退目标、validator provenance、违规重试和终止动作。
12. 模块 B 的 correction type 必须与二分类、多分类/序数、纵向分类或回归终点一致。
13. 医生报告由锁定字段和版本化模板渲染；结论、概率、数值必须与 locked decision 一致，全部叙述字段接受中英文越界诊断扫描。

## 最终复现

独立审查 Agent 最后一轮只复现剩余的三个反例：

- 诊断披露片段经 StateCard 绕过；
- 越界确诊语句经 recommendations 绕过；
- 中文越界确诊语句经 finding 绕过。

三项均被拒绝，独立审查结论为 `PASS`。

本地验证命令：

```bash
/tmp/advoice827-schema-venv-2/bin/python scripts/validate_skill.py
python3 -m py_compile scripts/validate_skill.py
```

输出：

```text
PASS: initial Skill, schemas, fixtures and safety checks are internally consistent
NOTE: no training or clinical-performance evaluation was run
```

## 仍属于正式训练前门槛

这些不是当前方案遗漏，而是必须由真实数据和预注册协议完成的下一阶段工作：

1. 生成并冻结真实 `dataset_manifest`、`endpoint_harmonization`、`split_manifest` 和 `pooling_eligibility`。
2. 将 `REFERENCES.md` 的键解析到完整 primary sources；`R_PROJECT` 保持项目假设，未经验证不得进入医学解释。
3. 为 conditional metrics 运行真实任务/语言/方法验证并生成可信 artifact，而不是使用示例凭证。
4. 在训练前批准 seed、bootstrap、多重比较、选择性覆盖和模块 B 非劣效边界。
5. 比较乘积、几何均值、最小项和折内校准四种可靠度聚合器；外层测试不得参与选择。
6. 实现后为 P3 每类终点补齐模块 A 到医生报告的完整端到端 fixtures。
7. 医生评分目前不能假定存在；自动报告审计和未来真实医生评估必须分开标注。

## 当前判定

8.27 方案、Initial Skill 和跨模块契约已经可以提交用户确认。它尚不能被称为“模型已完成”或“临床有效”；正式实现和训练只能在上述门槛冻结后开始。

