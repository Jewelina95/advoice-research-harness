# 最新版 Evidence Agent 五组消融：中文口语稿

上一轮完整联合效果差，是因为使用了旧版只读转录的 B2，并不是真正的 Evidence Agent。这次我们直接使用 GitHub main 最新的联合函数，并选择 PREPARE 中九个已经完成完整证据审查的病例。

五组使用完全相同的监督概率、审查前后 StateCard、Agent 盲态判断和标签顺序。区别只在于是否开启状态修正、Agent screening 和 Agent staging 权限，因此这次消融真正对应当前架构。

仅监督模型的 Accuracy 是 0.778，Macro-F1 是 0.679。仅 Agent 的效果较差，说明 Agent 不能脱离监督主干独立诊断。加入状态修正后，Accuracy 提高到 0.889，Macro-F1 提高到 0.800。加入 Agent 独立判断后，Accuracy 同样是 0.889，但 Balanced Accuracy 和 Macro-F1 分别达到 0.929 和 0.862。完整联合模型在这九个病例中达到 1.000，同时 log loss 从监督模型的 0.712 降到 0.549。

因此当前可以选择完整联合模型作为下一阶段候选。它的优势不是让 Agent 取代监督模型，而是把三种信息分工：监督模型提供稳定先验；StateCard 修正证据结构；Agent 提供独立的 screening 和 staging 证据。三者在病例级门控下共同决定结果。

这仍然只是九例 pilot，不能用于声称超过 SpeechCARE。正式结论需要冻结这套五组定义，在开发集校准三个权限参数，再到未参与选择的 PREPARE 同协议测试集一次性评价。
