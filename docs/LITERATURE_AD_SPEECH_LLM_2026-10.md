# AD/MCI 语音筛查 + LLM 文献速览（检索日期 2026-10-08）

规则：每条均已打开所列 URL；PR = 同行评审，PP = 预印本，WEB = 非论文网页。原文未给出或本次未读到的数字标"未核实"。

## 1. 当前最强的语音 AD/MCI 检测系统

- **SpeechCARE**（Azadmaleki 等，npj Digit Med 2025，PR，DOI 10.1038/s41746-025-02026-x）。https://www.ebi.ac.uk/europepmc/webservices/rest/PMC12623413/fullTextXML 。mHuBERT（声学）+ mGTE（文本）+ 人口学变量，用 Adaptive Gating Fusion 融合，并对长音频做分段时序建模。PREPARE 数据 1,655 人（英/西/中），测试集 412 人：F1 72.11±0.44，micro-AUC 86.83±0.46。消融（F1/AUC）：纯声学 68.23/84.85，纯文本 68.88/85.00，语音+文本 AGF 70.51/86.57，cross-attention 70.51/86.61，intermediate fusion 70.10/86.28，AGF 加年龄 72.11/86.83。几种融合方式之间差距不到 0.5 F1，加入年龄带来 +1.6 F1。PREPARE Phase 2 log-loss：首次提交 0.6553，排第 4；Model Report 中为 0.6460。阈值优化后 MCI 召回率从 43.14% 升到 54.90%。在 ADReSSo 上 Acc 86.62%。编码器的微调细节本次未读完，标"未核实"。配套预印本 arXiv 2511.08132（PP）报告三分类 AUC 0.88、F1 0.72。
- **PREPARE Phase 2 声学赛道获奖方案**（DrivenData 博客，WEB）。https://blog.drivendata.org/blog/prepare-phase2-winners 。第 1 名 log-loss 0.6299：Whisper 编码器 + 临床特征，两阶段训练以削弱语言偏差，用 CAM 做解释。第 2 名 0.6343：语义聚类、官方声学特征、Whisper 微调、triplet 对比微调 Whisper（目标是对年龄、性别、语义组不变）四者集成。第 3 名 0.6523：融合声纹、音色和语义。三队的 F1/AUC 均未公开，标"未核实"。
- **Yuan 等**（Interspeech 2020，PR）。https://www.isca-archive.org/interspeech_2020/yuan20_interspeech.html 。在转录文本中编码停顿后微调 ERNIE，ADReSS 测试集 Acc 89.6%（基线 75.0%）。
- **ADReSSo 概述**（Luz 等，Interspeech 2021，PR）。https://www.isca-archive.org/interspeech_2021/luz21_interspeech.html 。基线 Acc 78.87%。冠军成绩该页未给出，标"未核实"。
- **TAUKADIAL 2024**（中英双语 MCI）。Drexel 团队（Brain Sciences 2024，PR，DOI 10.3390/brainsci14121292，经 Europe PMC 打开）用 Whisper 嵌入做集成，MCI 分类 UAR 81.83%，自报第 2；MMSE 回归 RMSE 1.196，自报第 1。FAU 新闻页（WEB，https://lme.tf.fau.de/?p=23174 ）称 FAU 赢得两项任务，未给数字。两方说法冲突，官方排名标"未核实"。
- **Amini 等**（Alzheimer's & Dementia 2024，PR，DOI 10.1002/alz.13886）。Framingham 队列 166 名 MCI，ASR 转写 + 语言模型嵌入 + 年龄/性别/教育，预测 6 年内由 MCI 进展为 AD：Acc 78.5%，敏感度 81.1%。

## 2. LLM / 智能体用于 AD 语音或转录文本

- **Heitz 等**（GPT-4 提取的语言特征，COLING 2025，PR；会议名取自 ACL Anthology 索引条目，未打开该页核对）。https://arxiv.org/html/2412.15772v1 。数据为 ADReSS，10 折交叉验证 AUROC，人工转录：传统特征 + GPT 特征 + RF 0.931，仅传统特征 + RF 0.885，微调 GPT-4o 0.886，GPT-4 零样本 0.827，GPT-4o 零样本 0.677，仅 GPT 特征 0.767。换成 ASR 转录后增益很小：Google 0.893→0.900，Whisper 0.874→0.886。结论：LLM 单独使用不如有监督模型；LLM 提取的特征叠加在有监督模型上有增益，但增益主要出现在人工转录上。
- **Delta-KNN**（Li 等，ACL 2025，PR）。https://arxiv.org/html/2506.03476 。Llama-3.1-8B 做上下文学习，用 Delta-KNN 挑选示例。ADReSS-test Acc 83.6；同一张表中 SVM 79.9、BERT 79.3±3.2、GPT-3 嵌入 + SVM 80.3、微调 Llama 77.1、零样本 57.6。在 ADReSS-train 上 SVM（80.7）和 BERT（81.2）略高于它（80.0）。
- **MMSE 校准少样本提示**（Sweidan 等，arXiv 2509.19926，PP）。https://arxiv.org/html/2509.19926v1 。Mistral-7B 在 ADReSS-test 上：零样本 Acc 0.65，k=14 时 Acc 0.82、AUC 0.86。论文没有与有监督 SOTA 对比。
- **Balamurali & Chen**（arXiv 2402.01751，PP；Diagnostics 2024 的期刊版本因 403 未能打开，标"未核实"）。https://arxiv.org/abs/2402.01751 。GPT-4 零样本：识别 CN 的 F1 62%；Bard：识别 AD 的召回率 89%、F1 71%，但会把 CN 误判为 AD。
- **Chen & Li**（EMBC 2024，PR，依据 arXiv 备注）。https://arxiv.org/abs/2409.12541 。用 LLM 推理归纳患者层面的语言缺陷属性，再输入 ALBERT：在 ADReSS 上相对无推理增强的基线 Acc +8.51%、F1 +8.34%。绝对值标"未核实"。
- **NeuroXVocal**（MICCAI 2025，PR）。https://arxiv.org/abs/2502.10108 。声学 + 文本 + 嵌入经 Transformer 分类，再由 RAG-LLM 结合 AD 文献库生成解释；ADReSSo Acc 95.77%。解释质量是否经临床评估，标"未核实"。

## 3. 可解释 / 知识引导方法，内容特征与纯声学的对比

- **Ng 等**（ICASSP 2025，PR，依据 arXiv 备注）。https://arxiv.org/abs/2502.01685 。自动提取 Cookie Theft 内容信息单元（CIU）并构建时空语义图，效果与人工标注相当，组间差异更大；作者把基于 LLM 的 CIU 提取列为未来工作。
- **Li 等**（ISCSLP 2024，PR）。https://arxiv.org/abs/2411.18922 。用 LLM 视觉能力 + TF-IDF 构造紧凑、可解释的 Cookie Theft 特征，摘要称在两种分类器上都优于传统语言特征；具体数字标"未核实"。
- **Lima 等**（Communications Medicine 2025，PR，依据 arXiv 记录）。https://arxiv.org/abs/2501.18731 。DementiaBank 291 人，RF + 语言特征：敏感度 69.4%，特异度 83.3%；在 22 人外部试点上特异度降到 52.5%。本次读到的摘要页中没有语言特征与声学特征的直接对比。
- 内容特征强于纯声学的证据：SpeechCARE 的消融中纯文本（68.88）略高于纯声学（68.23）；Yuan 2020 的最优系统是加入停顿编码的文本模型。反方向的证据是 PREPARE 第 1 名以 Whisper 声学编码器为主。本次检索没有找到针对 AD 语音的 concept-bottleneck 论文，这一方向在文献中属于空白，不能当作已有证据引用。

## 4. 顶会/顶刊的医疗智能体框架

- **MDAgents**（NeurIPS 2024，PR）。https://proceedings.neurips.cc/paper_files/paper/2024/hash/90d1fc07f46e31387978b88e7e057a31-Abstract.html 。按问题复杂度自适应决定单个或多个智能体协作，10 个基准中 7 个最优，最多比此前最佳高 4.2%；加入主持人审核和外部知识后平均 +11.8%。所有基准均为医学问答或多模态推理，没有与有监督分类器对比。
- **Ferber 等**（Nature Cancer 2025，PR，DOI 10.1038/s43018-025-00991-6）。GPT-4 加上病理/影像模型、OncoKB/PubMed 检索和约 6,800 份指南后，决策准确率从单用 GPT-4 的 30.3% 升到 87.2%；只有 20 个模拟病例。
- **Hager 等**（Nature Medicine 2024，PR，DOI 10.1038/s41591-024-03097-1）。2,400 例 MIMIC 腹痛病例中，LLM 诊断显著差于医生，指令遵循差，对信息的顺序和数量敏感。
- **Brown 等**（JAMIA 2025，PR，DOI 10.1093/jamia/ocaf038）。临床预测任务 AUROC：GBM 0.847/0.894，GPT-4 0.629/0.602，GPT-3.5 0.537/0.517；Brier 分数同样是 GBM 更优。
（上面三篇的 Europe PMC 检索页：https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=DOI:10.1038/s43018-025-00991-6%20OR%20DOI:10.1038/s41591-024-03097-1%20OR%20DOI:10.1093/jamia/ocaf038 ）

## 5. 对本设计的启示

- 在已核实的对比中，LLM 直接分类均输给或仅追平有监督基线。零样本 GPT-4 低于 RF（0.827 vs 0.885），JAMIA 研究中 GPT-4 远低于 GBM；在 ADReSS 上只有精心挑选示例的 ICL 才与 SVM/BERT 持平。因此智能体应以有监督双头分类器的输出为输入，而不是取代它。
- 智能体的增益来自工具携带的信息（Ferber：30.3%→87.2%）。我们的"状态 z 分数 + 分类器概率"就相当于这里的工具。这一类比来自肿瘤学的 20 个病例，在 AD 语音任务上没有直接证据。
- LLM 提取的特征只在人工转录上有明显增益（+0.046 AUROC），在 ASR 转录上只有 +0.007 到 +0.012。PREPARE 必须走 ASR，而且是多语种，因此 LLM 提取的内容单元预期增益很小，必须用消融实验证明。
- SpeechCARE 的消融显示，换融合方式只差 0.4 F1，加入年龄却 +1.6 F1。我们的健康参照 z 分数如果按年龄/教育/语言分层，比改融合结构更可能带来提升。
- 我们的编码器是冻结的，而 PREPARE 第 2 名明确微调了 Whisper，SpeechCARE 正文提到对 mGTE/mHuBERT 做了微调（训练细节未读完）。目前没有证据表明冻结编码器 + 手工指标能追平微调模型，正面对标有较大风险。
- PREPARE 按 log-loss 排名，校准融合直接对应这个指标。第 1 名（0.6299）和第 2 名（0.6343）都低于 SpeechCARE（0.6460–0.6553），在同一协议下，用 log-loss 比较比用 F1 更公平。
- MCI 是瓶颈：SpeechCARE 的 MCI 召回率只有 43–55%。第二个头（MCI vs AD）应单独报告 MCI 召回率，并做阈值优化。

同一协议（PREPARE 412 测试集）下可能超过 SpeechCARE 的方法，均未在 F1/AUC 上核实：(1) 第 1 名的 Whisper 编码器 + 临床特征 + 两阶段去语言偏差训练；(2) 第 2 名的多路 Whisper 集成 + 对人口学不变的对比微调；(3) 沿用 SpeechCARE 的结构，把声学编码器换成微调后的 Whisper，并加入按年龄分层的规范化特征（推断）；(4) 有监督概率 + LLM 特征 + 校准的堆叠模型（推断；依据是 Heitz 在人工转录上的增益，在 ASR 上未核实）。
