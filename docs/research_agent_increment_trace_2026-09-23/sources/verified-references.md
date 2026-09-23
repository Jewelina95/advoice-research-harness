# Verified references

## Standards and reporting

- W3C. (2013). *PROV-Overview*. https://www.w3.org/TR/prov-overview/
- W3C. (2013). *PROV Model Primer*. https://www.w3.org/TR/prov-primer/
- Collins, G. S., et al. (2024). TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. *BMJ, 385*, e078378. https://doi.org/10.1136/bmj-2023-078378
- Lekadir, K., et al. (2025). FUTURE-AI: international consensus guideline for trustworthy and deployable artificial intelligence in healthcare. *BMJ, 388*, e081554. https://doi.org/10.1136/bmj-2024-081554

## Explanation faithfulness and intervenable concepts

- Lanham, T., et al. (2023). Measuring faithfulness in chain-of-thought reasoning. *arXiv*. https://arxiv.org/abs/2307.13702
- Lyu, Q., et al. (2024). Towards faithful model explanation in NLP: A survey. *Computational Linguistics, 50*(2), 657-723. https://aclanthology.org/2024.cl-2.6/
- Koh, P. W., et al. (2020). Concept bottleneck models. *Proceedings of ICML 2020*, 5338-5348. https://proceedings.mlr.press/v119/koh20a.html
- Pesapane, F., et al. (2026). Evidence over explanations: put medical AI to the test. *npj Artificial Intelligence, 2*, 53. https://doi.org/10.1038/s44387-026-00092-4

## Clinical provenance and argumentation

- Xiao, L., Zhou, H., & Fox, J. (2022). Towards a systematic approach for argumentation, recommendation, and explanation in clinical decision support. *Mathematical Biosciences and Engineering, 19*(10), 10445-10473. https://doi.org/10.3934/mbe.2022489
- Sassoon, I., Kokciyan, N., Modgil, S., & Parsons, S. (2021). Argumentation schemes for clinical decision support. *Argument & Computation, 12*(3). https://doi.org/10.3233/AAC-200550
- Curcin, V., et al. (2020). Non-repudiable provenance for clinical decision support systems. *arXiv*. https://arxiv.org/abs/2006.11233

## Clinical agents

- Schmidgall, S., et al. (2026). AgentClinic: a multimodal benchmark for tool-using clinical AI agents. *npj Digital Medicine, 9*, 499. https://doi.org/10.1038/s41746-026-02674-7
- Goh, E., et al. (2024). Large language model influence on diagnostic reasoning: A randomized clinical trial. *JAMA Network Open, 7*(10), e2440969. https://pubmed.ncbi.nlm.nih.gov/39466245/
- MedAgentBench. (2025). Benchmarking language agents in electronic health records. *NEJM AI*. https://doi.org/10.1056/AIdbp2500144
- Moulaei, K., et al. (2026). The role of agentic AI in healthcare: a scoping review. *npj Digital Medicine*. https://doi.org/10.1038/s41746-026-02517-5

## Speech-based cognitive screening and explainability

- Guan, Y., et al. (2025). SpeechCARE: dynamic multimodal modeling for cognitive screening in diverse linguistic and speech task contexts. *npj Digital Medicine, 8*. https://doi.org/10.1038/s41746-025-02026-x
- Shankar, R., et al. (2025). A systematic review of explainable artificial intelligence methods for speech-based cognitive decline detection. *npj Digital Medicine, 8*, 724. https://doi.org/10.1038/s41746-025-02105-z
- Chandler, C., et al. (2023). An explainable machine learning model of cognitive decline derived from speech. *Alzheimer's & Dementia: Diagnosis, Assessment & Disease Monitoring, 15*, e12516. https://pmc.ncbi.nlm.nih.gov/articles/PMC10752754/
- Haghbin, M., et al. (2026). From black-box to clinical insight: explainable multimodal cognitive screening. *arXiv preprint*. https://arxiv.org/abs/2606.27973

## Evidence notes

- W3C PROV supports entity-activity-agent provenance and responsibility. It does not prove that a medical inference is correct.
- Concept bottleneck models support explicit interventions on intermediate concepts and propagation to the target. They require valid, aligned concepts.
- Clinical argumentation systems represent support, opposition, rules, and proof graphs. Their logic is usually rule-driven rather than learned from speech.
- Clinical Agent benchmarks test interaction, planning, tool use, and sequential information gathering. They do not by themselves prove faithful diagnostic evidence use.
- Speech cognitive-screening XAI is dominated by SHAP, LIME, attention, and feature importance. The 2025 systematic review reports limited stability testing and limited real-world clinical validation.
- Free-form chain-of-thought is not a reliable record of the causal computation that produced an answer. Faithfulness requires intervention or other causal tests.
