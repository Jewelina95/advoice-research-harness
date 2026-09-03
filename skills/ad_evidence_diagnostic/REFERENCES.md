# Reference basis

This initial Skill separates well-supported clinical boundaries from project hypotheses. Metric-specific thresholds are not considered validated until the 8.27 evidence table links each metric to a primary source and a training-fold reference protocol.

1. Jack CR Jr, et al. Revised criteria for diagnosis and staging of Alzheimer's disease: Alzheimer's Association Workgroup. *Alzheimer's & Dementia* (2024). https://alz-journals.onlinelibrary.wiley.com/doi/10.1002/alz.13859
2. Zhang X, et al. Cognition-of-Thought Elicits Social-Aligned Reasoning in Large Language Models (2025 preprint). https://arxiv.org/abs/2509.23441
3. Li D, et al. Streamlining evidence based clinical recommendations with large language models. *npj Digital Medicine* (2025). https://www.nature.com/articles/s41746-025-02273-y
4. SpeechCARE: dynamic multimodal modeling for cognitive screening in diverse linguistic and speech task contexts. https://pmc.ncbi.nlm.nih.gov/articles/PMC12623413/

## Metric registry reference keys

The initial registry also carries reference keys retained from the project's July literature audit: `R_VOLETI2019`, `R_GAUDER2024`, `R_TRACEY2023`, `R_BITTNER2022`, `R_MCCARTHY2010`, `R_BURKE2023`, `R_SNOWDON1996`, `R_LYONS1994`, `R_LIAN2025`, `R_YUAN2021`, `R_HENRY2004`, and `R_CLARKE2021`. These keys record the rationale trail but are not yet publication-ready citations. Before training, each must be resolved to a verified primary-paper record with DOI/URL, population, task, language, effect direction and limitation. `R_PROJECT` explicitly marks a project hypothesis rather than established medical evidence.

## Interpretation status

- Reference 1 defines the diagnostic boundary: speech screening does not establish biological AD.
- Reference 2 motivates explicit inference-time monitoring and rollback; the project adapts the idea rather than claiming an exact reproduction.
- Reference 3 supports standardized staged workflows, evidence assessment and inspectable intermediate outputs.
- Reference 4 is the principal comparison for dynamic multimodal cognitive screening; protocol-matched comparison is required.
