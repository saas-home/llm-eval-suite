# Benchmark data sources

Seeded samples (seed 20261106) of official public benchmarks, fetched by `scripts/fetch_benchmark_data.py`, plus original tasks written for this repository. Each official item keeps its original answer key and is redistributed under the license listed; attribution belongs to the original dataset authors. GPQA is deliberately not included (its authors ask that items not be published).

| Suite | File | Items | License | Source | sha256 (16) |
|---|---|---|---|---|---|
| coding | multipl_e_csharp.json | 100 | MIT | https://huggingface.co/datasets/nuprl/MultiPL-E (humaneval-cs; HumanEval MIT) | f45da1ad31464c9e |
| coding | multipl_e_go.json | 100 | MIT | https://huggingface.co/datasets/nuprl/MultiPL-E (humaneval-go; HumanEval MIT) | b6401feebc73fb69 |
| coding | multipl_e_java.json | 100 | MIT | https://huggingface.co/datasets/nuprl/MultiPL-E (humaneval-java; HumanEval MIT) | d66cea654aa545f6 |
| coding | multipl_e_python.json | 100 | MIT | https://huggingface.co/datasets/openai/openai_humaneval (test) | fc7e8c822f0f5b0e |
| coding | multipl_e_rust.json | 100 | MIT | https://huggingface.co/datasets/nuprl/MultiPL-E (humaneval-rs; HumanEval MIT) | 06bf523f9ca865d7 |
| coding | multipl_e_typescript.json | 100 | MIT | https://huggingface.co/datasets/nuprl/MultiPL-E (humaneval-ts; HumanEval MIT) | ed6d7c7545773758 |
| judge_validation | llmbar.json | 40 | MIT | https://github.com/princeton-nlp/LLMBar (Natural, Adversarial/Neighbor) | a957cdaa581d997c |
| judge_validation | summeval.json | 40 | MIT | https://huggingface.co/datasets/mteb/summeval (test) | bbebe84a19058ba2 |
| legal | cuad_contracts.json | 45 | CC-BY-4.0 | https://huggingface.co/datasets/theatticusproject/cuad (CUAD v1) | afa3b5bf2bff5e34 |
| legal | cuad_review.json | 30 | CC-BY-4.0 | https://huggingface.co/datasets/theatticusproject/cuad (CUAD v1) | aa81e3b1e9f56ed0 |
| legal | legalbench.json | 100 | CC-BY-4.0 | https://huggingface.co/datasets/nguha/legalbench (CC-BY-4.0 tasks only) | 9528fe4d74222c35 |
| medical | medmcqa.json | 100 | Apache-2.0 | https://huggingface.co/datasets/openlifescienceai/medmcqa (validation; test has no keys) | ce112e2eb27ccca2 |
| medical | medqa_usmle.json | 100 | CC-BY-4.0 | https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options (test) | 4980e1e50badbd6d |
| medical | mmlu_medical.json | 100 | MIT | https://huggingface.co/datasets/cais/mmlu (6 medical subjects, test) | 4d41353a07856923 |
| medical | pubmedqa.json | 100 | MIT | https://huggingface.co/datasets/qiaojin/PubMedQA (pqa_labeled) | f66bd532d0eb19c2 |
| personal | personal_tasks.json | 20 | same as repository | original tasks written for this repository | 9ab68489ef80584d |
| personal | simpleqa.json | 100 | MIT | https://github.com/openai/simple-evals (SimpleQA test set) | e3a536e939c6ff44 |
| personal | xstest.json | 100 | CC-BY-4.0 | https://huggingface.co/datasets/natolambert/xstest-v2-copy (XSTest v2) | bb3e13f34749c976 |
| reasoning | bbh.json | 100 | MIT | https://github.com/suzgunmirac/BIG-Bench-Hard (12 tasks) | cc6907cc1bc6d530 |
| reasoning | math500.json | 100 | MIT | https://huggingface.co/datasets/HuggingFaceH4/MATH-500 (from Hendrycks MATH / PRM800K) | 38c76699e38d81b1 |
| reasoning | mmlu_pro.json | 100 | MIT | https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro (test) | 5a479254a9c71dda |
| writing | billsum.json | 30 | CC0-1.0 | https://huggingface.co/datasets/FiscalNote/billsum (test) | f295ad7e55ad94ea |
| writing | ifeval_full.json | 541 | Apache-2.0 | https://huggingface.co/datasets/google/IFEval (all prompts) | c562a202fd259d0f |
| writing | writing_tasks.json | 20 | same as repository | original tasks written for this repository | 058ea73c1c9074f9 |
