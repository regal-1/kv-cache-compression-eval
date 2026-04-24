# kv-cache-compression-eval
 
Implementation and evaluation of two KV cache compression methods — RKV and ChunkKV — for long-context LLM inference. Part of a broader study benchmarking eight compression strategies across LongBench tasks; full results and analysis are in the [report PDF](DSC291___Efficient_AI_Report.pdf).
 
## Overview
 
As LLMs increasingly operate on long contexts (100K+ tokens), the KV attention cache becomes a dominant GPU memory bottleneck. This project evaluates whether RKV and ChunkKV can reduce that footprint without destroying task performance — and measures the real latency cost of doing so.
 
**Methods implemented:** RKV, ChunkKV (+ full-cache baseline)
 
**Cache budgets:** 100% (baseline), 50%, 20%, 10%
 
**Tasks (LongBench):** NarrativeQA, Qasper, MultiFieldQA, HotpotQA, 2WikiMQA, GovReport
 
**Model:** Qwen2.5-1.5B-Instruct
 
## Key Findings
 
- **ChunkKV dominates at aggressive compression.** At 10% cache retention it achieves nearly 2× the macro-average accuracy of RKV by preserving contiguous semantic spans.
- **Task type matters more than method choice.** Summarization (GovReport) retains 77–86% of baseline performance even at 10% budget. QA tasks rarely exceed 40% retention even at 50% budget.
- **Compression does not reduce latency** in pipeline-level implementations. Both methods add overhead relative to the full-cache baseline.
## Notebook
 
Full implementation, eviction logic, evaluation pipeline, and results are in [`rkv_chunkv_eval.ipynb`](rkv_chunkv_eval.ipynb).
 
## Setup
 
```bash
pip install transformers datasets torch accelerate kvpress rouge
```
 
Experiments were run on a GPU with `Qwen/Qwen2.5-1.5B-Instruct` loaded in `bfloat16`.
