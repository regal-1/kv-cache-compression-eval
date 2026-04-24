# kv-cache-compression-eval
 
A systematic benchmarking study of KV cache compression strategies for long-context LLM inference. We evaluate eight state-of-the-art methods across six LongBench tasks under varying cache budgets, using a unified pipeline built on [kvpress](https://github.com/NVIDIA/kvpress) and Qwen2.5-1.5B-Instruct.
 
## Overview
 
As LLMs increasingly operate on long contexts (100K+ tokens), the KV attention cache becomes a dominant GPU memory bottleneck. This project evaluates whether popular compression strategies can reduce that footprint without destroying task performance — and measures the real latency cost of doing so.
 
**Methods evaluated:** H2O, StreamingLLM, SnapKV, AdaKV, RKV, ChunkKV, KNorm, ScissorHands
 
**Cache budgets:** 100% (baseline), 50%, 20%, 10%
 
**Tasks (LongBench):** NarrativeQA, Qasper, MultiFieldQA, HotpotQA, 2WikiMQA, GovReport
 
## Key Findings
 
- **ChunkKV dominates at aggressive compression.** At 10% cache retention it achieves nearly 2× the macro-average accuracy of the next-best method by preserving contiguous semantic spans.
- **Task type matters more than method choice.** Summarization (GovReport) retains 77–86% of baseline performance even at 10% budget. QA tasks rarely exceed 40% retention even at 50% budget.
- **Compression does not reduce latency** in pipeline-level implementations. All methods are slower than the uncompressed baseline (1.27s/sample). RKV is the slowest at 3.3–3.6s; AdaKV and KNorm are the most efficient compressed methods at 1.6–2.0s.
- **No method dominates across all tasks.** ScissorHands leads on 2WikiMQA, SnapKV on GovReport, ChunkKV on Qasper.

## Notebook
 
Full implementation, eviction logic, evaluation pipeline, and results are in [`rkv_chunkv_eval.ipynb`](rkv_chunkv_eval.ipynb).
 
## Setup
 
```bash
pip install transformers datasets torch accelerate kvpress rouge
```
 
Experiments were run on a GPU with `Qwen/Qwen2.5-1.5B-Instruct` loaded in `bfloat16`.
 
