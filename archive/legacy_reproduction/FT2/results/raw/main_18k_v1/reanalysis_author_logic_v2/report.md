# FT2 author-logic re-scoring

Source runs: 18000; source controls: 150.
All completed fault-injection runs are included; protected-clean correctness is not an evaluability gate.
Primary SDC is binary: normalized reference-token recall must equal 1.0.

## Overall by mode

| Mode | Runs | Masked identical | Masked semantic | SDC | SDC rate | Literal repo 1-mean-recall |
|---|---:|---:|---:|---:|---:|---:|
| no_protection | 6000 | 5266 | 428 | 306 | 5.1000% | 4.6931% |
| paper_clamp_first_token_bounds | 6000 | 5577 | 399 | 24 | 0.4000% | 0.3333% |
| paper_clamp_offline_bounds | 6000 | 5642 | 327 | 31 | 0.5167% | 0.4222% |

## Previously excluded runs

Original NON_EVALUABLE_CLEAN_FAILURE runs: 240.
- MASKED_IDENTICAL: 228
- MASKED_SEMANTIC: 12

## Qwen2-Math-7B / GSM8K by mode

| Mode | Runs | SDC | SDC rate | Author-scored clean correct |
|---|---:|---:|---:|---:|
| no_protection | 1200 | 41 | 3.4167% | 10/10 |
| paper_clamp_first_token_bounds | 1200 | 1 | 0.0833% | 10/10 |
| paper_clamp_offline_bounds | 1200 | 1 | 0.0833% | 10/10 |

## Reproducibility notes

- Upstream commit: `90510aec5d26850739cf583c97c65cbd71256cb1`
- Source summary SHA-256: `83b488d678e39c60791ec41ea9cbc524f271861fe5ae29d3c5aeb83f84bf3d2f`
- Analysis script SHA-256: `3c6f9b3dbf00a36c20117dd98599b35fe8f85c0a0ebb4d48a587723a3e8e59b9`
- Stored outputs are scored in full, without EOS truncation, matching the upstream fixed-step loops.
- The author's literal mean-recall result is retained as a secondary metric; the primary binary result implements the paper's Masked/SDC intent.
