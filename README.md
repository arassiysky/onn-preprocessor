\# ONN Pre-Processor (v0.L)



Operator-defined embedding augmentation for LLMs.



This repository implements \*\*Option 2 (augment)\*\* of the ONN pre-processing pipeline:

deterministic, operator-derived features are concatenated to token embeddings and projected

back to the model dimension \*\*before attention\*\*.



\## What this is

\- A standalone ONN pre-processor

\- Model-agnostic (Bring Your Own Llama)

\- No finetuning

\- No token dropping (yet)



\## Roadmap

\- v0.L: embedding augmentation

\- v0.M: optional token compression

\- v0.N: global structure / multi-view conditioning


1 benchmark no generation

seq     | base_avg     | onn_avg      | overhead   | logits_delta   | onn_determin
------- | ------------ | ------------ | ---------- | -------------- | --------------
------- | ------------ | ------------ | ---------- | -------------- | --------------
8       | 291.74ms     | 296.40ms     | +1.6%      | 0.021868       | 0
8       | 291.74ms     | 296.40ms     | +1.6%      | 0.021868       | 0
32      | 362.20ms     | 364.25ms     | +0.6%      | 0.016455       | 0
32      | 362.20ms     | 364.25ms     | +0.6%      | 0.016455       | 0
128     | 699.06ms     | 697.36ms     | -0.2%      | 0.020385       | 0
128     | 699.06ms     | 697.36ms     | -0.2%      | 0.020385       | 0
512     | 2267.09ms    | 2268.68ms    | +0.1%      | 0.010338       | 0
512     | 2267.09ms    | 2268.68ms    | +0.1%      | 0.010338       | 0


v0.L defaults

projector_kind = "residual_det"
epsilon = 0.001
feature_dim = 8
deterministic seed = 42