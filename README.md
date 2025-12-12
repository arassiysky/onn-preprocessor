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



