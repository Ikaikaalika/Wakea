# Wakea — Multimodal Small Reasoning Model (MM-SRM)

Wakea is a small (≈1–3B target) multimodal reasoning model skeleton with:
- Text LM backbone hooks (with RoPE), multimodal adapters (vision/audio)
- Tool use via a simple tool router (calculator, RAG, code-exec stub)
- RAG stack stubs (embed, index, retrieve, rerank)
- Training script scaffolds (SFT, DPO, PPO/GRPO) with an optional bridge to Raschka's reasoning-from-scratch
- Serving stub with a minimal CLI loop

This repo is a scaffold to accelerate research and prototyping.

## Quickstart

1) Explore the demo router and tools
```
python -m wakea.serving.server --help
python -m wakea.serving.server --prompt "What is 23*19? Use tool:calculator"
```

2) Try RAG (in-memory toy index)
```
python -m wakea.scripts.build_index --docs wakea/examples/docs.txt
python -m wakea.serving.server --prompt "RAG: Tell me about Wakea."
```

3) Training (stubs; replace logic as needed)
```
python -m wakea.scripts.train_sft --config wakea/configs/train_sft.yaml
```

See configs under `wakea/configs/` and scripts under `wakea/scripts/`.
