# Wakea — Multimodal Small Reasoning Model (MM-SRM)

Wakea is a small (≈1–3B target) multimodal reasoning model scaffold designed for research:
- Text LM backbone (RoPE) with clean, typed modules and generation API
- Multimodal adapters (vision/audio) ready for frozen encoders (CLIP/Whisper)
- Tool use via a typed router (calculator, RAG, code-exec, web stub)
- RAG stack with in-memory index plus persistence and API
- Research-grade training flows for SFT and DPO (PyTorch), PPO hooks
- Config-driven runs (YAML), logging, seeding, and checkpoint I/O

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

3) Train (SFT) and load for generation
```
python -m wakea.scripts.train_sft --config wakea/configs/train_sft.yaml
python -m wakea.serving.server --ckpt checkpoints/sft/model.pt --model-cfg wakea/configs/model.yaml --prompt "Hello Wakea"
```

4) Build or load RAG index
```
python -m wakea.scripts.build_index --docs wakea/examples/docs.txt --out checkpoints/rag.pkl
python -m wakea.scripts.build_index --load checkpoints/rag.pkl
```

See configs under `wakea/configs/` and scripts under `wakea/scripts/`.

## Training Notes
- SFT: JSONL `wakea/data/schemas/sft_dialogue.jsonl` demonstrates the expected chat format.
- DPO: JSONL `wakea/data/schemas/pref_pairs.jsonl` demonstrates prompt/choice pairs.
- Replace the SimpleTokenizer with a Hugging Face tokenizer by setting `data.tokenizer` in the config (e.g., `gpt2`) and installing the `tokenizers` extra.
- Checkpoints are saved with `torch.save` and can be loaded by the server CLI for quick generation tests.

## Design
- Modular modeling: `WakeaForCausalLM` wraps a compact Transformer with LMHead and RoPE.
- Multimodal fusion: adapters project encoder features to LM hidden and concatenate fused tokens.
- Tool routing: typed interface with calculator/RAG/code; extend with your tools via `serving/router.py`.
- RAG: toy embeddings (hash-based) + in-memory index with persistence; swap in your embedding model and index backend.

## Caveats
- This is a research scaffold. Replace toy components (tokenizer, embeddings, code sandbox) for real experiments.
- PPO/GRPO training is stubbed behind `training/rfs_bridge.py`; integrate TRL or the reasoning-from-scratch adapter.
