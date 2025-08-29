# Wakea — Multimodal Small Reasoning Model (MM-SRM)

Wakea is a small (≈1–3B target) multimodal reasoning model scaffold designed for research:
- Text LM backbone (RoPE) with clean, typed modules and generation API
- Multimodal adapters (vision/audio) ready for frozen encoders (CLIP/Whisper)
- Tool use via a typed router (calculator, RAG, code-exec, web stub)
- RAG stack with FAISS (cosine) or in-memory index, sentence-transformers embeddings, persistence, and API
- Research-grade training flows for SFT and DPO (PyTorch), PPO hooks
- Config-driven runs (YAML), logging, seeding, and checkpoint I/O

This repo is a scaffold to accelerate research and prototyping.

## Quickstart

1) Explore the demo router and tools
```
python -m serving.server --help
python -m serving.server --prompt "What is 23*19? Use tool:calculator"
```

2) Try RAG (FAISS + sentence-transformers if installed)
```
python -m scripts.build_index --docs examples/docs.txt --config configs/rag.yaml
python -m serving.server --prompt "RAG: Tell me about Wakea."
```

3) Train (SFT) and load for generation
```
python -m scripts.train_sft --config configs/train_sft.yaml
python -m serving.server --ckpt checkpoints/sft/model.pt --model-cfg configs/model.yaml --prompt "Hello Wakea"
```

4) Build or load RAG index
```
python -m scripts.build_index --docs examples/docs.txt --out checkpoints/rag.idx --config configs/rag.yaml
python -m scripts.build_index --load checkpoints/rag.idx --config configs/rag.yaml
```

5) Train tool-use head (web, rag, calculator, code)
```
python -m scripts.train_sft --config configs/train_sft.yaml
# Config enables joint LM + tool classification via tool_data + tool_loss_weight
```

See configs under `configs/` and scripts under `scripts/`.

Install extras for production features:
```
pip install -e .[rag,tokenizers,server]
```

Configure web search provider (SerpAPI example):
```
export SERPAPI_API_KEY="<your-key>"
# configs/tools.yaml -> web_config: { provider: serpapi, api_key_env: SERPAPI_API_KEY }
```

## Training Notes
- SFT: JSONL `data/schemas/sft_dialogue.jsonl` demonstrates the expected chat format.
- DPO: JSONL `data/schemas/pref_pairs.jsonl` demonstrates prompt/choice pairs.
- Replace the SimpleTokenizer with a Hugging Face tokenizer by setting `data.tokenizer` in the config (e.g., `gpt2`) and installing the `tokenizers` extra.
- Checkpoints are saved with `torch.save` and can be loaded by the server CLI for quick generation tests.

Dataset building for web tool-use
```
python -m scripts.build_webqa --queries queries.txt --out data/webqa.jsonl --tools-cfg configs/tools.yaml --top-k 5
```
This creates a tool-use SFT file with context for the `web` tool.

## Design
- Modular modeling: `WakeaForCausalLM` wraps a compact Transformer with LMHead and RoPE.
- Multimodal fusion: adapters project encoder features to LM hidden and concatenate fused tokens.
- Tool routing: typed interface with calculator/RAG/code; extend with your tools via `serving/router.py`.
- RAG: toy embeddings (hash-based) + in-memory index with persistence; swap in your embedding model and index backend.

## Caveats
- This is a research scaffold. Replace toy components (tokenizer, embeddings, code sandbox) for real experiments.
- PPO/GRPO training is stubbed behind `training/rfs_bridge.py`; integrate TRL or the reasoning-from-scratch adapter.
