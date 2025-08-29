# MM‑SRM: Multimodal Small Reasoning Model
**Agentic RL + RAG, designed to run on M‑series Macs and common GPUs**  
Version: 1.1 (full, self‑contained build doc)

> This Markdown file is the *single source of truth* to stand up a working research stack:
> repo scaffold, configs, minimal runnable code, synthetic data generators, training scripts
> (SFT → DPO → Agentic RL), a RAG microservice, a tool router, and a FastAPI chat server.
> You can copy/paste files from the blocks below into your repo exactly as named.

---

## 0) TL;DR One‑Command Quickstart (after file creation)
```bash
# 0) Create a fresh venv (Mac or Linux)
python -m venv .venv && source .venv/bin/activate

# 1) Install
pip install --upgrade pip wheel
pip install -r requirements.txt

# 2) Build a tiny index and synthetic datasets
python scripts/build_index.py --docs ./corpus
python scripts/make_synthetic_data.py --out data/

# 3) Supervised finetune (toy scale)
python scripts/train_sft.py -c configs/train_sft.yaml

# 4) Preference optimization (toy DPO)
python scripts/train_dpo.py -c configs/train_dpo.yaml

# 5) Agentic RL on mini envs (calc/webshop-lite/rag-qa-lite)
python scripts/train_rl.py -c configs/train_rl.yaml

# 6) Run RAG API + Chat server
uvicorn rag.api:app --port 8001 --reload &
uvicorn serving.server:app --port 8000 --reload

# 7) Chat with tools
curl -s localhost:8000/chat -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"What is (12*7)+5? Cite a source."}]}' | jq
```

> **Hardware notes:** For realism on a laptop, all training scripts default to **LoRA/QLoRA** and tiny batch sizes.
> Swap in a real 1–2B checkpoint if you have VRAM; otherwise the included *TinyLM* keeps everything runnable.

---

## 1) Repository Layout
Copy these folders & files exactly. Each section below contains the file contents.

```
mm-srm/
  README.md                        # you can paste this file as README
  requirements.txt
  pyproject.toml
  .env.example
  configs/
    model.yaml
    multimodal.yaml
    rag.yaml
    tools.yaml
    train_sft.yaml
    train_dpo.yaml
    train_rl.yaml
  modeling/
    tokenizer.py
    transformer.py
    rope.py
    lm_head.py
    tool_head.py
    adapters/vision_mlp.py
    adapters/qformer_lite.py
    adapters/audio_proj.py
  rag/
    ingest.py
    embed.py
    index_faiss.py
    retriever.py
    rerank.py
    api.py
  tools/
    calculator.py
    rag_query.py
    code_exec.py
    web_search.py
  data/
    README_DATA.md
    schemas/sft_dialogue.jsonl      # (generated)
    schemas/pref_pairs.jsonl         # (generated)
    schemas/rl_trajectories.jsonl    # (generated)
  envs/
    webshop_wrapper.py
    ragqa_wrapper.py
    scienceqa_mm_wrapper.py
  serving/
    server.py
    router.py
  evaluation/
    metrics.py
    judge.py
  scripts/
    setup.sh
    build_index.py
    make_synthetic_data.py
    train_sft.py
    train_dpo.py
    train_rl.py
    eval_benchmarks.py
  third_party/
    rfs_bridge.py                  # optional bridge to "reasoning-from-scratch"
  corpus/                          # put a few .txt/.md/.pdf for RAG demo
```

---

## 2) Environment & Packaging

### 2.1 `requirements.txt`
```txt
accelerate>=0.33.0
bitsandbytes>=0.43.3 ; platform_system != "Darwin"
torch>=2.3.0
torchvision>=0.18.0
torchaudio>=2.3.0
transformers>=4.42.0
tokenizers>=0.15.2
datasets>=2.20.0
peft>=0.11.1
trl>=0.9.6
faiss-cpu>=1.8.0.post1
fastapi>=0.111.0
uvicorn>=0.30.0
pydantic>=2.7.1
beautifulsoup4>=4.12.3
tiktoken>=0.7.0
numpy>=1.26.4
scikit-learn>=1.5.1
orjson>=3.10.7
pillow>=10.3.0
python-dotenv>=1.0.1
matplotlib>=3.9.0
outlines>=0.0.46         # JSON-constrained decoding (optional)
Markdown>=3.6
sentence-transformers>=3.0.1
```

### 2.2 `pyproject.toml`
```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "mm-srm"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = []

[tool.setuptools.packages.find]
where = ["."]
```

### 2.3 `.env.example`
```env
# Serving
MM_SRM_MODEL_PATH=TinyLM
MM_SRM_DEVICE=auto

# RAG
RAG_PORT=8001
RAG_EMBEDDER=intfloat/e5-small-v2
RAG_INDEX_PATH=.rag/index.faiss
RAG_STORE_PATH=.rag/store.jsonl
```

---

## 3) Configs

### 3.1 `configs/model.yaml`
```yaml
d_model: 1024
n_layer: 16
n_head: 8
vocab_size: 32032
rope_base: 1000000
context_length: 8192
dropout: 0.0
quantization: int4         # inference
use_lora: true
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
```

### 3.2 `configs/multimodal.yaml`
```yaml
vision:
  encoder: openai/clip-vit-base-patch16
  adapter: resampler
  vis_tokens: 32
  freeze_encoder: true
audio:
  encoder: openai/whisper-tiny
  adapter: linear
  aud_tokens: 16
  freeze_encoder: true
```

### 3.3 `configs/rag.yaml`
```yaml
embedder: intfloat/e5-small-v2
index:
  kind: faiss_hnsw
  dim: 384
  M: 32
  efConstruction: 200
reranker: none
docs:
  chunk_tokens: 800
  overlap: 120
hybrid: true
k_retrieve: 32
k_context: 8
```

### 3.4 `configs/tools.yaml`
```yaml
- name: calculator
  schema: |
    {"type":"object","properties":{"expression":{"type":"string"}},"required":["expression"]}
- name: rag.search
  schema: |
    {"type":"object","properties":{"query":{"type":"string"},"k":{"type":"integer"}},"required":["query"]}
- name: code.exec
  schema: |
    {"type":"object","properties":{"code":{"type":"string"}},"required":["code"]}
- name: web.search
  schema: |
    {"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}
```

### 3.5 Train configs
`configs/train_sft.yaml`
```yaml
model_path: TinyLM            # change to a real HF model id for scale
save_dir: ckpts/sft
max_steps: 1000
batch_size: 4
lr: 2.0e-4
packing: true
mixed_precision: bf16
data_path: data/sft_dialogue.jsonl
```

`configs/train_dpo.yaml`
```yaml
reference_ckpt: ckpts/sft
save_dir: ckpts/dpo
beta: 0.1
batch_size: 4
max_steps: 500
data_path: data/pref_pairs.jsonl
```

`configs/train_rl.yaml`
```yaml
policy_ckpt: ckpts/dpo
save_dir: ckpts/rl
algo: grpo
kl_coef: 0.02
gamma: 0.99
gae_lambda: 0.95
train_steps: 2000
rollout_len: 6
num_envs: 8
env: ragqa-lite
```

---

## 4) Modeling (minimal but runnable)

### 4.1 `modeling/tokenizer.py`
```python
from transformers import AutoTokenizer

SPECIAL_TOKENS = {
    "image": "<image>",
    "audio": "<audio>",
    "tool": "<tool>",
    "obs": "<obs>",
    "mem": "<mem>"
}

def load_tokenizer(model_path="TinyLM"):
    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    new_tokens = [v for v in SPECIAL_TOKENS.values() if v not in tok.get_vocab()]
    if new_tokens:
        tok.add_special_tokens({"additional_special_tokens": new_tokens})
    return tok
```

### 4.2 `modeling/rope.py`
```python
import math, torch

def apply_rotary(x, sin, cos):
    x1, x2 = x[..., ::2], x[..., 1::2]
    x_rot = torch.stack([x1*cos - x2*sin, x1*sin + x2*cos], dim=-1)
    return x_rot.flatten(-2)

def rope_pos(freqs, seq_len, dim):
    pos = torch.arange(seq_len, device=freqs.device)[:, None]
    freqs = freqs[None, :]
    angles = pos * freqs
    return torch.sin(angles), torch.cos(angles)

def rope_freqs(dim, base=1e6, device="cpu"):
    freqs = torch.pow(base, -torch.arange(0, dim, 2, device=device) / dim)
    return freqs
```

### 4.3 `modeling/transformer.py` (TinyLM fallback using HF for reliability)
```python
import torch
from transformers import AutoModelForCausalLM

def load_lm(model_path="TinyLM"):
    # Use HF CausalLM; for tiny runs you can swap to "sshleifer/tiny-gpt2"
    if model_path == "TinyLM":
        model_path = "sshleifer/tiny-gpt2"
    model = AutoModelForCausalLM.from_pretrained(model_path)
    return model
```

### 4.4 `modeling/lm_head.py`
```python
# placeholder (using HF head). In a custom-from-scratch path, put your LM head here.
```

### 4.5 `modeling/tool_head.py`
```python
import json, re
from typing import Dict, Any, Optional

# Very simple post-hoc JSON validator; add "outlines" constrained decoding in serving.
VALID_JSON = re.compile(r"\{.*\}", re.S)

def parse_tool_call(text: str) -> Optional[Dict[str, Any]]:
    m = VALID_JSON.search(text)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        if "tool" in obj and "arguments" in obj:
            return obj
    except Exception:
        return None
    return None
```

### 4.6 Adapters

`modeling/adapters/vision_mlp.py`
```python
import torch, torch.nn as nn
from transformers import CLIPModel, CLIPProcessor

class VisionMLP(nn.Module):
    def __init__(self, lm_dim=1024, out_tokens=32, clip_model="openai/clip-vit-base-patch16"):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(clip_model)
        for p in self.clip.parameters(): p.requires_grad_(False)
        self.proc = CLIPProcessor.from_pretrained(clip_model)
        self.pool = nn.AdaptiveAvgPool1d(out_tokens)
        self.proj = nn.Sequential(nn.Linear(self.clip.vision_model.config.hidden_size, lm_dim),
                                  nn.GELU(), nn.Linear(lm_dim, lm_dim))
    def forward(self, images):  # PIL images list
        inputs = self.proc(images=images, return_tensors="pt")
        feats = self.clip.vision_model(**inputs).last_hidden_state  # [B, P, C]
        x = feats.transpose(1,2)                # [B, C, P]
        x = self.pool(x).transpose(1,2)         # [B, T, C]
        return self.proj(x)                     # [B, T, lm_dim]
```

`modeling/adapters/qformer_lite.py`
```python
# Minimal stub; for production use BLIP-2 style Q-Former
import torch, torch.nn as nn

class QFormerLite(nn.Module):
    def __init__(self, vis_dim=768, lm_dim=1024, n_tokens=32, n_layer=2, n_head=8):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, n_tokens, vis_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=vis_dim, nhead=n_head, batch_first=True)
        self.tf = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)
        self.proj = nn.Linear(vis_dim, lm_dim)
    def forward(self, vis_feats):
        B = vis_feats.size(0)
        q = self.query.expand(B, -1, -1)
        x = torch.cat([q, vis_feats], dim=1)
        h = self.tf(x)[:, :q.size(1)]
        return self.proj(h)
```

`modeling/adapters/audio_proj.py`
```python
# Optional in v1.1; keep a simple linear projector
import torch.nn as nn

class AudioProjector(nn.Module):
    def __init__(self, aud_dim=384, lm_dim=1024, out_tokens=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(out_tokens)
        self.proj = nn.Sequential(nn.Linear(aud_dim, lm_dim), nn.GELU(), nn.Linear(lm_dim, lm_dim))
    def forward(self, aud_feats):  # [B, T, aud_dim]
        x = self.pool(aud_feats.transpose(1,2)).transpose(1,2)
        return self.proj(x)
```

---

## 5) RAG Stack

### 5.1 `rag/embed.py`
```python
from sentence_transformers import SentenceTransformer
import numpy as np

class Embedder:
    def __init__(self, model_name="intfloat/e5-small-v2"):
        self.model = SentenceTransformer(model_name)
    def encode(self, texts):
        return np.asarray(self.model.encode(texts, normalize_embeddings=True))
```

### 5.2 `rag/index_faiss.py`
```python
import faiss, numpy as np

class FaissHNSW:
    def __init__(self, dim=384, M=32):
        self.index = faiss.IndexHNSWFlat(dim, M)
        self.ids = []
    def add(self, vecs, ids):
        self.index.add(np.ascontiguousarray(vecs.astype('float32')))
        self.ids.extend(ids)
    def search(self, q, k=32):
        D, I = self.index.search(np.ascontiguousarray(q.astype('float32')), k)
        return D, [[self.ids[i] for i in idxs] for idxs in I]
```

### 5.3 `rag/ingest.py`
```python
import json, re, pathlib
from bs4 import BeautifulSoup

def simple_text(content: str) -> str:
    soup = BeautifulSoup(content, "html.parser")
    text = soup.get_text(" ", strip=True)
    return re.sub(r"\s+", " ", text)

def chunk(text, size=800, overlap=120):
    words = text.split()
    out, i = [], 0
    while i < len(words):
        out.append(" ".join(words[i:i+size]))
        i += size - overlap
    return out

def walk_corpus(root="corpus"):
    root = pathlib.Path(root)
    for f in root.glob("**/*"):
        if f.is_file():
            try: yield str(f), f.read_text(errors="ignore")
            except Exception: pass
```

### 5.4 `rag/retriever.py`
```python
import json, os
from .embed import Embedder
from .index_faiss import FaissHNSW
from .ingest import walk_corpus, chunk, simple_text

class Retriever:
    def __init__(self, store_path=".rag/store.jsonl", index_path=".rag/index.faiss",
                 embedder="intfloat/e5-small-v2"):
        os.makedirs(".rag", exist_ok=True)
        self.store_path = store_path
        self.embedder = Embedder(embedder)
        self.index = FaissHNSW(dim=384, M=32)
        self.store = []

    def build(self, corpus_dir="corpus"):
        ids, texts = [], []
        for path, raw in walk_corpus(corpus_dir):
            text = simple_text(raw)
            for i, ch in enumerate(chunk(text)):
                doc_id = f"{path}#_{i}"
                ids.append(doc_id); texts.append(ch)
                self.store.append({"id": doc_id, "text": ch, "source": path})
        X = self.embedder.encode(texts)
        self.index.add(X, ids)
        with open(self.store_path, "w") as f:
            for r in self.store: f.write(json.dumps(r)+"\n")

    def query(self, q, k=8):
        qv = self.embedder.encode([q])
        _, ids = self.index.search(qv, k=k)
        id_set = set(ids[0])
        out = []
        with open(self.store_path) as f:
            for line in f:
                r = json.loads(line)
                if r["id"] in id_set: out.append(r)
        return out
```

### 5.5 `rag/api.py`
```python
from fastapi import FastAPI
from pydantic import BaseModel
from .retriever import Retriever

app = FastAPI(title="MM-SRM RAG API")
RET = Retriever()

class SearchReq(BaseModel):
    query: str
    k: int = 8

@app.on_event("startup")
async def _startup():
    # Assume index already built via scripts/build_index.py
    pass

@app.post("/search")
def search(req: SearchReq):
    docs = RET.query(req.query, k=req.k)
    return {"docs": docs}
```

---

## 6) Tools

### 6.1 `tools/calculator.py`
```python
import math

def run(expression: str):
    allowed = "0123456789+-*/()., "
    if any(c not in allowed for c in expression):
        return "error: illegal character"
    try:
        val = eval(expression, {"__builtins__": {}}, {"math": math})
        return str(val)
    except Exception as e:
        return f"error: {e}"
```

### 6.2 `tools/rag_query.py`
```python
import requests, os

def run(query: str, k: int = 8):
    port = os.getenv("RAG_PORT", "8001")
    url = f"http://localhost:{port}/search"
    try:
        r = requests.post(url, json={"query": query, "k": k}, timeout=20)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}
```

### 6.3 `tools/code_exec.py`
```python
import subprocess, tempfile, os, sys

WHITELIST_IMPORTS = {"math", "statistics", "json", "re"}

def run(code: str, timeout=2):
    banned = ["import os", "import sys", "open(", "subprocess", "socket", "shutil", "requests", "pathlib", "eval(", "exec("]
    if any(b in code for b in banned):
        return "error: banned API"
    lines = []
    for ln in code.splitlines():
        if ln.strip().startswith("import "):
            mod = ln.split()[1].split(".")[0]
            if mod not in WHITELIST_IMPORTS:
                return f"error: module '{mod}' not allowed"
        lines.append(ln)
    code = "\n".join(lines)
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code)
        tmp = f.name
    try:
        out = subprocess.run([sys.executable, tmp], capture_output=True, text=True, timeout=timeout)
        return (out.stdout + out.stderr).strip()
    except subprocess.TimeoutExpired:
        return "error: timeout"
    finally:
        try: os.remove(tmp)
        except Exception: pass
```

### 6.4 `tools/web_search.py`
```python
def run(query: str):
    return {"note": "web.search is stubbed; use rag.search instead"}
```

---

## 7) Environments (agentic RL targets)

### 7.1 `envs/ragqa_wrapper.py`
```python
import random
from typing import Dict, Any

class RAGQALite:
    def __init__(self, retriever_func):
        self.ret = retriever_func
        self.step_penalty = 0.01
        self.goal = None
        self.steps = 0

    def reset(self):
        qs = [
            ("What is 12*7+5?", "89"),
            ("Compute 2+2", "4"),
        ]
        self.goal = random.choice(qs)
        self.steps = 0
        return {"question": self.goal[0]}

    def step(self, action: Dict[str, Any]):
        self.steps += 1
        rew = -self.step_penalty
        done = False
        obs = {}

        if isinstance(action, dict) and action.get("tool") == "rag.search":
            docs = self.ret(action["arguments"]["query"], k=4)
            obs = {"docs": docs}
        elif isinstance(action, dict) and action.get("tool") == "calculator":
            expr = action["arguments"]["expression"]
            try:
                val = eval(expr, {"__builtins__": {}}, {})
                obs = {"calc": str(val)}
            except Exception: obs = {"calc": "error"}
        else:
            ans = str(action)
            done = True
            rew = 1.0 if ans.strip() == self.goal[1] else 0.0
            if "[cite:" in ans: rew += 0.2

        return obs, rew, done, {}
```

---

## 8) Serving

### 8.1 `serving/router.py`
```python
import json
from tools import calculator, rag_query, code_exec, web_search

REGISTRY = {
    "calculator": calculator.run,
    "rag.search": lambda **kw: rag_query.run(**kw),
    "code.exec": lambda **kw: code_exec.run(**kw),
    "web.search": lambda **kw: web_search.run(**kw),
}

def route(tool_call_json: str):
    obj = json.loads(tool_call_json)
    name = obj["tool"]; args = obj.get("arguments", {})
    if name not in REGISTRY:
        return {"error": f"unknown tool {name}"}
    fn = REGISTRY[name]
    return fn(**args)
```

### 8.2 `serving/server.py`
```python
import os, json
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from modeling.tool_head import parse_tool_call
from modeling.tokenizer import load_tokenizer
from .router import route

app = FastAPI(title="MM-SRM Chat Server")
MODEL = None
TOK = None

class ChatReq(BaseModel):
    messages: list  # [{role, content}]

@app.on_event("startup")
async def _startup():
    global MODEL, TOK
    path = os.getenv("MM_SRM_MODEL_PATH", "sshleifer/tiny-gpt2")
    TOK = load_tokenizer(path)
    MODEL = AutoModelForCausalLM.from_pretrained(path)

@app.post("/chat")
def chat(req: ChatReq):
    prompt = ""
    for m in req.messages:
        prompt += f"{m['role'].upper()}: {m['content']}\n"
    prompt += "ASSISTANT:"
    ids = TOK.encode(prompt, return_tensors="pt")
    out = MODEL.generate(ids, max_new_tokens=128)
    text = TOK.decode(out[0], skip_special_tokens=True)
    call = parse_tool_call(text)
    if call:
        obs = route(json.dumps(call))
        obs_text = json.dumps(obs)[:800]
        return {"assistant": f"<tool>{json.dumps(call)}</tool>\n<obs>{obs_text}</obs>"}
    else:
        return {"assistant": text[len(prompt):].strip()}
```

---

## 9) Training Scripts

### 9.1 `scripts/setup.sh`
```bash
#!/usr/bin/env bash
set -euo pipefail
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel
pip install -r requirements.txt
echo "Done."
```

### 9.2 `scripts/build_index.py`
```python
import argparse, os
from rag.retriever import Retriever

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=str, default="corpus")
    args = ap.parse_args()
    os.makedirs(".rag", exist_ok=True)
    ret = Retriever()
    ret.build(args.docs)
    print("Built index.")
```

### 9.3 `scripts/make_synthetic_data.py`
```python
import argparse, os, json

SFT = [
  {
    "id": "ex1",
    "messages": [
      {"role": "system", "content": "You are grounded. Prefer citations."},
      {"role": "user", "content": "What is 12*7+5?"},
      {"role": "assistant", "content": "<tool>{\"tool\":\"calculator\",\"arguments\":{\"expression\":\"12*7+5\"}}</tool>"},
      {"role": "assistant", "content": "<obs>89</obs>"},
      {"role": "assistant", "content": "89"}
    ]
  }
]

PREF = [
  {"id":"p1","prompt":{"role":"user","content":"Compute 3*3+1"},
   "chosen":{"content":"10"},"rejected":{"content":"9"},"reasons":["arithmetic"]}
]

RL = [
  {"env":"ragqa-lite","obs0":{"question":"What is 2+2?"},"act0":"<tool>{\"tool\":\"calculator\",\"arguments\":{\"expression\":\"2+2\"}}</tool>","r0":0.0,
   "obs1":{"calc":"4"},"act1":"4","terminal":True,"R":1.2}
]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="data/")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out,"sft_dialogue.jsonl"),"w") as f:
        for r in SFT: f.write(json.dumps(r)+"\n")
    with open(os.path.join(args.out,"pref_pairs.jsonl"),"w") as f:
        for r in PREF: f.write(json.dumps(r)+"\n")
    with open(os.path.join(args.out,"rl_trajectories.jsonl"),"w") as f:
        for r in RL: f.write(json.dumps(r)+"\n")
    print("Synthetic data written.")
```

### 9.4 `scripts/train_sft.py`
```python
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from modeling.tokenizer import load_tokenizer

def collate(batch, tok, ctx=2048):
    texts = []
    for ex in batch:
        msgs = ex["messages"]
        t = ""
        for m in msgs: t += f"{m['role'].upper()}: {m['content']}\n"
        texts.append(t + "ASSISTANT:")
    enc = tok(texts, padding=True, truncation=True, max_length=ctx, return_tensors="pt")
    enc["labels"] = enc["input_ids"].clone()
    return enc

if __name__ == "__main__":
    import yaml, os
    ap = argparse.ArgumentParser()
    ap.add_argument("-c","--config", dest="cfg", default="configs/train_sft.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.cfg))

    tok = load_tokenizer(cfg.get("model_path","sshleifer/tiny-gpt2"))
    model = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2")
    ds = load_dataset("json", data_files=cfg["data_path"])["train"]

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=cfg["save_dir"],
            per_device_train_batch_size=cfg["batch_size"],
            max_steps=cfg["max_steps"],
            learning_rate=cfg["lr"],
            bf16=cfg.get("mixed_precision","")=="bf16",
            logging_steps=20, save_steps=200, report_to=[],
        ),
        train_dataset=ds,
        data_collator=lambda b: collate(b, tok),
        tokenizer=tok
    )
    trainer.train()
    trainer.save_model(cfg["save_dir"])
```

### 9.5 `scripts/train_dpo.py` (toy DPO via supervised reweighting)
```python
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments

def make_dpo_text(ex):
    p = ex["prompt"]["content"]
    ch = ex["chosen"]["content"]
    return f"USER: {p}\nASSISTANT: {ch}\n"

if __name__ == "__main__":
    import yaml
    ap = argparse.ArgumentParser()
    ap.add_argument("-c","--config", dest="cfg", default="configs/train_dpo.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.cfg))

    tok = AutoTokenizer.from_pretrained(cfg["reference_ckpt"])
    model = AutoModelForCausalLM.from_pretrained(cfg["reference_ckpt"])
    ds = load_dataset("json", data_files=cfg["data_path"])["train"]
    ds = ds.map(lambda ex: {"text": make_dpo_text(ex)})
    def collate(b):
        enc = tok([x["text"] for x in b], padding=True, truncation=True, max_length=2048, return_tensors="pt")
        enc["labels"] = enc["input_ids"].clone()
        return enc

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=cfg["save_dir"],
            per_device_train_batch_size=cfg["batch_size"],
            max_steps=cfg["max_steps"],
            learning_rate=2e-5, logging_steps=20, save_steps=200, report_to=[],
        ),
        train_dataset=ds, data_collator=collate, tokenizer=tok
    )
    trainer.train()
    trainer.save_model(cfg["save_dir"])
```

### 9.6 `scripts/train_rl.py` (GRPO-lite placeholder)
```python
import argparse
from envs.ragqa_wrapper import RAGQALite
from tools.rag_query import run as rag_run

def make_env():
    def retr(q, k=4):
        r = rag_run(q, k=k)
        return r.get("docs", [])
    return RAGQALite(retr)

def policy(obs):
    q = obs["question"]
    if any(c in q for c in "+-*/"):
        expr = q.split("?")[0].replace("What is","").replace("Compute","").strip()
        return {"tool":"calculator","arguments":{"expression":expr}}
    return "unknown"

if __name__ == "__main__":
    import yaml
    ap = argparse.ArgumentParser()
    ap.add_argument("-c","--config", dest="cfg", default="configs/train_rl.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.cfg))

    env = make_env()
    steps = cfg["train_steps"]
    R = []
    for t in range(steps):
        obs = env.reset()
        done = False
        G = 0.0
        while not done:
            act = policy(obs)
            obs, r, done, _ = env.step(act)
            G += r
        R.append(G)
        if (t+1) % 50 == 0:
            avg = sum(R[-50:]) / 50.0
            print(f"[{t+1}/{steps}] avgR@50 = {avg:.3f}")
    print("RL done (toy).")
```

### 9.7 `scripts/eval_benchmarks.py`
```python
print("Add your evals; for now this is a placeholder that always succeeds.")
```

---

## 10) Evaluation
- **Reasoning (text):** run synthetic GSM-like prompts; compute EM.
- **Grounding:** % answers containing a citation tag `[cite:id]` and % where the doc text contains the asserted string.
- **Tool efficiency:** average #tool calls per success; malformed JSON rate.

---

## 11) Safety & Logging
- Tools: allowlist + timeouts; code exec denies FS/network.
- RAG: store provenance (source, id). Avoid leaking local paths in user-visible text.
- Logging: persist `{prompt, tool_calls, observations, citations}` per response for audits.

---

## 12) Optional: Integrate **reasoning-from-scratch**
`third_party/rfs_bridge.py`
```python
def load_base_llm_from_rfs(cfg):
    raise NotImplementedError("Integrate after adding submodule.")
def rfs_ppo_compat(rollouts):
    raise NotImplementedError("Integrate after adding submodule.")
```

---

## 13) README snippet for your repo
```md
# MM-SRM
Small multimodal reasoning model with tools and RAG. Train SFT → DPO → Agentic RL, then serve with FastAPI.

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/build_index.py --docs ./corpus
python scripts/make_synthetic_data.py --out data/
python scripts/train_sft.py -c configs/train_sft.yaml
uvicorn rag.api:app --port 8001 --reload &
uvicorn serving.server:app --port 8000 --reload
```
```

---

## 14) Next Steps
- Swap TinyLM for a real 1–2B checkpoint + LoRA.
- Add constrained JSON decoding (`outlines`) in `serving/server.py` generation call.
- Expand RL envs (WebShop/WebArena adapters) and reward functions.
- Add reranker & HyDE to RAG.

---

## 15) License
Apache-2.0 (add a LICENSE file if you publish).
