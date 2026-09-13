# LLM Inference Frameworks: TensorRT, TGI, vLLM, SGLang

This document covers the four dominant frameworks for deploying large language models at scale.  
Each solves a different layer of the inference problem.

---

## The Inference Problem

Training a 7B parameter model takes weeks and petaflops. Serving it in production has a completely different set of challenges:

| Challenge | Problem |
|---|---|
| **Latency** | Time to first token must be < 1s for interactive use |
| **Throughput** | Serve many concurrent users without 10× more GPUs |
| **Memory** | A 7B model in fp16 = 14 GB VRAM — leaves little room for activations |
| **KV Cache** | Each generated token requires storing attention keys/values for all previous tokens |
| **Batching** | Requests arrive at different times with different lengths — static batching wastes GPU |

---

## 1. TensorRT

**What it is**: NVIDIA's compiler and runtime for optimising neural networks specifically for NVIDIA GPUs.

**Layer in the stack**: Sits at the lowest level — takes a trained model and compiles it into a highly optimised NVIDIA-specific engine.

### Core Optimisations

#### 1. Graph Fusion
Multiple operations are merged into a single CUDA kernel, eliminating memory round-trips:

```
BEFORE: Linear → LayerNorm → GELU  (3 separate kernel launches, 3× memory read/write)
AFTER:  FusedLinearLayerNormGELU    (1 kernel, data stays in registers)
```

#### 2. Quantisation (INT8 / FP8 / INT4)

```python
# TensorRT-LLM quantisation config
from tensorrt_llm.quantization import QuantConfig, QuantAlgo

quant_config = QuantConfig(
    quant_algo=QuantAlgo.W4A16,    # 4-bit weights, 16-bit activations
    kv_cache_quant_algo=QuantAlgo.FP8,
)
```

Memory reduction: fp16 (16-bit) → int8 (8-bit) = 2× smaller model, ~1.5× faster inference with minimal accuracy loss.

#### 3. TensorRT-LLM: LLM-Specific Optimisations

```python
import tensorrt_llm
from tensorrt_llm.models import LLaMAForCausalLM
from tensorrt_llm.builder import BuildConfig

# Build an optimised engine from a HuggingFace checkpoint
build_config = BuildConfig(
    max_input_len=2048,
    max_output_len=512,
    max_batch_size=8,
    strongly_typed=True,
)

# In-flight batching, paged KV cache, multi-query attention — all compiled in
llama = LLaMAForCausalLM.from_hugging_face(
    'meta-llama/Llama-2-7b-hf',
    dtype='float16',
)
engine = tensorrt_llm.build(llama, build_config)
engine.save('./llama_engine')
```

#### 4. Inference with the built engine

```python
from tensorrt_llm.runtime import ModelRunner
import torch

runner = ModelRunner.from_dir('./llama_engine')

input_ids = torch.tensor([[1, 450, 3437, 310, 4272]])  # tokenised input
outputs = runner.generate(
    input_ids,
    max_new_tokens=100,
    temperature=0.8,
    top_p=0.95,
)
print(outputs)
```

### When to Use TensorRT

- NVIDIA GPU deployment (A100, H100, L40S, RTX 4090)
- Need maximum single-request throughput
- Production where you can afford a multi-hour compilation step
- Used as the backend by Triton Inference Server

### Limitations

- NVIDIA only (no AMD, no CPU)
- Compilation can take hours for large models
- Requires re-compilation when model changes

---

## 2. TGI (Text Generation Inference)

**What it is**: HuggingFace's production-grade serving framework for LLMs.

**Layer in the stack**: HTTP serving layer with LLM-specific batching. Runs on top of PyTorch or TensorRT backends.

### Architecture

```
Client (HTTP/gRPC)
      │
      ▼
┌─────────────────────────────┐
│  TGI Router (Rust)          │  ← handles HTTP, queuing, load balancing
│  - Request batching         │
│  - Token streaming (SSE)    │
│  - Health checks            │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  Python Model Server        │  ← runs the actual model
│  - Flash Attention 2        │
│  - Paged KV cache           │
│  - Continuous batching      │
│  - Tensor parallelism       │
└─────────────────────────────┘
```

### Key Feature: Continuous Batching

Classic batching waits for a full batch before running inference — wasteful because different requests finish at different times. TGI uses **continuous batching** (also called in-flight batching):

```
Time →  0    1    2    3    4    5
Req A:  ████ ████ ████ done
Req B:  ████ ████ done
Req C:             ← NEW request joins immediately after B finishes
Req D:                  ████ ████

Static batching:   [A,B    ]   [    ] [C,D  ]    (GPU idle between batches)
Continuous:        [A,B    ]   [A,C ]  [A,D  ]   (GPU always busy)
```

### Running TGI

```bash
# Docker — the simplest way
docker run --gpus all --shm-size 1g \
    -p 8080:80 \
    -v $PWD/data:/data \
    ghcr.io/huggingface/text-generation-inference:2.0 \
    --model-id meta-llama/Meta-Llama-3-8B-Instruct \
    --max-concurrent-requests 128 \
    --quantize bitsandbytes-nf4
```

### Client API

```python
import requests

# OpenAI-compatible generate endpoint
response = requests.post(
    "http://localhost:8080/generate",
    json={
        "inputs": "What is the capital of France?",
        "parameters": {
            "max_new_tokens": 100,
            "temperature": 0.7,
            "top_p": 0.95,
            "do_sample": True,
        }
    }
)
print(response.json()["generated_text"])

# Streaming (Server-Sent Events)
import sseclient
response = requests.post(
    "http://localhost:8080/generate_stream",
    json={"inputs": "Tell me a story", "parameters": {"max_new_tokens": 200}},
    stream=True,
)
client = sseclient.SSEClient(response)
for event in client.events():
    print(event.data, end='', flush=True)
```

### TGI Built-in Optimisations

| Optimisation | Description |
|---|---|
| **Flash Attention 2** | Fused attention kernel, O(1) memory w.r.t. seq length |
| **Paged KV Cache** | Allocates KV memory in pages, eliminates fragmentation |
| **Tensor Parallelism** | Splits model across multiple GPUs (via `--num-shard`) |
| **Speculative Decoding** | Draft model generates tokens; main model verifies in parallel |
| **GPTQ / AWQ / NF4** | On-the-fly quantisation support |

---

## 3. vLLM

**What it is**: A high-throughput inference engine centred around **PagedAttention** — a memory management innovation that dramatically increases GPU utilisation.

**Origin**: UC Berkeley Sky Computing Lab (Kwon et al., 2023)

### The KV Cache Problem

During autoregressive generation, each token requires storing its key and value vectors for all previous tokens. For a batch of requests with variable lengths, naive allocation wastes memory:

```
Request A (max 512 tokens):  [KV KV KV ... KV |  empty space  ]  <- wasted
Request B (max 512 tokens):  [KV KV |        empty space       ]  <- wasted
```

Memory fragmentation means only 20–40% of KV cache memory is actually used in practice.

### PagedAttention

Inspired by virtual memory in operating systems. KV cache is divided into fixed-size **pages** (blocks), allocated on demand:

```
Physical KV memory (divided into pages of 16 tokens each):
┌────┬────┬────┬────┬────┬────┬────┬────┐
│ P0 │ P1 │ P2 │ P3 │ P4 │ P5 │ P6 │ P7 │
└────┴────┴────┴────┴────┴────┴────┴────┘

Request A (grows dynamically):  [P0] → [P0, P3] → [P0, P3, P6]
Request B (grows dynamically):  [P1] → [P1, P4]
Request C:                       [P2] → [P2, P5, P7]

No internal fragmentation — every page is fully used.
```

This allows vLLM to batch 2–4× more requests simultaneously, increasing throughput proportionally.

### Using vLLM

```python
from vllm import LLM, SamplingParams

# Load model — downloads and caches automatically
llm = LLM(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    tensor_parallel_size=1,      # set to number of GPUs for multi-GPU
    gpu_memory_utilization=0.90, # fraction of GPU memory for KV cache
    max_model_len=8192,
    quantization="awq",          # or "gptq", "fp8", None
)

sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=512,
)

# Offline batch inference
prompts = [
    "Explain quantum entanglement in simple terms.",
    "Write a Python function to merge two sorted lists.",
    "What are the main causes of the French Revolution?",
]
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(f"Prompt: {output.prompt[:50]}...")
    print(f"Output: {output.outputs[0].text}\n")
```

### vLLM OpenAI-Compatible Server

```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --port 8000 \
    --tensor-parallel-size 2 \
    --max-model-len 8192
```

```python
# Use exactly like the OpenAI Python SDK
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="token")

completion = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=100,
)
print(completion.choices[0].message.content)
```

### vLLM Advanced Features

#### Speculative Decoding
```python
llm = LLM(
    model="meta-llama/Meta-Llama-3-70B-Instruct",   # main (verifier) model
    speculative_model="meta-llama/Meta-Llama-3-8B-Instruct",  # draft model
    num_speculative_tokens=5,  # draft 5 tokens per step
)
# Typically 2-3× speedup for the 70B model at same output quality
```

#### Prefix Caching
```python
# vLLM automatically caches the KV of shared prefixes (e.g. system prompts)
llm = LLM(model="...", enable_prefix_caching=True)
# If 1000 requests share the same system prompt, its KV is computed once
```

---

## 4. SGLang

**What it is**: A language for **structured generation** + a high-performance runtime that exploits structure to cache and reuse computation.

**Origin**: LMSYS (Zheng et al., 2023) — the team behind Vicuna, FastChat, and LMSYS Chatbot Arena.

### The Core Innovation: RadixAttention

Most LLM calls share prefixes — system prompts, few-shot examples, document context. SGLang builds a **radix tree** (trie) of all KV cache prefixes, enabling:

```
System prompt KV: "You are a helpful assistant..."  [computed once]
        │
        ├── "Summarise this: ..."   [user 1]
        ├── "Summarise this: ..."   [user 2, same prefix → KV reused!]
        └── "Translate this: ..."   [user 3, different branch]
```

This is **prefix caching at a finer granularity** than vLLM — any shared subsequence is cached, not just common prefixes.

### SGLang Programs

SGLang lets you write LLM programs as Python functions with `@sgl.function`:

```python
import sglang as sgl

@sgl.function
def multi_turn_qa(s, question):
    s += sgl.system("You are a concise assistant.")
    s += sgl.user(question)
    s += sgl.assistant(sgl.gen("answer", max_tokens=100))
    s += sgl.user("Give a one-sentence summary of your answer.")
    s += sgl.assistant(sgl.gen("summary", max_tokens=30))
    return s

# The runtime batches and schedules these automatically
runtime = sgl.Runtime(model_path="meta-llama/Meta-Llama-3-8B-Instruct")
sgl.set_default_backend(runtime)

state = multi_turn_qa.run(question="What is gradient descent?")
print(state["answer"])
print(state["summary"])
runtime.shutdown()
```

### Structured Generation (Constrained Decoding)

SGLang enforces output format at the token level — guaranteed JSON, regex, grammar:

```python
@sgl.function
def extract_entities(s, text):
    s += sgl.system("Extract structured data from text.")
    s += sgl.user(f"Text: {text}\n\nExtract as JSON:")
    s += sgl.assistant(
        sgl.gen(
            "entities",
            max_tokens=200,
            # Constrain output to valid JSON matching this schema
            json_schema={
                "type": "object",
                "properties": {
                    "names":     {"type": "array", "items": {"type": "string"}},
                    "locations": {"type": "array", "items": {"type": "string"}},
                    "dates":     {"type": "array", "items": {"type": "string"}},
                },
            }
        )
    )

state = extract_entities.run(
    text="Albert Einstein was born in Ulm on March 14, 1879."
)
import json
print(json.loads(state["entities"]))
# → {"names": ["Albert Einstein"], "locations": ["Ulm"], "dates": ["March 14, 1879"]}
```

### Parallel Generation (Fork / Join)

```python
@sgl.function
def generate_and_critique(s, topic):
    s += sgl.user(f"Write a short essay on: {topic}")
    s += sgl.assistant(sgl.gen("essay", max_tokens=300))

    # Fork: run 3 critiques in parallel from the same prefix
    forks = s.fork(3)
    for i, f in enumerate(forks):
        f += sgl.user(f"Critique #{i+1}: identify one weakness in the essay above.")
        f += sgl.assistant(sgl.gen(f"critique_{i}", max_tokens=100))

    # All critiques share the essay prefix KV → only computed once
    s += sgl.user("Synthesise the critiques into improvements.")
    s += sgl.assistant(sgl.gen("improvements", max_tokens=150))

state = generate_and_critique.run(topic="the benefits of exercise")
```

### SGLang OpenAI-Compatible Server

```bash
python -m sglang.launch_server \
    --model-path meta-llama/Meta-Llama-3-8B-Instruct \
    --port 30000 \
    --tp 1 \
    --enable-flashinfer \
    --disable-radix-cache   # disable to compare with/without prefix caching
```

---

## Framework Comparison

| | TensorRT-LLM | TGI | vLLM | SGLang |
|---|---|---|---|---|
| **Primary focus** | Compiler optimisation | Production serving | KV memory efficiency | Structured generation & prefix caching |
| **Key innovation** | Kernel fusion, quantisation | Continuous batching | PagedAttention | RadixAttention |
| **Hardware** | NVIDIA only | Any GPU | Any GPU | Any GPU |
| **Setup complexity** | High (compilation step) | Low (Docker) | Low (pip) | Low (pip) |
| **Throughput** | Highest single-GPU | High | High | Very high with shared prefixes |
| **Structured output** | No (need separate layer) | Limited | Limited | First-class |
| **Context length** | Excellent | Good | Excellent | Excellent |
| **Multi-GPU** | Yes (tensor parallel) | Yes (`--num-shard`) | Yes (`tensor_parallel_size`) | Yes (`--tp`) |
| **Quantisation** | INT4/8/FP8 (native) | GPTQ/AWQ/NF4 | GPTQ/AWQ/FP8 | GPTQ/AWQ/FP8 |
| **Best for** | Max perf on NVIDIA | HuggingFace ecosystem | General-purpose serving | Agent/multi-turn apps |

---

## Memory Layout: KV Cache Comparison

```
STATIC BATCHING (baseline):
┌────────────────────────────────────────┐
│ Req A KV (max_len allocated = 2048)    │  ← 60% empty if avg len is 800
│ Req B KV (max_len allocated = 2048)    │  ← 70% empty
│ ...                                    │
└────────────────────────────────────────┘
Utilisation: ~30%

vLLM PAGEDATTENTION:
┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐
│A0 │A1 │B0 │A2 │C0 │B1 │C1 │A3 │B2 │C2 │  ← pages allocated on demand
└───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘
Utilisation: ~95%

SGLang RADIXATTENTION:
        system_prompt KV (shared across 1000 requests)
               │
       ┌───────┴──────────┐
       │                  │
   few_shot_ex          few_shot_ex
    KV (1000)            KV (1000)
       │                  │
    [req 1..500]      [req 501..1000]
Reuse ratio: system prompt computed once, not 1000 times.
```

---

## Throughput Benchmarks (approximate, A100 80GB)

These numbers vary significantly by model size, sequence length, batch size, and hardware.

| Framework | Model | Tokens/sec (throughput) | Notes |
|---|---|---|---|
| Baseline (HF generate) | LLaMA-3-8B | ~800 | No batching optimisations |
| TGI | LLaMA-3-8B | ~3,500 | Continuous batching |
| vLLM | LLaMA-3-8B | ~4,200 | PagedAttention |
| SGLang | LLaMA-3-8B | ~4,500 | RadixAttention + flashinfer |
| TensorRT-LLM | LLaMA-3-8B | ~5,500 | Compiled INT8 engine |

**Takeaway**: All four are 4–7× faster than the baseline. Choice depends on use-case, not raw speed.

---

## Which Framework to Choose?

```
Is output structure critical (JSON, regex, grammar)?
    YES → SGLang

Do you have shared prefixes (system prompts, few-shot, RAG docs)?
    YES → SGLang (best prefix caching) or vLLM (good prefix caching)

Do you need maximum raw throughput on NVIDIA?
    YES → TensorRT-LLM

Do you need simple deployment from HuggingFace models?
    YES → TGI (docker run in one command)

Do you need a general-purpose OpenAI-compatible server?
    → vLLM (most widely used, best community support)

Do you need AMD GPU support?
    → vLLM (ROCm support) or TGI (ROCm support)
```

---

## Stack Diagram: Where Each Framework Lives

```
┌──────────────────────────────────────────────────┐
│            Your Application / API                │
├──────────────────────────────────────────────────┤
│  HTTP/gRPC Layer   TGI  │  vLLM  │  SGLang       │  ← serving layer
├──────────────────────────────────────────────────┤
│  Batching Engine         Continuous / Paged      │  ← scheduling layer
├──────────────────────────────────────────────────┤
│  Model Runtime    PyTorch │ TensorRT-LLM          │  ← compute layer
├──────────────────────────────────────────────────┤
│  CUDA Kernels   FlashAttention │ cuBLAS │ Triton  │  ← kernel layer
├──────────────────────────────────────────────────┤
│                  GPU Hardware                    │
└──────────────────────────────────────────────────┘
```

TensorRT-LLM operates at the **kernel + runtime** layers.  
TGI, vLLM, and SGLang operate at the **serving + scheduling + runtime** layers — they can use TensorRT-LLM as their backend.