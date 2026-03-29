# Chimera API Reference

## Overview

This document provides comprehensive API reference for Chimera's Python client, server endpoints, and kernel interfaces.

## Table of Contents

- [Python Client API](#python-client-api)
- [Server HTTP API](#server-http-api)
- [OpenAI-Compatible API](#openai-compatible-api)
- [Kernel API](#kernel-api)
- [Configuration API](#configuration-api)
- [Error Codes](#error-codes)

---

## Python Client API

### Engine Interface

The `Engine` class provides synchronous inference capabilities.

```python
from chimera import Engine
# or
from sglang import Engine
```

#### Constructor

```python
engine = Engine(
    model_path: str,
    tokenizer_path: Optional[str] = None,
    load_format: str = "auto",
    tokenizer_mode: str = "auto",
    trust_remote_code: bool = False,
    mem_fraction_static: float = 0.9,
    tp_size: int = 1,
    device: str = "cuda",
    schedule_policy: str = "lpm",
    enable_prefix_caching: bool = True,
    disable_radix_cache: bool = False,
    random_seed: int = 42,
    log_level: str = "warning",
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_path` | str | - | Path to model or HuggingFace model ID |
| `tokenizer_path` | str | None | Path to tokenizer (defaults to model_path) |
| `load_format` | str | "auto" | Model loading format (auto, pt, safetensors) |
| `tokenizer_mode` | str | "auto" | Tokenizer mode (auto, slow) |
| `trust_remote_code` | bool | False | Trust remote code in model |
| `mem_fraction_static` | float | 0.9 | Fraction of GPU memory for static allocation |
| `tp_size` | int | 1 | Tensor parallelism size |
| `device` | str | "cuda" | Device to run on (cuda, cpu) |
| `schedule_policy` | str | "lpm" | Scheduling policy (lpm, random, fcfs) |
| `enable_prefix_caching` | bool | True | Enable prefix caching for radix attention |
| `disable_radix_cache` | bool | False | Disable radix cache |
| `random_seed` | int | 42 | Random seed for reproducibility |
| `log_level` | str | "warning" | Logging level |

#### Methods

##### `generate()`

Generate text completion.

```python
response = engine.generate(
    prompt: str,
    sampling_params: Optional[SamplingParams] = None,
    return_logprob: bool = False,
    top_logprobs_num: int = 0,
    lora_path: Optional[str] = None,
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | str | - | Input prompt text |
| `sampling_params` | SamplingParams | None | Sampling parameters |
| `return_logprob` | bool | False | Return log probabilities |
| `top_logprobs_num` | int | 0 | Number of top log probs to return |
| `lora_path` | str | None | Path to LoRA adapter |

**Returns:** `GenerateResponse` object

**Example:**
```python
from chimera import Engine, SamplingParams

engine = Engine(model_path="meta-llama/Llama-3.1-8B-Instruct")

params = SamplingParams(
    temperature=0.7,
    max_new_tokens=256,
    top_p=0.95,
)

response = engine.generate(
    prompt="What is the capital of France?",
    sampling_params=params,
)

print(response.text)
print(f"Tokens: {response.completion_tokens}")
```

##### `generate_batch()`

Batch generation for multiple prompts.

```python
responses = engine.generate_batch(
    prompts: List[str],
    sampling_params: Optional[SamplingParams] = None,
)
```

**Example:**
```python
prompts = [
    "What is Python?",
    "What is Rust?",
    "What is Go?",
]

responses = engine.generate_batch(
    prompts=prompts,
    sampling_params=SamplingParams(max_new_tokens=128),
)

for prompt, response in zip(prompts, responses):
    print(f"Q: {prompt}")
    print(f"A: {response.text}\n")
```

##### `encode()`

Encode text to embeddings.

```python
embeddings = engine.encode(
    text: Union[str, List[str]],
)
```

**Example:**
```python
text = "Hello, world!"
embedding = engine.encode(text)
print(f"Embedding shape: {embedding.shape}")
```

##### `shutdown()`

Shutdown the engine and release resources.

```python
engine.shutdown()
```

---

### Runtime Interface

The `Runtime` class provides asynchronous inference with a server backend.

```python
from chimera import Runtime
```

#### Constructor

```python
runtime = Runtime(
    server_url: str = "http://localhost:30000",
    api_key: Optional[str] = None,
    timeout: int = 600,
)
```

#### Methods

##### `generate()`

```python
async def generate(
    prompt: str,
    sampling_params: Optional[SamplingParams] = None,
    stream: bool = False,
) -> Union[GenerateResponse, AsyncIterator[GenerateResponse]]:
```

**Example:**
```python
import asyncio
from chimera import Runtime, SamplingParams

async def main():
    runtime = Runtime(server_url="http://localhost:30000")
    
    # Non-streaming
    response = await runtime.generate(
        prompt="Tell me a joke",
        sampling_params=SamplingParams(max_new_tokens=100),
    )
    print(response.text)
    
    # Streaming
    async for chunk in await runtime.generate(
        prompt="Write a story",
        sampling_params=SamplingParams(max_new_tokens=500),
        stream=True,
    ):
        print(chunk.text, end="", flush=True)

asyncio.run(main())
```

##### `chat()`

Chat completion with conversation history.

```python
async def chat(
    messages: List[Dict[str, str]],
    sampling_params: Optional[SamplingParams] = None,
) -> ChatResponse:
```

**Example:**
```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is machine learning?"},
]

response = await runtime.chat(messages=messages)
print(response.content)
```

---

### SamplingParams

Configuration for text generation sampling.

```python
from chimera import SamplingParams

params = SamplingParams(
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = -1,
    min_p: float = 0.0,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    repetition_penalty: float = 1.0,
    min_new_tokens: int = 0,
    max_new_tokens: int = 128,
    stop: Optional[Union[str, List[str]]] = None,
    stop_token_ids: Optional[List[int]] = None,
    spaces_between_special_tokens: bool = True,
    n: int = 1,
    logprobs: Optional[int] = None,
    top_logprobs_num: Optional[int] = None,
    response_format: Optional[Dict] = None,
    guided_json: Optional[Union[str, Dict]] = None,
    guided_regex: Optional[str] = None,
    guided_choice: Optional[List[str]] = None,
    guided_grammar: Optional[str] = None,
    ignore_eos: bool = False,
    skip_special_tokens: bool = True,
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `temperature` | float | 1.0 | Sampling temperature (0 = greedy) |
| `top_p` | float | 1.0 | Nucleus sampling probability |
| `top_k` | int | -1 | Top-k sampling (-1 = disabled) |
| `min_p` | float | 0.0 | Minimum token probability |
| `frequency_penalty` | float | 0.0 | Penalize frequent tokens |
| `presence_penalty` | float | 0.0 | Penalize present tokens |
| `repetition_penalty` | float | 1.0 | Penalize repetition |
| `min_new_tokens` | int | 0 | Minimum tokens to generate |
| `max_new_tokens` | int | 128 | Maximum tokens to generate |
| `stop` | str/List | None | Stop sequences |
| `stop_token_ids` | List[int] | None | Stop token IDs |
| `n` | int | 1 | Number of completions to generate |
| `logprobs` | int | None | Number of log probs to return |
| `guided_json` | str/Dict | None | JSON schema for guided generation |
| `guided_regex` | str | None | Regex pattern for guided generation |
| `guided_choice` | List[str] | None | Force choice from list |
| `guided_grammar` | str | None | CFG grammar for guided generation |

**Example:**
```python
# Creative writing
creative = SamplingParams(
    temperature=0.9,
    top_p=0.95,
    max_new_tokens=500,
)

# Factual QA
factual = SamplingParams(
    temperature=0.1,
    top_p=0.9,
    max_new_tokens=256,
)

# JSON output
json_params = SamplingParams(
    guided_json={
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "number"},
        },
    },
    max_new_tokens=200,
)
```

---

## Server HTTP API

### Endpoints

Base URL: `http://localhost:3000`

#### `POST /generate`

Generate text completion.

**Request:**
```json
{
    "text": "What is the capital of France?",
    "sampling_params": {
        "temperature": 0.7,
        "max_new_tokens": 256,
        "top_p": 0.95
    },
    "stream": false,
    "return_logprob": false
}
```

**Response:**
```json
{
    "text": "The capital of France is Paris.",
    "usage": {
        "prompt_tokens": 8,
        "completion_tokens": 12,
        "total_tokens": 20
    },
    "meta_info": {
        "id": "req_abc123",
        "finish_reason": "stop"
    }
}
```

#### `POST /generate_stream`

Stream text completion.

**Request:** Same as `/generate` with `stream: true`

**Response:** Server-Sent Events (SSE)
```
data: {"text": "The", "finish_reason": null}
data: {"text": " capital", "finish_reason": null}
data: {"text": " of", "finish_reason": null}
data: {"text": " France", "finish_reason": null}
data: {"text": " is", "finish_reason": null}
data: {"text": " Paris.", "finish_reason": "stop"}
data: [DONE]
```

#### `POST /v1/chat/completions`

OpenAI-compatible chat completions.

**Request:**
```json
{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "messages": [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello!"}
    ],
    "temperature": 0.7,
    "max_tokens": 256,
    "stream": false
}
```

**Response:**
```json
{
    "id": "chatcmpl_abc123",
    "object": "chat.completion",
    "created": 1234567890,
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Hello! How can I help you today?"
            },
            "finish_reason": "stop"
        }
    ],
    "usage": {
        "prompt_tokens": 15,
        "completion_tokens": 10,
        "total_tokens": 25
    }
}
```

#### `POST /v1/completions`

OpenAI-compatible text completions.

**Request:**
```json
{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "prompt": "Once upon a time",
    "max_tokens": 100,
    "temperature": 0.7
}
```

#### `POST /encode`

Encode text to embeddings.

**Request:**
```json
{
    "text": "Hello, world!",
    "model": "meta-llama/Llama-3.1-8B-Instruct"
}
```

**Response:**
```json
{
    "embedding": [0.123, -0.456, 0.789, ...],
    "usage": {
        "prompt_tokens": 4,
        "total_tokens": 4
    }
}
```

#### `GET /health`

Health check endpoint.

**Response:**
```json
{
    "status": "healthy",
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "uptime": 3600
}
```

#### `GET /metrics`

Prometheus metrics endpoint.

**Response:** Prometheus text format
```
# HELP chimera_request_count Total number of requests
# TYPE chimera_request_count counter
chimera_request_count{type="generate"} 1234
chimera_request_count{type="chat"} 5678

# HELP chimera_token_count Total tokens processed
# TYPE chimera_token_count counter
chimera_token_count{type="prompt"} 123456
chimera_token_count{type="completion"} 234567
```

#### `GET /get_model_info`

Get model information.

**Response:**
```json
{
    "model_path": "meta-llama/Llama-3.1-8B-Instruct",
    "context_length": 131072,
    "vocab_size": 128256,
    "embedding_size": 4096,
    "num_layers": 32,
    "num_heads": 32,
    "head_dim": 128
}
```

#### `POST /flush_cache`

Flush the KV cache.

**Response:**
```json
{
    "success": true,
    "message": "Cache flushed successfully"
}
```

#### `GET /get_server_info`

Get server configuration and statistics.

**Response:**
```json
{
    "version": "0.4.6",
    "commit": "abc123",
    "config": {
        "mem_fraction_static": 0.9,
        "tp_size": 1,
        "max_batch_size": 256
    },
    "stats": {
        "num_running_requests": 5,
        "num_queued_requests": 2,
        "gpu_memory_usage": 0.75
    }
}
```

---

## OpenAI-Compatible API

Chimera provides full OpenAI API compatibility.

### Client Example

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:3000/v1",
    api_key="EMPTY"  # Not required unless configured
)

# Chat completion
response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[
        {"role": "user", "content": "Hello!"}
    ],
    temperature=0.7,
    max_tokens=256,
)

print(response.choices[0].message.content)

# Streaming
stream = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[
        {"role": "user", "content": "Tell me a story"}
    ],
    stream=True,
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

### Supported Endpoints

| OpenAI Endpoint | Chimera Endpoint | Status |
|-----------------|------------------|--------|
| `/v1/chat/completions` | `/v1/chat/completions` | ✅ Full |
| `/v1/completions` | `/v1/completions` | ✅ Full |
| `/v1/embeddings` | `/v1/embeddings` | ✅ Full |
| `/v1/models` | `/v1/models` | ✅ Full |
| `/v1/audio/transcriptions` | - | ❌ Not supported |
| `/v1/images/generations` | - | ❌ Not supported |

---

## Kernel API

### GEMM Operations

#### `cutedsl_fp8_blockwise_scaled_mm`

FP8 blockwise scaled matrix multiplication.

```python
import torch
from sgl_kernel import cutedsl_fp8_blockwise_scaled_mm

# Input tensors (FP8)
mat_a = torch.randn(128, 512, dtype=torch.float8_e4m3fn).cuda()
mat_b = torch.randn(256, 512, dtype=torch.float8_e4m3fn).cuda()

# Scale factors
scales_a = torch.randn(1, 4, dtype=torch.float32).cuda()
scales_b = torch.randn(2, 4, dtype=torch.float32).cuda()

# Execute
output = cutedsl_fp8_blockwise_scaled_mm(
    mat_a=mat_a,
    mat_b=mat_b,
    scales_a=scales_a,
    scales_b=scales_b,
    out_dtype=torch.float16,
)

print(f"Output shape: {output.shape}")  # (128, 256)
```

#### `cutedsl_mla_decode`

Multi-head latent attention decode.

```python
from sgl_kernel import cutedsl_mla_decode

out = torch.zeros(8, 32, 128, dtype=torch.float16).cuda()
q_nope = torch.randn(8, 32, 128, dtype=torch.float16).cuda()
q_pe = torch.randn(8, 32, 128, dtype=torch.float16).cuda()
kv_cache = torch.randn(8192, 256, dtype=torch.float16).cuda()
seq_lens = torch.tensor([1024] * 8, dtype=torch.int32).cuda()
page_table = torch.arange(8, dtype=torch.int32).cuda()
workspace = torch.empty(1048576, dtype=torch.uint8).cuda()

cutedsl_mla_decode(
    out=out,
    q_nope=q_nope,
    q_pe=q_pe,
    kv_c_and_k_pe_cache=kv_cache,
    seq_lens=seq_lens,
    page_table=page_table,
    workspace=workspace,
    sm_scale=1.0 / (128 ** 0.5),
    num_kv_splits=1,
)
```

#### `cutedsl_es_fp8_blockwise_scaled_grouped_mm`

FP8 expert-specialized grouped GEMM for MoE.

```python
from sgl_kernel import cutedsl_es_fp8_blockwise_scaled_grouped_mm

# MoE configuration
num_experts = 8
batch_tokens = 1024
hidden_dim = 4096
intermediate_dim = 11008

# Tensors
output = torch.zeros(batch_tokens, intermediate_dim, dtype=torch.float16).cuda()
a = torch.randn(batch_tokens, hidden_dim, dtype=torch.float8_e4m3fn).cuda()
b = torch.randn(num_experts, hidden_dim, intermediate_dim, dtype=torch.float8_e4m3fn).cuda()

# Configuration tensors
scales_a = torch.randn(batch_tokens, dtype=torch.float32).cuda()
scales_b = torch.randn(num_experts, intermediate_dim, dtype=torch.float32).cuda()
problem_sizes = torch.tensor([[128, intermediate_dim, hidden_dim]] * num_experts, dtype=torch.int32).cuda()
expert_offsets = torch.tensor([0, 128, 256, 384, 512, 640, 768, 896], dtype=torch.int32).cuda()

# Strides and workspace
stride_a = torch.tensor([a.stride(0), a.stride(1)], dtype=torch.int64).cuda()
stride_b = torch.tensor([b.stride(0), b.stride(1), b.stride(2)], dtype=torch.int64).cuda()
stride_d = torch.tensor([hidden_dim], dtype=torch.int64).cuda()
workspace = torch.empty(16777216, dtype=torch.uint8).cuda()

cutedsl_es_fp8_blockwise_scaled_grouped_mm(
    output=output,
    a=a,
    b=b,
    scales_a=scales_a,
    scales_b=scales_b,
    stride_a=stride_a,
    stride_b=stride_b,
    stride_d=stride_d,
    problem_sizes=problem_sizes,
    expert_offsets=expert_offsets,
    workspace=workspace,
)
```

---

## Configuration API

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `CHIMERA_TORCH_PROFILER_DIR` | Directory for torch profiler output | - |
| `CHIMERA_ENABLE_TORCH_COMPILE` | Enable torch compile | 0 |
| `CHIMERA_MEM_FRACTION` | Memory fraction for model | 0.9 |
| `SGLANG_TORCH_PROFILER_DIR` | (Legacy) Same as CHIMERA_* | - |
| `SGLANG_ENABLE_TORCH_COMPILE` | (Legacy) Same as CHIMERA_* | 0 |
| `CUDA_VISIBLE_DEVICES` | GPU device selection | All |
| `NCCL_DEBUG` | NCCL logging level | WARN |
| `FLASHINFER_VERSION` | FlashInfer version to use | Latest |

### Runtime Configuration

```python
from chimera import set_config, get_config

# Set configuration
set_config("mem_fraction", 0.85)
set_config("max_batch_size", 256)
set_config("enable_prefix_caching", True)

# Get configuration
mem_frac = get_config("mem_fraction")
print(f"Memory fraction: {mem_frac}")
```

---

## Error Codes

### HTTP Error Codes

| Code | Meaning | Description |
|------|---------|-------------|
| 200 | OK | Request succeeded |
| 400 | Bad Request | Invalid request format |
| 401 | Unauthorized | Invalid API key |
| 404 | Not Found | Resource not found |
| 429 | Too Many Requests | Rate limit exceeded |
| 500 | Internal Server Error | Server error |
| 503 | Service Unavailable | Server overloaded |

### Application Error Codes

| Code | Meaning | Description |
|------|---------|-------------|
| `INVALID_PROMPT` | Invalid prompt | Prompt is empty or malformed |
| `INVALID_SAMPLING_PARAMS` | Invalid sampling params | Sampling parameters out of range |
| `CONTEXT_LENGTH_EXCEEDED` | Context too long | Input exceeds model context length |
| `OUT_OF_MEMORY` | GPU OOM | Insufficient GPU memory |
| `MODEL_NOT_LOADED` | Model error | Model failed to load |
| `TOKENIZER_ERROR` | Tokenizer error | Tokenization failed |
| `KERNEL_ERROR` | Kernel error | GPU kernel execution failed |

### Error Response Format

```json
{
    "error": {
        "code": "CONTEXT_LENGTH_EXCEEDED",
        "message": "Input length (150000) exceeds maximum context length (131072)",
        "type": "invalid_request_error",
        "param": "text"
    }
}
```

---

## Examples

### Basic Usage

```python
from chimera import Engine, SamplingParams

# Initialize
engine = Engine(
    model_path="meta-llama/Llama-3.1-8B-Instruct",
    mem_fraction_static=0.9,
)

# Generate
response = engine.generate(
    prompt="Explain quantum computing in one paragraph.",
    sampling_params=SamplingParams(
        temperature=0.7,
        max_new_tokens=200,
        top_p=0.95,
    ),
)

print(response.text)
```

### Batch Processing

```python
from chimera import Engine

engine = Engine(model_path="meta-llama/Llama-3.1-8B-Instruct")

prompts = [
    "What is Python?",
    "What is Rust?",
    "What is Go?",
    "What is Zig?",
]

responses = engine.generate_batch(
    prompts=prompts,
    sampling_params={"max_new_tokens": 100},
)

for prompt, response in zip(prompts, responses):
    print(f"Q: {prompt}")
    print(f"A: {response.text}\n")
```

### Streaming

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:3000/v1")

stream = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[
        {"role": "user", "content": "Write a haiku about coding"}
    ],
    stream=True,
)

for chunk in stream:
    content = chunk.choices[0].delta.content
    if content:
        print(content, end="", flush=True)
```

### Guided Generation

```python
from chimera import Engine, SamplingParams
import json

engine = Engine(model_path="meta-llama/Llama-3.1-8B-Instruct")

# JSON schema
schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "city": {"type": "string"},
    },
    "required": ["name", "age", "city"],
}

response = engine.generate(
    prompt="Generate a person profile:",
    sampling_params=SamplingParams(
        guided_json=schema,
        max_new_tokens=200,
    ),
)

profile = json.loads(response.text)
print(profile)
```

---

**Last Updated**: March 29, 2026
**Version**: Chimera v1.0
