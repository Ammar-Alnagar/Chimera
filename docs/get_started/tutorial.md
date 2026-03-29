# Getting Started with Chimera

## Welcome to Chimera!

This tutorial will guide you through your first steps with Chimera, from installation to running your first LLM inference.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Your First Inference](#your-first-inference)
- [Understanding the Basics](#understanding-the-basics)
- [Next Steps](#next-steps)

---

## Prerequisites

### Hardware Requirements

**Minimum:**
- NVIDIA GPU with 16GB VRAM (e.g., RTX 3090, A10)
- 32GB system RAM
- 100GB free disk space

**Recommended:**
- NVIDIA H100/A100 with 80GB VRAM
- 64GB+ system RAM
- 500GB+ NVMe storage

**Check your GPU:**
```bash
nvidia-smi
```

### Software Requirements

- **OS:** Ubuntu 20.04+ or similar Linux distribution
- **CUDA:** 12.6 or later
- **Python:** 3.10 or later
- **NVIDIA Driver:** 550+ (for Hopper/Blackwell)

**Check versions:**
```bash
python3 --version
nvcc --version
nvidia-smi --query-gpu=driver_version --format=csv
```

---

## Installation

### Option 1: Quick Install (Recommended for beginners)

```bash
# Create virtual environment
python3 -m venv chimera-env
source chimera-env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install Chimera
pip install sglang

# Verify installation
python -c "import sglang; print(f'Chimera version: {sglang.__version__}')"
```

### Option 2: Docker (Easiest, most reproducible)

```bash
# Pull pre-built image
docker pull lmsysorg/sglang:latest

# Or build from source
git clone https://github.com/sgl-project/sglang.git
cd sglang
docker build -t chimera:latest -f docker/Dockerfile.chimera .
```

### Option 3: From Source (For developers)

```bash
# Clone repository
git clone https://github.com/sgl-project/sglang.git
cd sglang

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install in development mode
pip install -e .

# Build kernels
cd sgl-kernel
pip install scikit-build-core
python -m pip install .
```

### Verify Installation

```python
import torch
import sglang as sgl

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
print(f"Chimera version: {sgl.__version__}")
```

Expected output:
```
PyTorch version: 2.9.1+cu128
CUDA available: True
CUDA version: 12.8
GPU: NVIDIA H100
Chimera version: 0.4.6
```

---

## Quick Start

### Method 1: Python API (Simplest)

```python
from sglang import Engine, SamplingParams

# Initialize engine (downloads model on first run)
engine = Engine(
    model_path="meta-llama/Llama-3.1-8B-Instruct",
    mem_fraction_static=0.9,
)

# Generate text
response = engine.generate(
    prompt="What is the capital of France?",
    sampling_params=SamplingParams(
        temperature=0.7,
        max_new_tokens=100,
    ),
)

print(response.text)
```

### Method 2: Command Line Server

```bash
# Start server
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --host 0.0.0.0 \
  --port 30000

# In another terminal, send request
curl http://localhost:30000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "What is the capital of France?",
    "sampling_params": {"max_new_tokens": 100}
  }'
```

### Method 3: OpenAI-Compatible API

```python
from openai import OpenAI

# Connect to local server
client = OpenAI(
    base_url="http://localhost:3000/v1",
    api_key="EMPTY"  # Not required unless configured
)

# Chat completion
response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[
        {"role": "user", "content": "Hello! How are you?"}
    ],
    temperature=0.7,
    max_tokens=100,
)

print(response.choices[0].message.content)
```

---

## Your First Inference

### Tutorial 1: Simple Q&A

Let's run a complete example step by step.

**Step 1: Create a Python script**

Create `first_inference.py`:

```python
#!/usr/bin/env python3
"""
Your first Chimera inference!
This script demonstrates basic text generation.
"""

from sglang import Engine, SamplingParams

def main():
    print("🚀 Initializing Chimera Engine...")
    
    # Initialize the engine
    # This will download the model on first run (~15GB for 8B model)
    engine = Engine(
        model_path="meta-llama/Llama-3.1-8B-Instruct",
        mem_fraction_static=0.85,  # Use 85% of GPU memory for model
    )
    
    print("✅ Engine ready!")
    print("\n📝 Generating response...\n")
    
    # Define sampling parameters
    params = SamplingParams(
        temperature=0.7,      # Higher = more creative
        top_p=0.95,           # Nucleus sampling
        max_new_tokens=200,   # Maximum tokens to generate
    )
    
    # Generate response
    prompt = "Explain what artificial intelligence is in simple terms."
    
    response = engine.generate(
        prompt=prompt,
        sampling_params=params,
    )
    
    print(f"❓ Question: {prompt}")
    print(f"\n🤖 Answer: {response.text}")
    print(f"\n📊 Statistics:")
    print(f"   Prompt tokens: {response.prompt_tokens}")
    print(f"   Completion tokens: {response.completion_tokens}")
    print(f"   Total tokens: {response.prompt_tokens + response.completion_tokens}")

if __name__ == "__main__":
    main()
```

**Step 2: Run the script**

```bash
python first_inference.py
```

**Expected output:**
```
🚀 Initializing Chimera Engine...
[INFO] Loading model...
[INFO] Model loaded successfully
✅ Engine ready!

📝 Generating response...

❓ Question: Explain what artificial intelligence is in simple terms.

🤖 Answer: Artificial Intelligence (AI) refers to the simulation of human 
intelligence in machines that are programmed to think and learn like humans...

📊 Statistics:
   Prompt tokens: 12
   Completion tokens: 85
   Total tokens: 97
```

### Tutorial 2: Batch Processing

Process multiple prompts efficiently.

Create `batch_inference.py`:

```python
#!/usr/bin/env python3
"""
Batch inference example.
Process multiple prompts in parallel for better throughput.
"""

from sglang import Engine, SamplingParams

def main():
    print("🚀 Initializing Chimera Engine for batch processing...")
    
    engine = Engine(
        model_path="meta-llama/Llama-3.1-8B-Instruct",
        mem_fraction_static=0.85,
    )
    
    # Define multiple questions
    questions = [
        "What is Python used for?",
        "What is Rust known for?",
        "What is Go's main advantage?",
        "What is JavaScript mainly used for?",
        "What is C++ best suited for?",
    ]
    
    print(f"\n📝 Processing {len(questions)} questions in batch...\n")
    
    # Process all questions in one batch
    responses = engine.generate_batch(
        prompts=questions,
        sampling_params=SamplingParams(
            temperature=0.5,
            max_new_tokens=100,
        ),
    )
    
    # Display results
    for i, (question, response) in enumerate(zip(questions, responses), 1):
        print(f"{i}. ❓ {question}")
        print(f"   🤖 {response.text[:200]}...\n")
    
    print("✅ Batch processing complete!")

if __name__ == "__main__":
    main()
```

Run it:
```bash
python batch_inference.py
```

### Tutorial 3: Streaming Response

Get responses token by token as they're generated.

Create `streaming_inference.py`:

```python
#!/usr/bin/env python3
"""
Streaming inference example.
Get responses in real-time as they're generated.
"""

from openai import OpenAI

def main():
    # Connect to server (make sure server is running first!)
    client = OpenAI(
        base_url="http://localhost:3000/v1",
        api_key="EMPTY",
    )
    
    print("📝 Ask a question (streaming):\n")
    
    # Create streaming request
    stream = client.chat.completions.create(
        model="meta-llama/Llama-3.1-8B-Instruct",
        messages=[
            {"role": "user", "content": "Write a short poem about coding."}
        ],
        temperature=0.8,
        max_tokens=200,
        stream=True,  # Enable streaming
    )
    
    # Print tokens as they arrive
    print("🤖 ", end="", flush=True)
    for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    
    print("\n\n✅ Streaming complete!")

if __name__ == "__main__":
    main()
```

First, start the server in one terminal:
```bash
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --port 30000
```

Then run the streaming script in another terminal:
```bash
python streaming_inference.py
```

---

## Understanding the Basics

### Key Concepts

#### 1. Engine vs Runtime

```python
# Engine: Synchronous, embedded inference
from sglang import Engine
engine = Engine(model_path="...")
response = engine.generate("...")

# Runtime: Asynchronous, client-server inference
from sglang import Runtime
runtime = Runtime(server_url="http://localhost:30000")
response = await runtime.generate("...")
```

#### 2. Sampling Parameters

Control how text is generated:

```python
from sglang import SamplingParams

params = SamplingParams(
    # Temperature: 0 = deterministic, 1 = creative
    temperature=0.7,
    
    # Top-p: Nucleus sampling (0.95 = top 95% probability mass)
    top_p=0.95,
    
    # Top-k: Sample from top k tokens (-1 = disabled)
    top_k=-1,
    
    # Max tokens to generate
    max_new_tokens=256,
    
    # Stop sequences
    stop=["\n\n", "###"],
    
    # Repetition penalty (1 = no penalty, >1 = less repetition)
    repetition_penalty=1.1,
)
```

#### 3. Memory Management

```python
engine = Engine(
    model_path="...",
    
    # Fraction of GPU memory for model weights
    # Higher = more memory for model, less for KV cache
    mem_fraction_static=0.9,
    
    # Tensor parallelism (for multi-GPU)
    tp_size=1,
)
```

### Architecture Overview

```
┌─────────────────────────────────────────────────┐
│              Your Application                    │
│              (Python/HTTP Client)                │
└───────────────────┬─────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────┐
│              Chimera Engine                      │
│  ┌───────────┐  ┌───────────┐  ┌─────────────┐ │
│  │ Scheduler │  │  Memory   │  │   Batch     │ │
│  │           │  │  Manager  │  │  Executor   │ │
│  └───────────┘  └───────────┘  └─────────────┘ │
└───────────────────┬─────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────┐
│              Kernel Layer                        │
│  ┌───────────┐  ┌───────────┐  ┌─────────────┐ │
│  │  CuteDSL  │  │ CUTLASS   │  │  FlashInfer │ │
│  └───────────┘  └───────────┘  └─────────────┘ │
└───────────────────┬─────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────┐
│              NVIDIA GPU                          │
│         (Tensor Cores + VRAM)                    │
└─────────────────────────────────────────────────┘
```

### Common Workflows

#### Workflow 1: Chat Application

```python
from sglang import Engine, SamplingParams

engine = Engine(model_path="meta-llama/Llama-3.1-8B-Instruct")

# Conversation history
messages = [
    {"role": "system", "content": "You are a helpful coding assistant."},
    {"role": "user", "content": "How do I reverse a string in Python?"},
]

# Convert to prompt format
prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])

response = engine.generate(
    prompt=prompt,
    sampling_params=SamplingParams(max_new_tokens=200),
)

print(response.text)
```

#### Workflow 2: Document Summarization

```python
document = """
[Long document text here...]
"""

prompt = f"""
Please summarize the following document in 3-5 sentences:

{document}

Summary:
"""

response = engine.generate(
    prompt=prompt,
    sampling_params=SamplingParams(
        temperature=0.3,  # Lower for more factual
        max_new_tokens=150,
    ),
)

print(response.text)
```

#### Workflow 3: Code Generation

```python
prompt = """
Write a Python function that:
1. Takes a list of numbers
2. Returns the sum of all even numbers

Function:
"""

response = engine.generate(
    prompt=prompt,
    sampling_params=SamplingParams(
        temperature=0.1,  # Low for code
        max_new_tokens=300,
        stop=["\n\n"],
    ),
)

print(response.text)
```

---

## Next Steps

### Learn More

1. **API Reference**: See `docs/references/api_reference.md` for complete API documentation

2. **Performance Tuning**: See `docs/developer_guide/performance_tuning.md` for optimization tips

3. **CuteDSL Guide**: See `docs/developer_guide/cutedsl_integration.md` for kernel development

4. **Troubleshooting**: See `docs/references/troubleshooting.md` for common issues

### Try These Examples

```bash
# Examples directory
cd examples/

# Chatbot example
python chatbot.py

# RAG example
python rag_example.py

# Function calling
python function_calling.py
```

### Join the Community

- **GitHub**: https://github.com/sgl-project/sglang
- **Discord**: https://discord.gg/sglang
- **Documentation**: https://docs.sglang.io

### What's Next?

1. ✅ You've installed Chimera
2. ✅ You've run your first inference
3. ✅ You understand the basics

**Next, try:**
- [ ] Deploy with Docker
- [ ] Set up multi-GPU inference
- [ ] Enable quantization (FP8/INT8)
- [ ] Build a chat application
- [ ] Integrate with your existing codebase

---

## Quick Reference Card

### Installation
```bash
pip install sglang
```

### Basic Usage
```python
from sglang import Engine, SamplingParams
engine = Engine(model_path="meta-llama/Llama-3.1-8B-Instruct")
response = engine.generate("Hello!", SamplingParams(max_new_tokens=100))
```

### Start Server
```bash
python -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct --port 30000
```

### Send Request
```bash
curl http://localhost:30000/generate -d '{"text": "Hello!", "sampling_params": {"max_new_tokens": 100}}'
```

### Common Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `temperature` | 0.7 | Creativity (0-1) |
| `max_new_tokens` | 128 | Max output length |
| `top_p` | 0.95 | Nucleus sampling |
| `mem_fraction_static` | 0.9 | Memory for model |

---

**Congratulations!** You're now ready to use Chimera for your LLM inference needs! 🎉

**Last Updated**: March 29, 2026
**Version**: Chimera v1.0
