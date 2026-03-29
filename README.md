# Chimera

Chimera is a high-performance LLM serving stack spun off from SGLang, with a kernel strategy centered on **CuteDSL** and **CUTLASS 4.x**.

This repository keeps SGLang's serving/runtime strengths while prioritizing newer NVIDIA kernel paths for:
- MLA decode
- FP8 blockwise GEMM
- Expert-specialized grouped GEMM

## Architecture Overview

```mermaid
flowchart TB
    subgraph "Application Layer"
        A1[Python Application]
        A2[Frontend DSL]
        A3[OpenAI API]
    end

    subgraph "Chimera Runtime"
        B1[Request Scheduler]
        B2[Memory Manager]
        B3[Batch Executor]
    end

    subgraph "Kernel Layer"
        C1[CuteDSL Kernels]
        C2[CUTLASS 4.x]
        C3[Fallback Kernels]
    end

    subgraph "Hardware"
        D1[NVIDIA GPU<br/>Hopper/Blackwell]
        D2[VRAM]
        D3[Tensor Cores]
    end

    A1 --> B1
    A2 --> B1
    A3 --> B1
    B1 --> B2
    B2 --> B3
    B3 --> C1
    B3 --> C2
    B3 --> C3
    C1 --> D1
    C2 --> D1
    C3 --> D1
    D1 --> D2
    D1 --> D3

    style C1 fill:#4CAF50,color:#fff
    style C2 fill:#4CAF50,color:#fff
    style C3 fill:#FF9800,color:#fff
```

## Why Chimera

Chimera focuses on one goal: predictable, production-grade throughput on modern NVIDIA architectures by tightening the integration between Python-side launch logic and CUTLASS/CuteDSL kernel implementations.

Key directions:
- CUTLASS 4.x-first kernel development
- CuteDSL launch path integration for new kernels
- Safe runtime fallbacks to existing `torch.ops.sgl_kernel.*` operators
- Incremental migration without breaking caller APIs

## Kernel Execution Flow

```mermaid
sequenceDiagram
    participant App as Application
    participant Runtime as Chimera Runtime
    participant Wrapper as Kernel Wrapper
    participant CuteDSL as CuteDSL/CUTLASS
    participant Fallback as Fallback Path

    App->>Runtime: Generate/Prefill Request
    Runtime->>Wrapper: Invoke Kernel Operation
    
    alt CuteDSL Available
        Wrapper->>CuteDSL: Execute cutedsl_* kernel
        CuteDSL-->>Wrapper: Result Tensor
        Wrapper-->>Runtime: Return Result
    else Fallback Required
        Wrapper->>Fallback: Execute torch.ops.sgl_kernel
        Fallback-->>Wrapper: Result Tensor
        Wrapper-->>Runtime: Return Result
    end
    
    Runtime-->>App: Output Response
    
    Note over CuteDSL,Fallback: Transparent to user<br/>Same API signature
```

## System Deployment Architecture

```mermaid
flowchart LR
    subgraph "Client Layer"
        C1[REST API Clients]
        C2[SDK Clients]
        C3[Web UI]
    end

    subgraph "Load Balancer"
        LB[Chimera Router<br/>Envoy/Nginx]
    end

    subgraph "Serving Cluster"
        subgraph "Node 1"
            S1[Chimera Server<br/>GPU 0-3]
        end
        subgraph "Node 2"
            S2[Chimera Server<br/>GPU 4-7]
        end
        subgraph "Node N"
            S3[...]
        end
    end

    subgraph "Storage"
        M1[Model Cache<br/>HF/Mooncake]
        M2[KV Cache<br/>Shared Memory]
    end

    C1 --> LB
    C2 --> LB
    C3 --> LB
    LB --> S1
    LB --> S2
    LB --> S3
    S1 <--> M1
    S2 <--> M1
    S3 <--> M1
    S1 <--> M2
    S2 <--> M2
    S3 <--> M2

    style LB fill:#2196F3,color:#fff
    style S1 fill:#4CAF50,color:#fff
    style S2 fill:#4CAF50,color:#fff
```

## Repository Layout

- `python/sglang/` - runtime, server, scheduling, and model integration layers
- `sgl-kernel/` - CUDA/CUTLASS kernels and Python wrappers
- `sgl-model-gateway/` - gateway components and bindings
- `benchmark/` - performance and behavior benchmarks
- `docs/` - project documentation

## Kernel Strategy

Chimera uses a two-path model while kernels are being migrated:

```mermaid
flowchart LR
    subgraph "Python Wrapper"
        A[Kernel API Call]
        B{CuteDSL<br/>Available?}
        C[cutedsl_*<br/>Execution]
        D[torch.ops.sgl_kernel<br/>Fallback]
        E[Return Result]
    end

    A --> B
    B -->|Yes| C
    B -->|No| D
    C --> E
    D --> E

    style C fill:#4CAF50,color:#fff
    style D fill:#FF9800,color:#fff
```

1. **CuteDSL/CUTLASS 4.x path (preferred)**:
   - Python wrapper calls `sgl_kernel.cutedsl_*` entrypoints.
   - Optimized for Hopper/Blackwell architectures
   - Leverages latest CUTLASS 4.x features

2. **Stable ops fallback (always available)**:
   - Wrapper falls back to `torch.ops.sgl_kernel.*` when CuteDSL path is unavailable.
   - Ensures backward compatibility
   - Provides safe migration path

This keeps runtime behavior stable during bring-up and avoids hard failures from partially implemented kernels.

## Build Notes

`sgl-kernel/CMakeLists.txt` resolves Python once and reuses `Python_EXECUTABLE` consistently. This avoids interpreter drift between configuration steps and Python-based discovery logic.

Core dependencies include:
- CUDA Toolkit
- PyTorch
- CUTLASS (fetched in `sgl-kernel/CMakeLists.txt`)
- FlashInfer / Triton / Flash-Attention / DeepGEMM (as configured in `sgl-kernel`)

## Quick Start

1. Create and activate your Python environment.
2. Install project dependencies.
3. Build/install `sgl-kernel` and runtime components.
4. Run tests or benchmark scripts under `sgl-kernel/tests` and `sgl-kernel/benchmark`.

Example workflow:

```bash
# from repo root
pip install -U pip setuptools wheel
pip install -e .

# kernel-focused tests
pytest sgl-kernel/tests/test_fp8_blockwise_gemm.py
pytest sgl-kernel/tests/test_cutlass_mla.py
pytest sgl-kernel/tests/test_es_fp8_blockwise_moe.py
```

## Current Focus Areas

- Complete CuteDSL implementations for:
  - MLA decode
  - FP8 blockwise scaled MM
  - Expert-specialized grouped MM
- Expand architecture-specific tuning for Hopper and Blackwell
- Preserve API compatibility for existing SGLang-integrated call sites

## Compatibility Contract

For kernel wrapper APIs in `sgl-kernel/python/sgl_kernel/`:
- In-place kernels must still return the output tensor when a return value is expected by callers.
- Wrappers should not return `None` for tensor-producing paths.
- Fallback behavior must remain correct under partial kernel migration.

## Contributing

Contributions are welcome, especially in:
- CUTLASS 4.x kernel implementations
- CuteDSL launch and scheduling logic
- correctness/performance test coverage
- architecture-specific tuning and profiling

When submitting changes:
- Include correctness validation (`pytest` for touched kernels)
- Include benchmark deltas when performance behavior changes
- Keep fallback paths intact until new kernel paths are fully production-ready

## Acknowledgements

Chimera is based on SGLang and builds on work across the SGLang and CUDA kernel ecosystem.

## Deployment

### Quick Deploy with Docker

Chimera provides production-ready Docker images for easy deployment:

```mermaid
flowchart TD
    A[Pull/Build Image] --> B[Configure Environment]
    B --> C{Deployment Type}
    C -->|Single GPU| D[docker run]
    C -->|Multi-GPU| E[Docker Compose]
    C -->|Kubernetes| F[k8s manifests]
    D --> G[Server Running]
    E --> G
    F --> G
    G --> H[Health Check]
    H --> I[Ready for Traffic]

    style A fill:#2196F3,color:#fff
    style G fill:#4CAF50,color:#fff
    style I fill:#4CAF50,color:#fff
```

#### Option 1: Docker Run (Single GPU)

```bash
# Build the image
docker build -t chimera:latest -f docker/Dockerfile.chimera .

# Run with a model
docker run --gpus all -p 30000:30000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e HF_TOKEN=your_token \
  chimera:latest \
  --model-path meta-llama/Llama-3.1-8B-Instruct
```

#### Option 2: Docker Compose (Recommended)

```bash
# Start with default configuration
HF_TOKEN=your_token \
  docker compose -f docker/compose.chimera.yaml up -d

# Scale with tensor parallelism (2 GPUs)
TP_SIZE=2 GPU_COUNT=2 \
  docker compose -f docker/compose.chimera.yaml up -d
```

#### Option 3: Kubernetes

```bash
# Deploy single-node service
kubectl apply -f docker/k8s-chimera-service.yaml

# Deploy distributed tensor-parallel service
kubectl apply -f docker/k8s-chimera-distributed-sts.yaml
```

### Container Deployment Flow

```mermaid
sequenceDiagram
    participant User as User
    participant Docker as Docker Daemon
    participant Chimera as Chimera Container
    participant GPU as NVIDIA GPU
    participant HF as HuggingFace

    User->>Docker: docker run/build
    Docker->>Chimera: Create Container
    Docker->>GPU: Allocate GPU
    Chimera->>HF: Download Model (if needed)
    HF-->>Chimera: Model Weights
    Chimera->>Chimera: Initialize Runtime
    Chimera->>Chimera: Load Model to VRAM
    Chimera->>Chimera: Start HTTP Server
    Chimera-->>User: Server Ready (port 30000)
    User->>Chimera: Send Inference Request
    Chimera->>GPU: Execute Kernels
    GPU-->>Chimera: Results
    Chimera-->>User: Response JSON
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MODEL_PATH` | Model path or HF model ID | `meta-llama/Llama-3.1-8B-Instruct` |
| `PORT` | Server port | `30000` |
| `TP_SIZE` | Tensor parallelism | `1` |
| `MEM_FRACTION` | Memory fraction for model | `0.9` |
| `HF_TOKEN` | HuggingFace API token | - |
| `CUDA_VISIBLE_DEVICES` | GPU device selection | `0` |

### Health Check

The server exposes a health endpoint at `/health`:

```bash
curl http://localhost:30000/health
```

### Monitoring

Chimera exposes Prometheus metrics at `/metrics` for monitoring and observability.

For more deployment options, see:
- [Kubernetes Deployment Guide](docker/k8s-chimera-service.yaml)
- [Docker Compose Configuration](docker/compose.chimera.yaml)
- [Environment Variables Reference](docs/references/environment_variables.md)
