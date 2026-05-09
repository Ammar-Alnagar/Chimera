# Chimera TileLang Migration Summary

## Overview
This document summarizes the complete migration of the Chimera project from the deprecated `CuteDSL` and `CUTLASS 4.x` infrastructure to the new **TileLang** Pythonic DSL.

TileLang enables rapid iteration of high-performance GPU kernels directly in Python by compiling to optimized RTRITON via Apache TVM. This eliminates the heavy C++ build times previously required by CuteDSL/CUTLASS, while maintaining or exceeding performance on modern NVIDIA architectures (Hopper, Blackwell).

## Key Components Migrated

### 1. Kernel Implementations (`sgl-kernel/python/sgl_kernel/`)
All `cutedsl_*.py` wrapper files have been fully replaced with native TileLang JIT implementations:
- **FP8 Blockwise Scaled GEMM**: Replaced `cutedsl_gemm.py` with `tilelang_gemm.py` (`@tilelang.jit` implementation).
- **MLA Decode Attention**: Replaced `cutedsl_attention.py` with `tilelang_attention.py`.
- **Expert Specialization (MoE)**: Replaced `cutedsl_expert_specialization.py` with `tilelang_expert_specialization.py`.

*Note: Robust PyTorch fallbacks (`torch.ops.sgl_kernel.*`) remain intact to ensure graceful degradation if TileLang JIT compilation fails or on unsupported hardware.*

### 2. Runtime and MoE Runner (`python/sglang/srt/layers/moe/`)
- The internal `FLASHINFER_CUTEDSL` backend was officially migrated and renamed to `FLASHINFER_TILELANG`.
- Updated dispatch logic in `ep_moe/layer.py` to route FP4 and FP8 blockwise MoE operations to the new TileLang executor.
- Added `flashinfer_tilelang_moe.py` to bridge the FlashInfer execution framework with the new TileLang grouped GEMM primitives.

### 3. Build System (`pyproject.toml` and `CMakeLists.txt`)
- Removed `nvidia-cutlass-dsl` dependency from both `python/pyproject.toml` and `sgl-kernel/pyproject.toml`.
- Added `tilelang>=0.1.9` as the core required dependency for `sgl-kernel`.

*(Note: We still maintain `CMakeLists.txt` for compiling legacy C++ extensions like elementwise operations, Mamba kernels, FlashInfer integration, and custom AllReduce routines. However, new dense and grouped GEMM kernels will now exclusively use TileLang.)*

### 4. Documentation and Terminology
- Completely rewrote `README.md` to emphasize TileLang as the primary kernel execution path.
- Renamed and updated developer guides (`tilelang_integration.md` and `tilelang_visual_guide.md`).
- Scrubbed all remaining `CuteDSL` terminology from architecture diagrams, environment variables, and system configurations.

### 5. Backward Compatibility
To prevent breaking existing caller logic and configurations during the rollout:
- **Environment Variables**: Maintained deprecated mappings (e.g., `SGLANG_CUTEDSL_MOE_NVFP4_DISPATCH` resolves to the new `SGLANG_TILELANG_MOE_NVFP4_DISPATCH`).
- **Python Imports**: Maintained aliased imports in `sgl-kernel/python/sgl_kernel/__init__.py` (e.g., `from sgl_kernel.tilelang_attention import cutedsl_mla_decode`).

## Future Steps
- Continue porting remaining legacy CUTLASS 3.x kernels to TileLang.
- Integrate TileLang auto-tuning (`@tilelang.autotune`) for automatic block size selection across different GPU tiers.
