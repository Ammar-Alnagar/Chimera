# Chimera Troubleshooting Guide & FAQ

## Overview

This guide helps you diagnose and resolve common issues when using Chimera. It includes frequently asked questions, error solutions, and debugging tips.

## Table of Contents

- [Installation Issues](#installation-issues)
- [Runtime Errors](#runtime-errors)
- [GPU Issues](#gpu-issues)
- [Performance Problems](#performance-problems)
- [Model Loading Issues](#model-loading-issues)
- [Network and API Issues](#network-and-api-issues)
- [Frequently Asked Questions](#frequently-asked-questions)
- [Debugging Tips](#debugging-tips)
- [Getting Help](#getting-help)

---

## Installation Issues

### Issue: `pip install` fails with CUDA error

**Symptoms:**
```
ERROR: Could not find a version that satisfies the requirement sgl-kernel
```

**Solutions:**

1. **Check CUDA version:**
```bash
nvcc --version
python -c "import torch; print(torch.version.cuda)"
```

2. **Match CUDA versions:**
```bash
# For CUDA 12.8/12.9
pip install sgl-kernel

# For CUDA 12.6
pip install https://github.com/sgl-project/whl/releases/download/v0.3.20/sgl_kernel-0.3.20+cu124-cp310-abi3-manylinux2014_x86_64.whl
```

3. **Upgrade pip:**
```bash
pip install --upgrade pip setuptools wheel
```

### Issue: Import error - `No module named 'sgl_kernel'`

**Symptoms:**
```python
ModuleNotFoundError: No module named 'sgl_kernel'
```

**Solutions:**

1. **Verify installation:**
```bash
pip list | grep sgl-kernel
```

2. **Reinstall:**
```bash
pip uninstall sgl-kernel
pip install sgl-kernel --force-reinstall
```

3. **Check Python path:**
```python
import sys
print(sys.path)
print(sys.executable)
```

### Issue: CMake build fails

**Symptoms:**
```
CMake Error: Could not find CMakePresets.json
```

**Solutions:**

1. **Update CMake:**
```bash
pip install --upgrade cmake
cmake --version  # Should be >= 3.31
```

2. **Install build dependencies:**
```bash
pip install scikit-build-core ninja
```

3. **Clean build:**
```bash
cd sgl-kernel
rm -rf build/
python -m pip install .
```

---

## Runtime Errors

### Issue: CUDA Out of Memory

**Symptoms:**
```
torch.cuda.OutOfMemoryError: CUDA out of memory.
```

**Solutions:**

1. **Reduce memory fraction:**
```bash
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --mem-fraction-static 0.85
```

2. **Enable quantization:**
```bash
--quantization fp8
--kv-cache-dtype fp8_e4m3
```

3. **Reduce batch size:**
```bash
--max-batch-size 64
--max-running-requests 32
```

4. **Use smaller model:**
```bash
# Use 8B instead of 70B
--model-path meta-llama/Llama-3.1-8B-Instruct
```

### Issue: Kernel launch failed

**Symptoms:**
```
RuntimeError: CUDA error: invalid device function
```

**Solutions:**

1. **Check GPU architecture:**
```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv
```

2. **Verify kernel compatibility:**
```python
# Check if CuteDSL is available
import sgl_kernel
print(dir(sgl_kernel))
```

3. **Rebuild for correct architecture:**
```bash
export TORCH_CUDA_ARCH_LIST="9.0"  # For Hopper
pip install -e . --no-build-isolation
```

### Issue: Segmentation fault

**Symptoms:**
```
Segmentation fault (core dumped)
```

**Solutions:**

1. **Check for incompatible libraries:**
```bash
ldd $(python -c "import torch; print(torch.__file__)")/lib/libtorch_cuda.so
```

2. **Update NCCL:**
```bash
pip install --upgrade nvidia-nccl-cu12
```

3. **Disable problematic features:**
```bash
--disable-cuda-graph
--disable-radix-cache
```

4. **Run with gdb for debugging:**
```bash
gdb -ex r --args python -m sglang.launch_server ...
```

### Issue: NCCL initialization failed

**Symptoms:**
```
RuntimeError: NCCL error - unhandled system error
```

**Solutions:**

1. **Check NCCL configuration:**
```bash
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
```

2. **Verify GPU visibility:**
```bash
nvidia-smi
export CUDA_VISIBLE_DEVICES=0,1,2,3
```

3. **Disable NVLink if problematic:**
```bash
export NCCL_P2P_DISABLE=1
```

4. **Check for IOMMU issues:**
```bash
# Check IOMMU status
dmesg | grep -i iommu

# Disable if needed (add to kernel params)
intel_iommu=off
```

---

## GPU Issues

### Issue: GPU not detected

**Symptoms:**
```
RuntimeError: Found no NVIDIA driver on your system
```

**Solutions:**

1. **Check driver:**
```bash
nvidia-smi
```

2. **Install/reinstall driver:**
```bash
# Ubuntu
sudo apt-get install nvidia-driver-550

# Or use runfile from nvidia.com
```

3. **Verify CUDA installation:**
```bash
nvcc --version
```

4. **Check device visibility:**
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
print(f"GPU name: {torch.cuda.get_device_name(0)}")
```

### Issue: Low GPU utilization

**Symptoms:**
```
nvidia-smi shows GPU utilization < 50%
```

**Solutions:**

1. **Increase batch size:**
```bash
--max-batch-size 256
```

2. **Enable continuous batching:**
```bash
--enable-continuous-batching
```

3. **Check for CPU bottleneck:**
```bash
htop
```

4. **Verify kernel selection:**
```python
import os
os.environ["SGL_KERNEL_DEBUG"] = "1"
```

### Issue: GPU temperature too high

**Symptoms:**
```
nvidia-smi shows temperature > 85°C
```

**Solutions:**

1. **Improve cooling:**
- Check airflow
- Clean dust filters
- Increase fan speed

2. **Reduce power limit:**
```bash
sudo nvidia-smi -pl 600  # Limit to 600W
```

3. **Reduce workload:**
```bash
--max-batch-size 128
```

---

## Performance Problems

### Issue: Slow inference latency

**Symptoms:**
```
Time per token > 100ms
```

**Solutions:**

1. **Enable FP8 quantization:**
```bash
--quantization fp8
```

2. **Use FlashInfer:**
```bash
--attention-backend flashinfer
```

3. **Reduce sequence length:**
```bash
--context-length 4096
```

4. **Check kernel fallback:**
```python
# Enable debug logging
import os
os.environ["SGL_KERNEL_LOG_LEVEL"] = "DEBUG"
```

### Issue: Poor throughput

**Symptoms:**
```
Tokens/sec much lower than expected
```

**Solutions:**

1. **Increase concurrent requests:**
```bash
--max-running-requests 128
```

2. **Enable prefix caching:**
```bash
--enable-prefix-caching
```

3. **Optimize scheduling:**
```bash
--schedule-policy shortest-job-first
```

4. **Profile to find bottleneck:**
```bash
python -m py-spy record -o profile.svg -- python -m sglang.launch_server ...
```

---

## Model Loading Issues

### Issue: Model download fails

**Symptoms:**
```
OSError: We couldn't connect to 'huggingface.co'
```

**Solutions:**

1. **Set HF token:**
```bash
export HF_TOKEN=your_token_here
```

2. **Use mirror:**
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

3. **Download manually:**
```bash
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct \
  --local-dir /path/to/model
```

4. **Use local path:**
```bash
--model-path /path/to/local/model
```

### Issue: Model architecture not supported

**Symptoms:**
```
ValueError: Model architecture not supported
```

**Solutions:**

1. **Check supported models:**
```bash
# See docs/supported_models/
```

2. **Use transformers fallback:**
```bash
--load-format transformers
```

3. **Update to latest version:**
```bash
pip install --upgrade sglang
```

### Issue: Tokenizer loading failed

**Symptoms:**
```
OSError: Couldn't load tokenizer
```

**Solutions:**

1. **Specify tokenizer path:**
```bash
--tokenizer-path /path/to/tokenizer
```

2. **Use slow tokenizer:**
```bash
--tokenizer-mode slow
```

3. **Trust remote code:**
```bash
--trust-remote-code
```

---

## Network and API Issues

### Issue: Connection refused

**Symptoms:**
```
ConnectionRefusedError: [Errno 111] Connection refused
```

**Solutions:**

1. **Check server status:**
```bash
curl http://localhost:30000/health
```

2. **Verify port:**
```bash
netstat -tlnp | grep 30000
```

3. **Check firewall:**
```bash
sudo ufw status
sudo ufw allow 30000
```

4. **Use correct host:**
```bash
--host 0.0.0.0  # Listen on all interfaces
```

### Issue: Request timeout

**Symptoms:**
```
requests.exceptions.Timeout
```

**Solutions:**

1. **Increase timeout:**
```python
import requests
response = requests.post(url, json=data, timeout=600)
```

2. **Reduce request size:**
```bash
--max-new-tokens 256
```

3. **Check server load:**
```bash
curl http://localhost:30000/get_server_info
```

### Issue: Rate limit exceeded

**Symptoms:**
```
HTTP 429: Too Many Requests
```

**Solutions:**

1. **Implement retry logic:**
```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential())
def generate(prompt):
    return requests.post(url, json={"text": prompt})
```

2. **Reduce request rate:**
```bash
--rate-limit 10  # requests per second
```

3. **Increase server capacity:**
```bash
--max-running-requests 256
```

---

## Frequently Asked Questions

### General Questions

#### Q: What GPUs are supported?

**A:** Chimera supports:
- **NVIDIA Hopper (H100, H200)** - Full support with FP8
- **NVIDIA Blackwell (B100, B200)** - Full support with FP8 microscaling
- **NVIDIA Ampere (A100, A10)** - Good support, no FP8
- **NVIDIA Ada (L40S, L4)** - Good support
- **AMD ROCm** - Limited support (see platform docs)

#### Q: What models are supported?

**A:** Chimera supports most popular LLM architectures:
- Llama 2/3/3.1
- Mistral/Mixtral
- Qwen/Qwen2
- Yi
- Gemma
- Phi
- And many more (see `docs/supported_models/`)

#### Q: How much VRAM do I need?

**A:** VRAM requirements vary by model:

| Model Size | FP16 | FP8 | INT4 |
|------------|------|-----|------|
| 7B | 14 GB | 7 GB | 4 GB |
| 13B | 26 GB | 13 GB | 7 GB |
| 34B | 68 GB | 34 GB | 17 GB |
| 70B | 140 GB | 70 GB | 35 GB |

Add 20-30% for KV cache and activations.

#### Q: Can I run multiple models simultaneously?

**A:** Yes, with sufficient VRAM:
```bash
# Start multiple servers on different ports
python -m sglang.launch_server --model-path model1 --port 30000
python -m sglang.launch_server --model-path model2 --port 30001
```

#### Q: Does Chimera support LoRA adapters?

**A:** Yes, LoRA is supported:
```bash
--lora-path /path/to/lora
--lora-scale 1.0
```

### Performance Questions

#### Q: How do I maximize throughput?

**A:** See `docs/developer_guide/performance_tuning.md` for detailed guide. Key tips:
- Use FP8 quantization
- Enable continuous batching
- Increase batch size
- Use appropriate scheduling policy

#### Q: How do I minimize latency?

**A:** For low latency:
- Use smaller models
- Enable FlashInfer
- Reduce concurrent requests
- Use greedy decoding (temperature=0)

#### Q: What's the difference between FP8 and FP16?

**A:** 
| Aspect | FP16 | FP8 |
|--------|------|-----|
| Precision | Higher | Lower |
| Memory | 2x | 1x |
| Speed | 1x | 2-3x |
| Quality | Baseline | Minimal loss |

### Development Questions

#### Q: How do I contribute kernels?

**A:** See `docs/developer_guide/cutedsl_integration.md`:
1. Implement kernel in C++/CuteDSL
2. Add Python wrapper
3. Write tests
4. Submit PR

#### Q: How do I debug kernel issues?

**A:** Enable debug mode:
```python
import os
os.environ["SGL_KERNEL_DEBUG"] = "1"
os.environ["SGL_KERNEL_LOG_LEVEL"] = "DEBUG"
```

#### Q: Can I use Chimera with vLLM/TGI?

**A:** Chimera is a standalone server. For multi-backend routing, use the Chimera router:
```bash
chimera-router --backends sglang,vllm
```

---

## Debugging Tips

### Enable Verbose Logging

```bash
# Server logging
--log-level debug

# Kernel logging
export SGL_KERNEL_LOG_LEVEL=DEBUG

# NCCL logging
export NCCL_DEBUG=INFO

# PyTorch logging
export TORCH_LOGS=+dynamo,recompiles
```

### Profile Execution

```bash
# Python profiling
py-spy record -o profile.svg -- python -m sglang.launch_server ...

# CUDA profiling
nsys profile --stats=true -o profile.qdrep python -m sglang.launch_server ...

# Torch profiling
export CHIMERA_TORCH_PROFILER_DIR=/tmp/profile
python -m sglang.launch_server ...
```

### Memory Debugging

```bash
# Monitor GPU memory
watch -n 1 nvidia-smi

# PyTorch memory summary
python -c "import torch; print(torch.cuda.memory_summary())"

# Enable memory debugging
export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.5
```

### Kernel Debugging

```python
# Check kernel availability
import sgl_kernel
print(dir(sgl_kernel))

# Test kernel directly
from sgl_kernel import cutedsl_fp8_blockwise_scaled_mm
# ... run test

# Check for fallback
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Network Debugging

```bash
# Check connectivity
curl -v http://localhost:30000/health

# Check open ports
netstat -tlnp | grep python

# Test with different clients
python -c "import requests; print(requests.get('http://localhost:30000/health').json())"
```

---

## Getting Help

### Resources

1. **Documentation:**
   - [API Reference](api_reference.md)
   - [Performance Tuning](performance_tuning.md)
   - [CuteDSL Guide](cutedsl_integration.md)

2. **GitHub:**
   - Issues: https://github.com/sgl-project/sglang/issues
   - Discussions: https://github.com/sgl-project/sglang/discussions

3. **Community:**
   - Discord: https://discord.gg/sglang
   - Slack: #chimera channel

### Reporting Issues

When reporting issues, include:

1. **System Information:**
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import sgl_kernel; print(f'SGL Kernel: {sgl_kernel.__version__}')"
nvcc --version
nvidia-smi
```

2. **Error Messages:**
- Full stack trace
- Server logs
- Kernel logs

3. **Reproduction Steps:**
- Model used
- Command line arguments
- Input that triggers the issue

4. **Expected vs Actual Behavior:**
- What you expected
- What actually happened

### Contact

For urgent issues or enterprise support:
- Email: support@chimera.ai (example)
- Enterprise Slack channel

---

## Quick Reference

### Common Commands

```bash
# Health check
curl http://localhost:30000/health

# Server info
curl http://localhost:30000/get_server_info

# Flush cache
curl -X POST http://localhost:30000/flush_cache

# Metrics
curl http://localhost:30000/metrics

# GPU status
nvidia-smi

# Process list
ps aux | grep sglang

# Port check
netstat -tlnp | grep 30000
```

### Environment Variables Quick Reference

```bash
# Debugging
export SGL_KERNEL_DEBUG=1
export SGL_KERNEL_LOG_LEVEL=DEBUG
export NCCL_DEBUG=INFO

# Performance
export SGL_KERNEL_FORCE_CUTEDSL=1
export SGL_KERNEL_USE_FLASHINFER=1

# Memory
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Profiling
export CHIMERA_TORCH_PROFILER_DIR=/tmp/profile
```

---

**Last Updated**: March 29, 2026
**Version**: Chimera v1.0
