# Chimera Migration Guide

## Overview

SGLang has been internally renamed to **Chimera**. This migration is designed to be **fully backward-compatible** - all existing SGLang code will continue to work without modification.

## What Changed

### New Naming

- **Project name**: SGLang → Chimera (internal naming)
- **Package name**: `sglang` → `chimera` (alias provided)
- **Commands**: `sglang.*` → `chimera.*` (both work)
- **Environment variables**: `SGLANG_*` → `CHIMERA_*` (both work)

### Backward Compatibility

**All existing SGLang imports and commands continue to work:**

```python
# Old way (still works)
import sglang as sgl
from sglang import Engine

# New way (also works)
import chimera as chim
from chimera import Engine
```

```bash
# Old way (still works)
python -m sglang.launch_server

# New way (also works)
python -m chimera.launch_server
```

```bash
# Old way (still works)
export SGLANG_TORCH_PROFILER_DIR=/tmp/profile

# New way (also works, takes precedence)
export CHIMERA_TORCH_PROFILER_DIR=/tmp/profile
```

## Migration Path

### Phase 1: Dual Support (Current)

Both `sglang` and `chimera` names work interchangeably:
- Import either `sglang` or `chimera`
- Use either `SGLANG_*` or `CHIMERA_*` environment variables
- Run either `sglang` or `chimera` commands

**Recommendation**: Start using `chimera` in new code, but existing code requires no changes.

### Phase 2: Documentation Updates

Documentation now prefers `chimera` naming, but all examples work with both names.

### Phase 3: Future Deprecation (Not Planned)

A future version may deprecate `sglang` naming, but this will require:
- Extended deprecation period (6+ months)
- Clear migration warnings
- Continued backward compatibility during transition

## Using Chimera

### Installation

No changes needed - install as usual:

```bash
pip install sglang
```

The `chimera` package is included automatically.

### Basic Usage

```python
# Option 1: Use chimera (new)
import chimera as chim
from chimera import Engine

engine = Engine(model_path="meta-llama/Llama-3.1-8B-Instruct")
output = engine.generate("What is the capital of France?")
print(output)

# Option 2: Use sglang (still works)
import sglang as sgl
from sglang import Engine

engine = Engine(model_path="meta-llama/Llama-3.1-8B-Instruct")
output = engine.generate("What is the capital of France?")
print(output)
```

### Command Line

```bash
# Launch server with chimera
python -m chimera.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct

# Launch server with sglang (still works)
python -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct

# CLI command
chimera  # or: sglang
```

### Environment Variables

```bash
# Use CHIMERA_* (new, takes precedence)
export CHIMERA_TORCH_PROFILER_DIR=/tmp/profile
export CHIMENA_ENABLE_TORCH_COMPILE=1

# Or use SGLANG_* (still works)
export SGLANG_TORCH_PROFILER_DIR=/tmp/profile
export SGLANG_ENABLE_TORCH_COMPILE=1
```

**Note**: If both `CHIMERA_*` and `SGLANG_*` are set, `CHIMERA_*` takes precedence.

## Benchmark Scripts

Benchmark scripts now have `chimera` aliases:

```bash
# Run benchmarks with chimera
python benchmark/react/bench_chimera.py

# Or use sglang (still works via symlink)
python benchmark/react/bench_sglang.py
```

## API Reference

All APIs are identical between `sglang` and `chimera`:

| SGLang | Chimera | Status |
|--------|---------|--------|
| `import sglang` | `import chimera` | ✅ Both work |
| `sglang.Engine` | `chimera.Engine` | ✅ Both work |
| `sglang.Runtime` | `chimera.Runtime` | ✅ Both work |
| `sglang.gen()` | `chimera.gen()` | ✅ Both work |
| `python -m sglang.launch_server` | `python -m chimera.launch_server` | ✅ Both work |
| `SGLANG_*` env vars | `CHIMERA_*` env vars | ✅ Both work |

## Technical Details

### Package Structure

The `chimera` package re-exports all symbols from `sglang`:

```
python/
├── sglang/          # Original package
│   ├── __init__.py
│   ├── srt/
│   └── ...
└── chimera/         # Alias package
    ├── __init__.py  # Re-exports from sglang
    ├── srt/         # Re-exports from sglang.srt
    └── ...
```

### Environment Variable Handling

Environment variables support both prefixes with `CHIMERA_*` taking precedence:

```python
# Internal implementation
def getenv(name, default=None):
    # Try CHIMERA_* first
    value = os.getenv(name)
    if value is not None:
        return value
    
    # Fall back to SGLANG_* for compatibility
    if name.startswith("CHIMERA_"):
        sglang_name = "SGLANG_" + name[8:]
        return os.getenv(sglang_name, default)
    
    return default
```

## Frequently Asked Questions

### Q: Do I need to change my existing code?

**A:** No! All existing SGLang code continues to work without modification.

### Q: Should I migrate to `chimera` naming?

**A:** For new code, yes. For existing code, it's optional. Both work identically.

### Q: Will `sglang` naming be removed?

**A:** Not in the foreseeable future. Any deprecation would have a 6+ month transition period.

### Q: Can I mix `sglang` and `chimera` imports?

**A:** Yes! They reference the same underlying objects:

```python
import sglang as sgl
import chimera as chim

assert sgl.Engine is chim.Engine  # True
```

### Q: What about third-party packages that depend on sglang?

**A:** They continue to work unchanged. The `sglang` package name remains fully functional.

## Support

For issues or questions:
- GitHub Issues: https://github.com/sgl-project/sglang/issues
- Documentation: https://docs.sglang.io

---

**Last updated**: March 2026  
**Version**: Chimera migration v1.0
