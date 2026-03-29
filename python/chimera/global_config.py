# Chimera - backward-compatible alias for SGLang
# This module provides a renamed interface to SGLang for internal use
# while maintaining full backward compatibility with sglang imports

from sglang.global_config import global_config
from sglang.version import __version__

# Re-export all module-level attributes from sglang
import sglang

# Module-level re-exports for direct access
__all__ = sglang.__all__ if hasattr(sglang, "__all__") else []
