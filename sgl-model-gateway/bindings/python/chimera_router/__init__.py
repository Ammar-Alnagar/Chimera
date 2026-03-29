# Chimera Router - backward-compatible alias for SGLang Router
# This module provides a renamed interface to SGLang Router for internal use
# while maintaining full backward compatibility with sglang_router imports

from chimera_router import *
from chimera_router import (
    Router,
    launch_router,
    launch_server,
    router_args,
    version,
    __version__,
)

__all__ = [
    "Router",
    "launch_router",
    "launch_server",
    "router_args",
    "version",
    "__version__",
]
