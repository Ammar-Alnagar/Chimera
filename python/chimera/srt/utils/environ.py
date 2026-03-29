# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Environment variable utilities for Chimera - supports both CHIMERA_* and SGLANG_* prefixes."""

import os
from typing import Optional


def get_env_var(name: str, default: Optional[str] = None) -> Optional[str]:
    """
    Get an environment variable, supporting both CHIMERA_* and SGLANG_* prefixes.
    CHIMERA_* takes precedence over SGLANG_* if both are set.

    For example:
    - get_env_var("CHIMERA_TORCH_PROFILER_DIR") will check:
      1. CHIMERA_TORCH_PROFILER_DIR
      2. SGLANG_TORCH_PROFILER_DIR (for backward compatibility)
    """
    # Try CHIMERA_* prefix first
    value = os.getenv(name)
    if value is not None:
        return value

    # If name starts with CHIMERA_, also try SGLANG_ for backward compatibility
    if name.startswith("CHIMERA_"):
        sglang_name = "SGLANG_" + name[8:]  # Replace CHIMERA_ with SGLANG_
        value = os.getenv(sglang_name)
        if value is not None:
            return value

    return default


def get_bool_env_var(name: str, default: str = "false") -> bool:
    """
    Get a boolean environment variable, supporting both CHIMERA_* and SGLANG_* prefixes.
    """
    value = get_env_var(name, default)
    value = value.lower()

    truthy_values = ("true", "1")
    falsy_values = ("false", "0")

    if value in truthy_values:
        return True
    return False


def get_int_env_var(name: str, default: int = 0) -> int:
    """
    Get an integer environment variable, supporting both CHIMERA_* and SGLANG_* prefixes.
    """
    value = get_env_var(name)
    if value is None or not value.strip():
        return default
    try:
        return int(value)
    except ValueError:
        return default


def get_float_env_var(name: str, default: float = 0.0) -> float:
    """
    Get a float environment variable, supporting both CHIMERA_* and SGLANG_* prefixes.
    """
    value = get_env_var(name)
    if value is None or not value.strip():
        return default
    try:
        return float(value)
    except ValueError:
        return default
