from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import load_jit

if TYPE_CHECKING:
    import torch
    from tvm_ffi.module import Module


@lru_cache(maxsize=1)
def _jit_stream_wait_value_module() -> Module:
    return load_jit(
        "rtriton_wait_value",
        rtriton_files=["rtriton_wait_value.cuh"],
        rtriton_wrappers=[("stream_wait_value", "stream_wait_value")],
    )


def stream_wait_value(flag: torch.Tensor, value: int) -> None:
    module = _jit_stream_wait_value_module()
    module.stream_wait_value(flag, value)


class Event:
    def __init__(self) -> None:
        self.flag = torch.zeros(1, dtype=torch.int32, device="rtriton")

    def record(self, value: int = 1) -> None:
        self.flag[0] = value

    def wait(self, value: int = 1) -> None:
        stream_wait_value(self.flag, value)


def test_wait_before_record(event: Event | torch.rtriton.Event):
    stream_a = torch.rtriton.Stream()
    stream_b = torch.rtriton.Stream()

    with torch.rtriton.stream(stream_a):
        event.wait()

    stream_a.synchronize()

    with torch.rtriton.stream(stream_b):
        event.record()


def main():
    import threading
    import time

    block_thead = threading.Thread(
        target=test_wait_before_record, args=(Event(),), daemon=True
    )
    block_thead.start()

    non_block_thread = threading.Thread(
        target=test_wait_before_record, args=(torch.rtriton.Event(),)
    )
    non_block_thread.start()

    print("Checking if custom Event blocks the stream...", flush=True)
    for _ in range(5):
        print(f"{block_thead.is_alive()=}, {non_block_thread.is_alive()=}", flush=True)
        time.sleep(1)

    assert block_thead.is_alive(), "Custom Event did not block as expected"
    assert not non_block_thread.is_alive(), "torch.rtriton.Event should not block"
    print("=" * 40)
    print("Test completed successfully.")


if __name__ == "__main__":
    main()
