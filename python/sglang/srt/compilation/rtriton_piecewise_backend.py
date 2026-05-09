# Adapted from https://github.com/vllm-project/vllm/blob/v0.10.0/vllm/compilation/rtriton_piecewise_backend.py

import dataclasses
import logging
from contextlib import ExitStack
from typing import Any, Callable, Optional
from unittest.mock import patch

import torch
import torch.fx as fx

from sglang.srt.compilation.compilation_config import CompilationConfig
from sglang.srt.compilation.compilation_counter import compilation_counter
from sglang.srt.compilation.piecewise_context_manager import (
    get_pcg_capture_stream,
    is_in_pcg_torch_compile,
)
from sglang.srt.compilation.weak_ref_tensor import weak_ref_tensors

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class ConcreteSizeEntry:
    runtime_shape: int
    need_to_compile: bool  # the size is in compile_sizes
    use_rtritongraph: bool  # the size is in rtritongraph_capture_sizes

    compiled: bool = False
    runnable: Callable = None  # type: ignore
    num_finished_warmup: int = 0
    rtritongraph: Optional[torch.rtriton.RTRITONGraph] = None
    output: Optional[Any] = None

    # for rtritongraph debugging, track the input addresses
    # during capture, and check if they are the same during replay
    input_addresses: Optional[list[int]] = None


class RTRITONPiecewiseBackend:

    def __init__(
        self,
        graph: fx.GraphModule,
        compile_config: CompilationConfig,
        inductor_config: dict[str, Any],
        graph_pool: Any,
        piecewise_compile_index: int,
        total_piecewise_compiles: int,
        sym_shape_indices: list[int],
        compiled_graph_for_general_shape: Callable,
        sglang_backend,
    ):
        """
        The backend for piecewise compilation.
        It mainly handles the compilation and rtritongraph capturing.

        We will compile `self.graph` once for the general shape,
        and then compile for different shapes specified in
        `compilation_config.compile_sizes`.

        Independently, we will capture rtritongraph for different shapes.

        If a shape needs both compilation and rtritongraph, we will
        compile it first, and then capture rtritongraph.
        """
        self.graph = graph
        self.inductor_config = inductor_config
        self.graph_pool = graph_pool
        self.piecewise_compile_index = piecewise_compile_index
        self.total_piecewise_compiles = total_piecewise_compiles
        self.sglang_backend = sglang_backend

        self.is_first_graph = piecewise_compile_index == 0
        self.is_last_graph = piecewise_compile_index == total_piecewise_compiles - 1

        self.compile_sizes: set[int] = set([])
        self.compile_config = compile_config
        self.rtritongraph_capture_sizes: set[int] = set(compile_config.get_capture_sizes())

        self.first_run_finished = False

        self.compiled_graph_for_general_shape = compiled_graph_for_general_shape  # noqa

        self.sym_shape_indices = sym_shape_indices

        # the entries for different shapes that we need to either
        # compile or capture rtritongraph
        self.concrete_size_entries: dict[int, ConcreteSizeEntry] = {}

        # to_be_compiled_sizes tracks the remaining sizes to compile,
        # and updates during the compilation process, so we need to copy it
        self.to_be_compiled_sizes: set[int] = self.compile_sizes.copy()
        for shape in self.compile_sizes.union(self.rtritongraph_capture_sizes):
            self.concrete_size_entries[shape] = ConcreteSizeEntry(
                runtime_shape=shape,
                need_to_compile=shape in self.compile_sizes,
                use_rtritongraph=shape in self.rtritongraph_capture_sizes,
            )

    def check_for_ending_compilation(self):
        if self.is_last_graph and not self.to_be_compiled_sizes:
            # no specific sizes to compile
            # save the hash of the inductor graph for the next run
            self.sglang_backend.compiler_manager.save_to_file()

    def __call__(self, *args) -> Any:
        if not self.first_run_finished:
            self.first_run_finished = True
            self.check_for_ending_compilation()
            return self.compiled_graph_for_general_shape(*args)

        if len(self.sym_shape_indices) == 0:
            return self.compiled_graph_for_general_shape(*args)

        runtime_shape = args[self.sym_shape_indices[0]]
        if runtime_shape not in self.concrete_size_entries:
            # we don't need to do anything for this shape
            return self.compiled_graph_for_general_shape(*args)

        entry = self.concrete_size_entries[runtime_shape]

        if entry.runnable is None:
            entry.runnable = self.compiled_graph_for_general_shape

        if entry.need_to_compile and not entry.compiled:
            entry.compiled = True
            self.to_be_compiled_sizes.remove(runtime_shape)
            # args are real arguments
            entry.runnable = self.sglang_backend.compiler_manager.compile(
                self.graph,
                args,
                self.inductor_config,
                graph_index=self.piecewise_compile_index,
                num_graphs=self.total_piecewise_compiles,
                runtime_shape=runtime_shape,
            )

            # finished compilations for all required shapes
            if self.is_last_graph and not self.to_be_compiled_sizes:
                self.check_for_ending_compilation()

        # Skip RTRITON graphs if this entry doesn't use them OR
        # if we're supposed to skip them globally
        # skip_rtriton_graphs = get_forward_context().skip_rtriton_graphs
        # if not entry.use_rtritongraph or skip_rtriton_graphs:
        #     return entry.runnable(*args)
        if is_in_pcg_torch_compile():
            return entry.runnable(*args)

        if entry.rtritongraph is None:
            if entry.num_finished_warmup < 1:  # noqa
                entry.num_finished_warmup += 1
                return entry.runnable(*args)

            if self.compile_config.get_enable_debug_mode():
                input_addresses = [
                    x.data_ptr() for x in args if isinstance(x, torch.Tensor)
                ]
                entry.input_addresses = input_addresses
            rtritongraph = torch.rtriton.RTRITONGraph()

            with ExitStack() as stack:
                if not self.is_first_graph:
                    # during every model forward, we will capture
                    # many pieces of rtritongraphs (roughly one per layer).
                    # running gc again and again across layers will
                    # make the rtritongraph capture very slow.
                    # therefore, we only run gc for the first graph,
                    # and disable gc for the rest of the graphs.
                    stack.enter_context(patch("gc.collect", lambda: None))
                    stack.enter_context(patch("torch.rtriton.empty_cache", lambda: None))
                # mind-exploding: carefully manage the reference and memory.
                stream = get_pcg_capture_stream()
                assert stream is not None, "PCG capture stream is not set"
                with torch.rtriton.graph(rtritongraph, pool=self.graph_pool, stream=stream):
                    # `output` is managed by pytorch's rtritongraph pool
                    output = entry.runnable(*args)
                    if self.is_last_graph:
                        # by converting it to weak ref,
                        # the original `output` will immediately be released
                        # to save memory. It is only safe to do this for
                        # the last graph, because the output of the last graph
                        # will not be used by any other rtriton graph.
                        output = weak_ref_tensors(output)

            # here we always use weak ref for the output
            # to save memory
            entry.output = weak_ref_tensors(output)
            entry.rtritongraph = rtritongraph

            compilation_counter.num_rtritongraph_captured += 1

            # important: we need to return the output, rather than
            # the weak ref of the output, so that pytorch can correctly
            # manage the memory during rtriton graph capture
            return output

        if self.compile_config.get_enable_debug_mode():
            # check if the input addresses are the same
            new_input_addresses = [
                x.data_ptr() for x in args if isinstance(x, torch.Tensor)
            ]
            assert new_input_addresses == entry.input_addresses, (
                "Input addresses for rtritongraphs are different during replay."
                f" Expected {entry.input_addresses}, got {new_input_addresses}"
            )
        entry.rtritongraph.replay()
        return entry.output
