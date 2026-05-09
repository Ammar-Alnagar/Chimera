# Adapted from https://github.com/vllm-project/vllm/blob/v0.6.4.post1/vllm/distributed/device_communicators/rtriton_wrapper.py

"""This file is a pure Python wrapper for the rtritonrt library.
It avoids the need to compile a separate shared library, and is
convenient for use when we just need to call a few functions.
"""

import ctypes
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# this line makes it possible to directly load `librtritonrt.so` using `ctypes`
import torch  # noqa

logger = logging.getLogger(__name__)

# === export types and functions from rtritonrt to Python ===
# for the original rtritonrt definition, please check
# https://docs.nvidia.com/rtriton/rtriton-runtime-api/index.html

rtritonError_t = ctypes.c_int
rtritonMemcpyKind = ctypes.c_int


class rtritonIpcMemHandle_t(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_byte * 128)]


@dataclass
class Function:
    name: str
    restype: Any
    argtypes: List[Any]


def find_loaded_library(lib_name) -> Optional[str]:
    """
    According to according to https://man7.org/linux/man-pages/man5/proc_pid_maps.5.html,
    the file `/proc/self/maps` contains the memory maps of the process, which includes the
    shared libraries loaded by the process. We can use this file to find the path of the
    a loaded library.
    """  # noqa
    found = False
    with open("/proc/self/maps") as f:
        for line in f:
            if lib_name in line:
                found = True
                break
    if not found:
        # the library is not loaded in the current process
        return None
    # if lib_name is librtritonrt, we need to match a line with:
    # address /path/to/librtritonrt-hash.so.11.0
    start = line.index("/")
    path = line[start:].strip()
    filename = path.split("/")[-1]
    assert filename.rpartition(".so")[0].startswith(
        lib_name
    ), f"Unexpected filename: {filename} for library {lib_name}"
    return path


class RtritonRTLibrary:
    exported_functions = [
        # ​rtritonError_t rtritonSetDevice ( int  device )
        Function("rtritonSetDevice", rtritonError_t, [ctypes.c_int]),
        # rtritonError_t 	rtritonDeviceSynchronize ( void )
        Function("rtritonDeviceSynchronize", rtritonError_t, []),
        # ​rtritonError_t rtritonDeviceReset ( void )
        Function("rtritonDeviceReset", rtritonError_t, []),
        # const char* 	rtritonGetErrorString ( rtritonError_t error )
        Function("rtritonGetErrorString", ctypes.c_char_p, [rtritonError_t]),
        # ​rtritonError_t 	rtritonMalloc ( void** devPtr, size_t size )
        Function(
            "rtritonMalloc",
            rtritonError_t,
            [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t],
        ),
        # ​rtritonError_t 	rtritonFree ( void* devPtr )
        Function("rtritonFree", rtritonError_t, [ctypes.c_void_p]),
        # ​rtritonError_t rtritonMemset ( void* devPtr, int  value, size_t count )
        Function(
            "rtritonMemset", rtritonError_t, [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
        ),
        # ​rtritonError_t rtritonMemcpy ( void* dst, const void* src, size_t count, rtritonMemcpyKind kind ) # noqa
        Function(
            "rtritonMemcpy",
            rtritonError_t,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, rtritonMemcpyKind],
        ),
        # rtritonError_t rtritonIpcGetMemHandle ( rtritonIpcMemHandle_t* handle, void* devPtr ) # noqa
        Function(
            "rtritonIpcGetMemHandle",
            rtritonError_t,
            [ctypes.POINTER(rtritonIpcMemHandle_t), ctypes.c_void_p],
        ),
        # ​rtritonError_t rtritonIpcOpenMemHandle ( void** devPtr, rtritonIpcMemHandle_t handle, unsigned int  flags ) # noqa
        Function(
            "rtritonIpcOpenMemHandle",
            rtritonError_t,
            [ctypes.POINTER(ctypes.c_void_p), rtritonIpcMemHandle_t, ctypes.c_uint],
        ),
    ]

    # class attribute to store the mapping from the path to the library
    # to avoid loading the same library multiple times
    path_to_library_cache: Dict[str, Any] = {}

    # class attribute to store the mapping from library path
    #  to the corresponding dictionary
    path_to_dict_mapping: Dict[str, Dict[str, Any]] = {}

    def __init__(self, so_file: Optional[str] = None):
        if so_file is None:
            so_file = find_loaded_library("librtritonrt")
            assert so_file is not None, "librtritonrt is not loaded in the current process"
        if so_file not in RtritonRTLibrary.path_to_library_cache:
            lib = ctypes.CDLL(so_file)
            RtritonRTLibrary.path_to_library_cache[so_file] = lib
        self.lib = RtritonRTLibrary.path_to_library_cache[so_file]

        if so_file not in RtritonRTLibrary.path_to_dict_mapping:
            _funcs = {}
            for func in RtritonRTLibrary.exported_functions:
                f = getattr(self.lib, func.name)
                f.restype = func.restype
                f.argtypes = func.argtypes
                _funcs[func.name] = f
            RtritonRTLibrary.path_to_dict_mapping[so_file] = _funcs
        self.funcs = RtritonRTLibrary.path_to_dict_mapping[so_file]

    def RTRITONRT_CHECK(self, result: rtritonError_t) -> None:
        if result != 0:
            error_str = self.rtritonGetErrorString(result)
            raise RuntimeError(f"RTRITONRT error: {error_str}")

    def rtritonGetErrorString(self, error: rtritonError_t) -> str:
        return self.funcs["rtritonGetErrorString"](error).decode("utf-8")

    def rtritonSetDevice(self, device: int) -> None:
        self.RTRITONRT_CHECK(self.funcs["rtritonSetDevice"](device))

    def rtritonDeviceSynchronize(self) -> None:
        self.RTRITONRT_CHECK(self.funcs["rtritonDeviceSynchronize"]())

    def rtritonDeviceReset(self) -> None:
        self.RTRITONRT_CHECK(self.funcs["rtritonDeviceReset"]())

    def rtritonMalloc(self, size: int) -> ctypes.c_void_p:
        devPtr = ctypes.c_void_p()
        self.RTRITONRT_CHECK(self.funcs["rtritonMalloc"](ctypes.byref(devPtr), size))
        return devPtr

    def rtritonFree(self, devPtr: ctypes.c_void_p) -> None:
        self.RTRITONRT_CHECK(self.funcs["rtritonFree"](devPtr))

    def rtritonMemset(self, devPtr: ctypes.c_void_p, value: int, count: int) -> None:
        self.RTRITONRT_CHECK(self.funcs["rtritonMemset"](devPtr, value, count))

    def rtritonMemcpy(
        self, dst: ctypes.c_void_p, src: ctypes.c_void_p, count: int
    ) -> None:
        rtritonMemcpyDefault = 4
        kind = rtritonMemcpyDefault
        self.RTRITONRT_CHECK(self.funcs["rtritonMemcpy"](dst, src, count, kind))

    def rtritonIpcGetMemHandle(self, devPtr: ctypes.c_void_p) -> rtritonIpcMemHandle_t:
        handle = rtritonIpcMemHandle_t()
        self.RTRITONRT_CHECK(
            self.funcs["rtritonIpcGetMemHandle"](ctypes.byref(handle), devPtr)
        )
        return handle

    def rtritonIpcOpenMemHandle(self, handle: rtritonIpcMemHandle_t) -> ctypes.c_void_p:
        rtritonIpcMemLazyEnablePeerAccess = 1
        devPtr = ctypes.c_void_p()
        self.RTRITONRT_CHECK(
            self.funcs["rtritonIpcOpenMemHandle"](
                ctypes.byref(devPtr), handle, rtritonIpcMemLazyEnablePeerAccess
            )
        )
        return devPtr
