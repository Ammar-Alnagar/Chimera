#pragma once

#ifdef __RTRITONCC__
#include <rtriton_bf16.h>
#include <rtriton_fp16.h>
#include <rtriton_fp8.h>
#endif

namespace device {

inline constexpr float FP8_E4M3_MAX = 448.0f;

}  // namespace device
