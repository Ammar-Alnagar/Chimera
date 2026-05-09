/* Copyright 2025 SGLang Team. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#include "utils.h"

// clang-format off
#include "tilelang/tilelang.h"
#include "tilelang/gemm/collective/collective_builder.hpp"
#include "tilelang/epilogue/collective/collective_builder.hpp"
#include "tilelang/gemm/device/gemm_universal_adapter.h"
#include "tilelang/gemm/kernel/gemm_universal.hpp"
#include "tilelang/util/packed_stride.hpp"
// clang-format on

/**
 * Helper function for checking TILELANG errors
 */
#define TILELANG_CHECK(status)                                                       \
  {                                                                                 \
    tilelang::Status error = status;                                                 \
    TORCH_CHECK(error == tilelang::Status::kSuccess, tilelangGetStatusString(error)); \
  }

using namespace cute;

// Helper function for next power of 2
inline uint32_t next_pow_2(uint32_t x) {
  if (x == 0) return 1;
  x--;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  return x + 1;
}

#if defined(TILELANG_ARCH_MMA_SM100_SUPPORTED) || defined(TILELANG_ARCH_MMA_SM120_SUPPORTED) || \
    defined(TILELANG_ARCH_MMA_SM121_SUPPORTED)
// Config(half_t/bfloat16_t) for M <= 128
template <typename T>
struct KernelConfigM128 {
  using OutputType = T;
  using MmaTileShape = Shape<_128, _256, _256>;
  using ClusterShape = Shape<int, int, _1>;
  using EpilogueTile = Shape<_128, _64>;  // Avoid register spilling
  using EpilogueSchedule = tilelang::epilogue::TmaWarpSpecialized1Sm;
  using MainloopSchedule = tilelang::gemm::KernelTmaWarpSpecialized1SmNvf4Sm100;
  const static dim3 preferred_cluster;
  const static dim3 fallback_cluster;
};
template <typename T>
const dim3 KernelConfigM128<T>::preferred_cluster(1, 4, 1);
template <typename T>
const dim3 KernelConfigM128<T>::fallback_cluster(1, 2, 1);

// Config(half_t/bfloat16_t) for M <= 256
template <typename T>
struct KernelConfigM256 {
  using OutputType = T;
  using MmaTileShape = Shape<_256, _256, _256>;
  using ClusterShape = Shape<int, int, _1>;
  using EpilogueTile = Shape<_128, _64>;  // Avoid register spilling
  using EpilogueSchedule = tilelang::epilogue::TmaWarpSpecialized2Sm;
  using MainloopSchedule = tilelang::gemm::KernelTmaWarpSpecialized2SmNvf4Sm100;
  const static dim3 preferred_cluster;
  const static dim3 fallback_cluster;
};
template <typename T>
const dim3 KernelConfigM256<T>::preferred_cluster(2, 4, 1);
template <typename T>
const dim3 KernelConfigM256<T>::fallback_cluster(2, 1, 1);

// Default config(half_t/bfloat16_t) for M > 256
template <typename T>
struct KernelConfigDefault {
  using OutputType = T;
  using MmaTileShape = Shape<_256, _256, _256>;
  using ClusterShape = Shape<int, int, _1>;
  using EpilogueTile = Shape<_128, _64>;  // Avoid register spilling
  using EpilogueSchedule = tilelang::epilogue::TmaWarpSpecialized2Sm;
  using MainloopSchedule = tilelang::gemm::KernelTmaWarpSpecialized2SmNvf4Sm100;
  const static dim3 preferred_cluster;
  const static dim3 fallback_cluster;
};
template <typename T>
const dim3 KernelConfigDefault<T>::preferred_cluster(4, 4, 1);
template <typename T>
const dim3 KernelConfigDefault<T>::fallback_cluster(2, 1, 1);

struct KernelConfigFp32 {
  using OutputType = float;
  using MmaTileShape = Shape<_128, _128, _256>;
  using ClusterShape = Shape<int, int, _1>;
  using EpilogueTile = tilelang::epilogue::collective::EpilogueTileAuto;
  using EpilogueSchedule = tilelang::epilogue::TmaWarpSpecialized1Sm;
  using MainloopSchedule = tilelang::gemm::KernelTmaWarpSpecialized1SmNvf4Sm100;
  const static dim3 preferred_cluster;
  const static dim3 fallback_cluster;
};
const dim3 KernelConfigFp32::preferred_cluster = dim3(1, 4, 1);
const dim3 KernelConfigFp32::fallback_cluster = dim3(1, 2, 1);

// SM120 specific configurations
struct sm120_fp4_config_M256 {
  using ClusterShape = Shape<_1, _1, _1>;
  using MmaTileShape = Shape<_128, _128, _128>;
  using PerSmTileShape_MNK = Shape<_128, _128, _128>;
};

struct sm120_fp4_config_default {
  using ClusterShape = Shape<_1, _1, _1>;
  using MmaTileShape = Shape<_256, _128, _128>;
  using PerSmTileShape_MNK = Shape<_256, _128, _128>;
};

template <typename KernelConfig>
struct Fp4GemmSm100 {
  using Config = KernelConfig;  // For generating args
  using OutputType = typename KernelConfig::OutputType;
  // A matrix configuration
  using ElementA = tilelang::nv_float4_t<tilelang::float_e2m1_t>;
  using LayoutATag = tilelang::layout::RowMajor;
  static constexpr int AlignmentA = 32;

  // B matrix configuration
  using ElementB = tilelang::nv_float4_t<tilelang::float_e2m1_t>;
  using LayoutBTag = tilelang::layout::ColumnMajor;
  static constexpr int AlignmentB = 32;

  // C/D matrix configuration
  using ElementD = OutputType;
  using ElementC = OutputType;
  using LayoutCTag = tilelang::layout::RowMajor;
  using LayoutDTag = tilelang::layout::RowMajor;
  static constexpr int AlignmentD = 128 / tilelang::sizeof_bits<ElementD>::value;
  static constexpr int AlignmentC = 128 / tilelang::sizeof_bits<ElementC>::value;
  // Kernel functional config
  using ElementAccumulator = float;
  using ArchTag = tilelang::arch::Sm100;
  using OperatorClass = tilelang::arch::OpClassBlockScaledTensorOp;

  // Kernel Perf config
  using MmaTileShape = typename KernelConfig::MmaTileShape;
  using ClusterShape = typename KernelConfig::ClusterShape;
  using EpilogueTile = typename KernelConfig::EpilogueTile;
  using EpilogueSchedule = typename KernelConfig::EpilogueSchedule;
  using MainloopSchedule = typename KernelConfig::MainloopSchedule;

  using CollectiveEpilogue = typename tilelang::epilogue::collective::CollectiveBuilder<
      ArchTag,
      OperatorClass,
      MmaTileShape,
      ClusterShape,
      EpilogueTile,
      ElementAccumulator,
      ElementAccumulator,
      void,
      LayoutCTag,
      AlignmentC,
      ElementD,
      LayoutDTag,
      AlignmentD,
      EpilogueSchedule,
      tilelang::epilogue::fusion::LinearCombination<ElementD, float, void, float>>::CollectiveOp;

  using CollectiveMainloop = typename tilelang::gemm::collective::CollectiveBuilder<
      ArchTag,
      OperatorClass,
      ElementA,
      LayoutATag,
      AlignmentA,
      ElementB,
      LayoutBTag,
      AlignmentB,
      ElementAccumulator,
      MmaTileShape,
      ClusterShape,
      tilelang::gemm::collective::StageCountAutoCarveout<static_cast<int>(
          sizeof(typename CollectiveEpilogue::SharedStorage))>,
      MainloopSchedule>::CollectiveOp;

  using GemmKernel =
      tilelang::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, void>;
  using Gemm = tilelang::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using LayoutA = decltype(cute::make_layout(make_shape(0, 0, 0), StrideA{}));
  using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using LayoutB = decltype(cute::make_layout(make_shape(0, 0, 0), StrideB{}));
  using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using LayoutC = decltype(cute::make_layout(make_shape(0, 0, 0), StrideC{}));
  using StrideD = typename Gemm::GemmKernel::StrideD;
  using LayoutD = decltype(cute::make_layout(make_shape(0, 0, 0), StrideD{}));
};

// SM120 specific GEMM template
template <typename Config, typename OutType>
struct Fp4GemmSm120 {
  using ElementA = tilelang::nv_float4_t<tilelang::float_e2m1_t>;
  using LayoutATag = tilelang::layout::RowMajor;
  static constexpr int AlignmentA = 32;

  using ElementB = tilelang::nv_float4_t<tilelang::float_e2m1_t>;
  using LayoutBTag = tilelang::layout::ColumnMajor;
  static constexpr int AlignmentB = 32;

  using ElementD = OutType;
  using ElementC = OutType;
  using LayoutCTag = tilelang::layout::RowMajor;
  using LayoutDTag = tilelang::layout::RowMajor;
  static constexpr int AlignmentD = 128 / tilelang::sizeof_bits<ElementD>::value;
  static constexpr int AlignmentC = 128 / tilelang::sizeof_bits<ElementC>::value;

  using ElementAccumulator = float;
  using ArchTag = tilelang::arch::Sm120;
  using OperatorClass = tilelang::arch::OpClassBlockScaledTensorOp;

  using MmaTileShape = typename Config::MmaTileShape;
  using ClusterShape = typename Config::ClusterShape;
  using PerSmTileShape_MNK = typename Config::PerSmTileShape_MNK;

  using CollectiveEpilogue = typename tilelang::epilogue::collective::CollectiveBuilder<
      ArchTag,
      OperatorClass,
      PerSmTileShape_MNK,
      ClusterShape,
      tilelang::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator,
      ElementAccumulator,
      ElementC,
      LayoutCTag,
      AlignmentC,
      ElementD,
      LayoutDTag,
      AlignmentD,
      tilelang::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

  using CollectiveMainloop = typename tilelang::gemm::collective::CollectiveBuilder<
      ArchTag,
      OperatorClass,
      ElementA,
      LayoutATag,
      AlignmentA,
      ElementB,
      LayoutBTag,
      AlignmentB,
      ElementAccumulator,
      MmaTileShape,
      ClusterShape,
      tilelang::gemm::collective::StageCountAutoCarveout<static_cast<int>(
          sizeof(typename CollectiveEpilogue::SharedStorage))>,
      tilelang::gemm::collective::KernelScheduleAuto>::CollectiveOp;

  using GemmKernel =
      tilelang::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, void>;

  using Gemm = tilelang::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

template <typename T>
typename T::Gemm::Arguments args_from_options(
    at::Tensor& D,
    at::Tensor const& A,
    at::Tensor const& B,
    at::Tensor const& A_sf,
    at::Tensor const& B_sf,
    at::Tensor const& alpha,
    int64_t M,
    int64_t N,
    int64_t K) {
  using ElementA = typename T::Gemm::ElementA;
  using ElementB = typename T::Gemm::ElementB;
  using ElementSFA = tilelang::float_ue4m3_t;
  using ElementSFB = tilelang::float_ue4m3_t;
  using ElementD = typename T::Gemm::ElementD;
  using ElementCompute = float;
  using StrideA = typename T::StrideA;
  using StrideB = typename T::StrideB;
  using StrideD = typename T::StrideD;
  using Sm1xxBlkScaledConfig = typename T::Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

  int m = static_cast<int>(M);
  int n = static_cast<int>(N);
  int k = static_cast<int>(K);
  auto stride_A = tilelang::make_cute_packed_stride(StrideA{}, {m, k, 1});
  auto stride_B = tilelang::make_cute_packed_stride(StrideB{}, {n, k, 1});
  auto stride_D = tilelang::make_cute_packed_stride(StrideD{}, {m, n, 1});

  auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(m, n, k, 1));
  auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(m, n, k, 1));

  typename T::Gemm::Arguments arguments{
      tilelang::gemm::GemmUniversalMode::kGemm,
      {m, n, k, 1},
      {// Mainloop arguments
       static_cast<ElementA const*>(A.data_ptr()),
       stride_A,
       static_cast<ElementB const*>(B.data_ptr()),
       stride_B,
       static_cast<ElementSFA const*>(A_sf.data_ptr()),
       layout_SFA,
       static_cast<ElementSFB const*>(B_sf.data_ptr()),
       layout_SFB},
      {     // Epilogue arguments
       {},  // epilogue.thread
       nullptr,
       stride_D,
       static_cast<ElementD*>(D.data_ptr()),
       stride_D}};
  auto& fusion_args = arguments.epilogue.thread;
  fusion_args.alpha_ptr = static_cast<ElementCompute const*>(alpha.data_ptr());
  using KernelConfig = typename T::Config;
  arguments.hw_info.cluster_shape = KernelConfig::preferred_cluster;
  arguments.hw_info.cluster_shape_fallback = KernelConfig::fallback_cluster;
  return arguments;
}

template <typename T>
void runGemm(
    at::Tensor& D,
    at::Tensor const& A,
    at::Tensor const& B,
    at::Tensor const& A_sf,
    at::Tensor const& B_sf,
    at::Tensor const& alpha,
    int64_t m,
    int64_t n,
    int64_t k,
    cudaStream_t stream) {
  typename T::Gemm gemm;
  auto arguments = args_from_options<T>(D, A, B, A_sf, B_sf, alpha, m, n, k);

  size_t workspace_size = T::Gemm::get_workspace_size(arguments);
  auto const workspace_options = torch::TensorOptions().dtype(torch::kUInt8).device(A.device());
  auto workspace = torch::empty(workspace_size, workspace_options);

  TILELANG_CHECK(gemm.can_implement(arguments));

  TILELANG_CHECK(gemm.initialize(arguments, workspace.data_ptr(), stream));

  TILELANG_CHECK(gemm.run(arguments, workspace.data_ptr(), stream));
}

// SM120 specific args_from_options function
template <typename Gemm>
typename Gemm::Arguments args_from_options_sm120(
    at::Tensor& D,
    at::Tensor const& A,
    at::Tensor const& B,
    at::Tensor const& A_sf,
    at::Tensor const& B_sf,
    torch::Tensor const& alpha,
    int M,
    int N,
    int K) {
  using ElementA = typename Gemm::ElementA;
  using ElementB = typename Gemm::ElementB;
  using ElementD = typename Gemm::ElementD;
  using ElementSFA = tilelang::float_ue4m3_t;
  using ElementSFB = tilelang::float_ue4m3_t;
  using ElementCompute = float;

  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;

  using Sm1xxBlkScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

  auto stride_A = tilelang::make_cute_packed_stride(StrideA{}, {M, K, 1});
  auto stride_B = tilelang::make_cute_packed_stride(StrideB{}, {N, K, 1});
  auto stride_D = tilelang::make_cute_packed_stride(StrideD{}, {M, N, 1});

  auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(M, N, K, 1));
  auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(M, N, K, 1));

  typename Gemm::Arguments arguments{
      tilelang::gemm::GemmUniversalMode::kGemm,
      {M, N, K, 1},
      {static_cast<ElementA const*>(A.data_ptr()),
       stride_A,
       static_cast<ElementB const*>(B.data_ptr()),
       stride_B,
       static_cast<ElementSFA const*>(A_sf.data_ptr()),
       layout_SFA,
       static_cast<ElementSFB const*>(B_sf.data_ptr()),
       layout_SFB},
      {{}, static_cast<ElementD const*>(D.data_ptr()), stride_D, static_cast<ElementD*>(D.data_ptr()), stride_D}};
  auto& fusion_args = arguments.epilogue.thread;
  fusion_args.alpha_ptr = static_cast<ElementCompute const*>(alpha.data_ptr());

  return arguments;
}

// SM120 specific runGemm function
template <typename Gemm>
void runGemmSm120(
    at::Tensor& D,
    at::Tensor const& A,
    at::Tensor const& B,
    at::Tensor const& A_sf,
    at::Tensor const& B_sf,
    torch::Tensor const& alpha,
    int M,
    int N,
    int K,
    cudaStream_t stream) {
  Gemm gemm;

  auto arguments = args_from_options_sm120<Gemm>(D, A, B, A_sf, B_sf, alpha, M, N, K);

  size_t workspace_size = Gemm::get_workspace_size(arguments);
  auto const workspace_options = torch::TensorOptions().dtype(torch::kUInt8).device(A.device());
  auto workspace = torch::empty(workspace_size, workspace_options);

  TILELANG_CHECK(gemm.can_implement(arguments));

  TILELANG_CHECK(gemm.initialize(arguments, workspace.data_ptr(), stream));

  TILELANG_CHECK(gemm.run(arguments, workspace.data_ptr(), stream));
}

// Dispatch function to select appropriate config based on M
template <typename OutType>
void tilelangFp4GemmDispatch(
    torch::Tensor& D,
    torch::Tensor const& A,
    torch::Tensor const& B,
    torch::Tensor const& A_sf,
    torch::Tensor const& B_sf,
    torch::Tensor const& alpha,
    int64_t m,
    int64_t n,
    int64_t k,
    cudaStream_t stream) {
  if (m <= 128) {
    // m in [1, 128]
    runGemm<Fp4GemmSm100<KernelConfigM128<OutType>>>(D, A, B, A_sf, B_sf, alpha, m, n, k, stream);
  } else if (m <= 256) {
    // m in (128, 256]
    runGemm<Fp4GemmSm100<KernelConfigM256<OutType>>>(D, A, B, A_sf, B_sf, alpha, m, n, k, stream);
  } else {
    // m in (256, inf)
    runGemm<Fp4GemmSm100<KernelConfigDefault<OutType>>>(D, A, B, A_sf, B_sf, alpha, m, n, k, stream);
  }
}

// Dispatch function to select appropriate config based on M
template <>
void tilelangFp4GemmDispatch<float>(
    torch::Tensor& D,
    torch::Tensor const& A,
    torch::Tensor const& B,
    torch::Tensor const& A_sf,
    torch::Tensor const& B_sf,
    torch::Tensor const& alpha,
    int64_t m,
    int64_t n,
    int64_t k,
    cudaStream_t stream) {
  runGemm<Fp4GemmSm100<KernelConfigFp32>>(D, A, B, A_sf, B_sf, alpha, m, n, k, stream);
}

// SM120 specific dispatch functions
void tilelang_fp4_bf16_gemm_dispatch_sm120(
    torch::Tensor& D,
    torch::Tensor const& A,
    torch::Tensor const& B,
    torch::Tensor const& A_sf,
    torch::Tensor const& B_sf,
    torch::Tensor const& alpha,
    int m,
    int n,
    int k,
    cudaStream_t stream) {
  uint32_t const mp2 = std::max(static_cast<uint32_t>(16), next_pow_2(m));
  if (mp2 <= 256) {
    runGemmSm120<Fp4GemmSm120<sm120_fp4_config_M256, tilelang::bfloat16_t>::Gemm>(
        D, A, B, A_sf, B_sf, alpha, m, n, k, stream);
  } else {
    runGemmSm120<Fp4GemmSm120<sm120_fp4_config_default, tilelang::bfloat16_t>::Gemm>(
        D, A, B, A_sf, B_sf, alpha, m, n, k, stream);
  }
}

void tilelang_fp4_f16_gemm_dispatch_sm120(
    torch::Tensor& D,
    torch::Tensor const& A,
    torch::Tensor const& B,
    torch::Tensor const& A_sf,
    torch::Tensor const& B_sf,
    torch::Tensor const& alpha,
    int m,
    int n,
    int k,
    cudaStream_t stream) {
  uint32_t const mp2 = std::max(static_cast<uint32_t>(16), next_pow_2(m));
  if (mp2 <= 256) {
    runGemmSm120<Fp4GemmSm120<sm120_fp4_config_M256, tilelang::half_t>::Gemm>(
        D, A, B, A_sf, B_sf, alpha, m, n, k, stream);
  } else {
    runGemmSm120<Fp4GemmSm120<sm120_fp4_config_default, tilelang::half_t>::Gemm>(
        D, A, B, A_sf, B_sf, alpha, m, n, k, stream);
  }
}

#else
template <typename T>
void tilelangFp4GemmDispatch(
    at::Tensor& D,
    at::Tensor const& A,
    at::Tensor const& B,
    at::Tensor const& A_sf,
    at::Tensor const& B_sf,
    at::Tensor const& alpha,
    int64_t m,
    int64_t n,
    int64_t k,
    cudaStream_t stream) {
  TORCH_CHECK(
      false,
      "Unsupported TILELANG version. Set VLLM_TILELANG_SRC_DIR to "
      "a TILELANG 3.8 source directory to enable support.");
}
#endif  // defined(TILELANG_ARCH_MMA_SM100_SUPPORTED) || defined(TILELANG_ARCH_MMA_SM120_SUPPORTED) ||
        // defined(TILELANG_ARCH_MMA_SM121_SUPPORTED)

// Undefine macros from utils.h to redefine with custom signatures
#undef CHECK_CONTIGUOUS
#undef CHECK_INPUT

#define CHECK_TYPE(x, st, m) TORCH_CHECK(x.scalar_type() == st, "Inconsistency of Tensor type:", m)
#define CHECK_TH_CUDA(x, m) TORCH_CHECK(x.is_cuda(), m, "must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x, m) TORCH_CHECK(x.is_contiguous(), m, "must be contiguous")
#define CHECK_INPUT(x, st, m) \
  CHECK_TH_CUDA(x, m);        \
  CHECK_CONTIGUOUS(x, m);     \
  CHECK_TYPE(x, st, m)

constexpr auto FLOAT4_E2M1X2 = at::ScalarType::Byte;
constexpr auto SF_DTYPE = at::ScalarType::Float8_e4m3fn;

void tilelang_scaled_fp4_mm_sm100a_sm120a(
    torch::Tensor& D,
    torch::Tensor const& A,
    torch::Tensor const& B,
    torch::Tensor const& A_sf,
    torch::Tensor const& B_sf,
    torch::Tensor const& alpha) {
  CHECK_INPUT(A, FLOAT4_E2M1X2, "a");
  CHECK_INPUT(B, FLOAT4_E2M1X2, "b");

  CHECK_INPUT(A_sf, SF_DTYPE, "scale_a");
  CHECK_INPUT(B_sf, SF_DTYPE, "scale_b");

  CHECK_INPUT(alpha, at::ScalarType::Float, "alpha");

  TORCH_CHECK(A.dim() == 2, "a must be a matrix");
  TORCH_CHECK(B.dim() == 2, "b must be a matrix");
  TORCH_CHECK(
      A.size(1) == B.size(1),
      "a and b shapes cannot be multiplied (",
      A.size(0),
      "x",
      A.size(1),
      " and ",
      B.size(0),
      "x",
      B.size(1),
      ")");

  auto const m = A.size(0);
  auto const n = B.size(0);
  auto const k = A.size(1) * 2;

  constexpr int alignment = 32;
  TORCH_CHECK(
      k % alignment == 0,
      "Expected k to be divisible by ",
      alignment,
      ", but got a shape: (",
      A.size(0),
      "x",
      A.size(1),
      "), k: ",
      k,
      ".");
  TORCH_CHECK(
      n % alignment == 0,
      "Expected n to be divisible by ",
      alignment,
      ", but got b shape: (",
      B.size(0),
      "x",
      B.size(1),
      ").");

  auto round_up = [](int x, int y) { return (x + y - 1) / y * y; };
  int rounded_m = round_up(m, 128);
  int rounded_n = round_up(n, 128);
  // Since k is divisible by 32 (alignment), k / 16 is guaranteed to be an
  // integer.
  int rounded_k = round_up(k / 16, 4);

  TORCH_CHECK(A_sf.dim() == 2, "scale_a must be a matrix");
  TORCH_CHECK(B_sf.dim() == 2, "scale_b must be a matrix");
  TORCH_CHECK(
      A_sf.size(1) == B_sf.size(1),
      "scale_a and scale_b shapes cannot be multiplied (",
      A_sf.size(0),
      "x",
      A_sf.size(1),
      " and ",
      B_sf.size(0),
      "x",
      B_sf.size(1),
      ")");
  TORCH_CHECK(
      A_sf.size(0) == rounded_m && A_sf.size(1) == rounded_k,
      "scale_a must be padded and swizzled to a shape (",
      rounded_m,
      "x",
      rounded_k,
      "), but got a shape (",
      A_sf.size(0),
      "x",
      A_sf.size(1),
      ")");
  TORCH_CHECK(
      B_sf.size(0) == rounded_n && B_sf.size(1) == rounded_k,
      "scale_b must be padded and swizzled to a shape (",
      rounded_n,
      "x",
      rounded_k,
      "), but got a shape (",
      B_sf.size(0),
      "x",
      B_sf.size(1),
      ")");

  auto out_dtype = D.dtype();
  at::cuda::CUDAGuard device_guard{(char)A.get_device()};
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(A.get_device());

  // Check SM version and dispatch accordingly
  auto sm_version = getSMVersion();

  if (sm_version == 120) {
    // Use SM120 specific dispatch
    if (out_dtype == at::ScalarType::Half) {
      tilelang_fp4_f16_gemm_dispatch_sm120(D, A, B, A_sf, B_sf, alpha, m, n, k, stream);
    } else if (out_dtype == at::ScalarType::BFloat16) {
      tilelang_fp4_bf16_gemm_dispatch_sm120(D, A, B, A_sf, B_sf, alpha, m, n, k, stream);
    } else {
      TORCH_CHECK(false, "Unsupported output data type of nvfp4 mm sm120 (", out_dtype, ")");
    }
  } else {
    // Use SM100 dispatch for other architectures
    if (out_dtype == at::ScalarType::Half) {
      tilelangFp4GemmDispatch<tilelang::half_t>(D, A, B, A_sf, B_sf, alpha, m, n, k, stream);
    } else if (out_dtype == at::ScalarType::BFloat16) {
      tilelangFp4GemmDispatch<tilelang::bfloat16_t>(D, A, B, A_sf, B_sf, alpha, m, n, k, stream);
    } else if (out_dtype == at::ScalarType::Float) {
      tilelangFp4GemmDispatch<float>(D, A, B, A_sf, B_sf, alpha, m, n, k, stream);
    } else {
      TORCH_CHECK(false, "Unsupported output data type of nvfp4 mm");
    }
  }
}
