// Adapted from
// https://github.com/vllm-project/vllm/blob/main/csrc/quantization/tilelang_w8a8/c3x/scaled_mm_blockwise_sm90_fp8_dispatch.cuh
#pragma once

#include "cute/tensor.hpp"
#include "tilelang/tilelang.h"
#include "tilelang/epilogue/collective/collective_builder.hpp"
#include "tilelang/epilogue/dispatch_policy.hpp"
#include "tilelang/gemm/collective/collective_builder.hpp"
#include "tilelang/gemm/device/gemm_universal_adapter.h"
#include "tilelang/gemm/dispatch_policy.hpp"
#include "tilelang/gemm/kernel/gemm_universal.hpp"
#include "tilelang/gemm/kernel/tile_scheduler_params.h"
#include "tilelang/numeric_types.h"
#include "tilelang/tensor_ref.h"
#include "tilelang_extensions/common.hpp"
#include "tilelang_extensions/gemm/tilelang_gemm_caller.cuh"
#include "tilelang_extensions/gemm/dispatch_policy.hpp"

using namespace cute;

template <
    typename SchedulerType,
    typename OutType,
    int GroupSizeM_,
    int GroupSizeN_,
    int GroupSizeK_,
    int TileSizeM_ = 128,
    class ClusterShape = Shape<_1, _2, _1>>
struct tilelang_3x_gemm_fp8_blockwise {
  using GroupSizeM = Int<GroupSizeM_>;
  using GroupSizeN = Int<GroupSizeN_>;
  using GroupSizeK = Int<GroupSizeK_>;
  using TileSizeM = Int<TileSizeM_>;

  static_assert(TileSizeM_ % GroupSizeM_ == 0, "TileSizeM must be a multiple of GroupSizeM");

  using ElementAB = tilelang::float_e4m3_t;

  // A matrix configuration
  using ElementA = ElementAB;
  using LayoutA = tilelang::layout::RowMajor;
  static constexpr int AlignmentA = 128 / tilelang::sizeof_bits<ElementA>::value;

  // B matrix configuration
  using ElementB = ElementAB;
  using LayoutB = tilelang::layout::ColumnMajor;
  static constexpr int AlignmentB = 128 / tilelang::sizeof_bits<ElementB>::value;

  // C/D matrix configuration
  using ElementC = void;
  using LayoutC = tilelang::layout::RowMajor;
  static constexpr int AlignmentC = 128 / tilelang::sizeof_bits<OutType>::value;

  using ElementD = OutType;
  using LayoutD = tilelang::layout::RowMajor;
  static constexpr int AlignmentD = AlignmentC;

  using ScaleTileShape = Shape<_1, _128, _128>;
  using ScaleConfig = decltype(tilelang::detail::sm90_trivial_blockwise_scale_config(ScaleTileShape{}));
  using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
  using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());

  // Multiply-accumulate blocking/pipelining details
  using ElementAccumulator = float;                            // Element type for internal accumulation
  using ElementCompute = float;                                // Element type for compute
  using TileShape = Shape<TileSizeM, GroupSizeN, GroupSizeK>;  // Threadblock-level tile size

  using ArchTag = tilelang::arch::Sm90;
  using OperatorClass = tilelang::arch::OpClassTensorOp;
  using EpilogueSchedule = tilelang::epilogue::TmaWarpSpecializedCooperative;
  using EpilogueTileType = tilelang::epilogue::collective::EpilogueTileAuto;
  using StoreEpilogueCompute = typename tilelang::epilogue::fusion::Sm90EVT<tilelang::epilogue::fusion::Sm90AccFetch>;

  using KernelSchedule = tilelang::gemm::KernelTmaWarpSpecializedCooperativeFP8Blockwise;
  using CollectiveEpilogue = typename tilelang::epilogue::collective::CollectiveBuilder<
      ArchTag,
      OperatorClass,
      TileShape,
      ClusterShape,
      EpilogueTileType,
      ElementAccumulator,
      ElementCompute,
      ElementC,
      LayoutC,
      AlignmentC,
      ElementD,
      LayoutD,
      AlignmentD,
      EpilogueSchedule,
      StoreEpilogueCompute>::CollectiveOp;

  using CollectiveMainloop = typename tilelang::gemm::collective::CollectiveBuilder<
      ArchTag,
      OperatorClass,
      ElementA,
      cute::tuple<LayoutA, LayoutSFA>,
      AlignmentA,
      ElementB,
      cute::tuple<LayoutB, LayoutSFB>,
      AlignmentB,
      ElementAccumulator,
      TileShape,
      ClusterShape,
      tilelang::gemm::collective::StageCountAutoCarveout<static_cast<int>(
          sizeof(typename CollectiveEpilogue::SharedStorage))>,
      KernelSchedule>::CollectiveOp;

  using GemmKernel = tilelang::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>,  // Indicates ProblemShape
      CollectiveMainloop,
      CollectiveEpilogue,
      SchedulerType>;
};

template <typename Gemm>
void tilelang_gemm_caller_blockwise(
    torch::Tensor& out,
    torch::Tensor const& a,
    torch::Tensor const& b,
    torch::Tensor const& a_scales,
    torch::Tensor const& b_scales) {
  using GemmKernel = typename Gemm::GemmKernel;
  using ElementAB = typename Gemm::ElementAB;
  using ElementA = ElementAB;
  using ElementB = ElementAB;
  using ElementD = typename Gemm::ElementD;
  using ElementBlockScale = float;

  using ScaleTileShape = Shape<_1, _128, _128>;
  using ScaleConfig = decltype(tilelang::detail::sm90_trivial_blockwise_scale_config(ScaleTileShape{}));
  using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
  using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());

  int m = a.size(0);
  int k = a.size(1);
  int n = b.size(1);

  auto a_ptr = static_cast<ElementA*>(a.data_ptr());
  auto b_ptr = static_cast<ElementB*>(b.data_ptr());

  auto a_s_ptr = static_cast<ElementBlockScale*>(a_scales.data_ptr());
  auto b_s_ptr = static_cast<ElementBlockScale*>(b_scales.data_ptr());

  using StrideA = typename GemmKernel::StrideA;
  using StrideB = typename GemmKernel::StrideB;
  using StrideD = typename GemmKernel::StrideD;
  using StrideC = typename GemmKernel::StrideC;

  StrideA a_stride = tilelang::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, 1));
  StrideB b_stride = tilelang::make_cute_packed_stride(StrideB{}, cute::make_shape(n, k, 1));
  StrideC c_stride = tilelang::make_cute_packed_stride(StrideC{}, cute::make_shape(m, n, 1));
  LayoutSFA layout_sfa = ScaleConfig::tile_atom_to_shape_SFA(make_shape(m, n, k, 1));
  LayoutSFB layout_sfb = ScaleConfig::tile_atom_to_shape_SFB(make_shape(m, n, k, 1));

  typename GemmKernel::MainloopArguments mainloop_args{
      a_ptr, a_stride, b_ptr, b_stride, a_s_ptr, layout_sfa, b_s_ptr, layout_sfb};
  auto c_ptr = static_cast<ElementD*>(out.data_ptr());
  typename GemmKernel::EpilogueArguments epilogue_args{{}, c_ptr, c_stride, c_ptr, c_stride};

  typename GemmKernel::TileSchedulerArguments scheduler;

  static constexpr bool UsesStreamKScheduler =
      cute::is_same_v<typename GemmKernel::TileSchedulerTag, tilelang::gemm::StreamKScheduler>;

  if constexpr (UsesStreamKScheduler) {
    using DecompositionMode =
        typename tilelang::gemm::kernel::detail::PersistentTileSchedulerSm90StreamKParams::DecompositionMode;
    using ReductionMode =
        typename tilelang::gemm::kernel::detail::PersistentTileSchedulerSm90StreamKParams::ReductionMode;

    scheduler.decomposition_mode = DecompositionMode::StreamK;
    scheduler.reduction_mode = ReductionMode::Nondeterministic;
  }

  tilelang_gemm_caller<GemmKernel>(a.device(), {m, n, k, 1}, mainloop_args, epilogue_args, scheduler);
}

template <typename OutType>
void tilelang_gemm_blockwise_sm90_fp8_dispatch(
    torch::Tensor& out,
    torch::Tensor const& a,
    torch::Tensor const& b,
    torch::Tensor const& a_scales,
    torch::Tensor const& b_scales) {
  auto k = a.size(1);
  auto n = b.size(1);

  if (k > 3 * n) {
    tilelang_gemm_caller_blockwise<tilelang_3x_gemm_fp8_blockwise<tilelang::gemm::StreamKScheduler, OutType, 1, 128, 128>>(
        out, a, b, a_scales, b_scales);
  } else {
    tilelang_gemm_caller_blockwise<
        tilelang_3x_gemm_fp8_blockwise<tilelang::gemm::PersistentScheduler, OutType, 1, 128, 128>>(
        out, a, b, a_scales, b_scales);
  }
}
