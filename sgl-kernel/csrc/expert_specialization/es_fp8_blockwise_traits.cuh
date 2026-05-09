#pragma once

// Misc
#include "cute/tensor.hpp"
#include "tilelang/arch/arch.h"
#include "tilelang/arch/mma.h"
#include "tilelang/tilelang.h"
#include "tilelang/detail/blockwise_scale_layout.hpp"
#include "tilelang/epilogue/dispatch_policy.hpp"
#include "tilelang/gemm/dispatch_policy.hpp"
#include "tilelang/gemm/group_array_problem_shape.hpp"
#include "tilelang/layout/layout.h"
#include "tilelang/numeric_conversion.h"
#include "tilelang/numeric_size.h"

// Collective Builder
#include "tilelang/epilogue/collective/collective_builder.hpp"
#include "tilelang/epilogue/fusion/sm90_callbacks_tma_warpspecialized.hpp"
#include "tilelang/epilogue/thread/activation.h"
#include "tilelang/gemm/collective/collective_builder.hpp"

// Integration
#include "tilelang/gemm/device/gemm_universal_adapter.h"
#include "tilelang/gemm/kernel/gemm_universal.hpp"

namespace expert_specialization {

using namespace cute;

struct PerfConfigLowMH20 {
  // Swap A/B
  using ElementA = tilelang::float_e4m3_t;
  using MmaTileShape = Shape<_128, _32, _128>;
  using ClusterShape = Shape<_2, _1, _1>;
  using KernelSchedule = tilelang::gemm::KernelPtrArrayTmaWarpSpecializedPingpongFP8Blockwise;
  using EpilogueSchedule = tilelang::epilogue::PtrArrayTmaWarpSpecializedPingpong;
  using ScaleConfig =
      tilelang::detail::Sm90BlockwiseScaleConfig<128, 1, 128, cute::GMMA::Major::K, cute::GMMA::Major::K>;
  using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
  using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());
};

struct PerfConfigLowMHx00 {
  // Swap A/B
  using ElementA = tilelang::float_e4m3_t;
  using MmaTileShape = Shape<_256, _32, _128>;
  using ClusterShape = Shape<_2, _1, _1>;
  using KernelSchedule = tilelang::gemm::KernelPtrArrayTmaWarpSpecializedCooperativeFP8Blockwise;
  using EpilogueSchedule = tilelang::epilogue::PtrArrayTmaWarpSpecializedCooperative;
  using ScaleConfig =
      tilelang::detail::Sm90BlockwiseScaleConfig<128, 1, 128, cute::GMMA::Major::K, cute::GMMA::Major::K>;
  using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
  using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());
};

struct PerfConfigMiddleMH20 {
  using ElementA = tilelang::float_e4m3_t;
  using MmaTileShape = Shape<_64, _128, _128>;
  using ClusterShape = Shape<_1, _2, _1>;
  using KernelSchedule = tilelang::gemm::KernelPtrArrayTmaWarpSpecializedPingpongFP8Blockwise;
  using EpilogueSchedule = tilelang::epilogue::PtrArrayTmaWarpSpecializedPingpong;
  using ScaleConfig =
      tilelang::detail::Sm90BlockwiseScaleConfig<1, 128, 128, cute::GMMA::Major::K, cute::GMMA::Major::K>;
  using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
  using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());
};

struct PerfConfigMiddleMHx00 {
  using ElementA = tilelang::float_e4m3_t;
  using MmaTileShape = Shape<_256, _64, _128>;
  using ClusterShape = Shape<_2, _1, _1>;
  using KernelSchedule = tilelang::gemm::KernelPtrArrayTmaWarpSpecializedCooperativeFP8Blockwise;
  using EpilogueSchedule = tilelang::epilogue::PtrArrayTmaWarpSpecializedCooperative;
  using ScaleConfig =
      tilelang::detail::Sm90BlockwiseScaleConfig<128, 1, 128, cute::GMMA::Major::K, cute::GMMA::Major::K>;
  using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
  using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());
};

struct PerfConfigHighMH20 {
  using ElementA = tilelang::float_e4m3_t;
  using MmaTileShape = Shape<_64, _128, _128>;
  using ClusterShape = Shape<_2, _1, _1>;
  using KernelSchedule = tilelang::gemm::KernelPtrArrayTmaWarpSpecializedPingpongFP8Blockwise;
  using EpilogueSchedule = tilelang::epilogue::PtrArrayTmaWarpSpecializedPingpong;
  using ScaleConfig =
      tilelang::detail::Sm90BlockwiseScaleConfig<1, 128, 128, cute::GMMA::Major::K, cute::GMMA::Major::K>;
  using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
  using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());
};

struct PerfConfigHighMHx00 {
  using ElementA = tilelang::float_e4m3_t;
  using MmaTileShape = Shape<_128, _128, _128>;
  using ClusterShape = Shape<_1, _2, _1>;
  using KernelSchedule = tilelang::gemm::KernelPtrArrayTmaWarpSpecializedCooperativeFP8Blockwise;
  using EpilogueSchedule = tilelang::epilogue::PtrArrayTmaWarpSpecializedCooperative;
  using ScaleConfig =
      tilelang::detail::Sm90BlockwiseScaleConfig<1, 128, 128, cute::GMMA::Major::K, cute::GMMA::Major::K>;
  using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
  using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());
};

template <typename OutType, typename LayoutD, typename PerfConfig>
struct ExpertSpecializationSm90FP8BlockwiseGroupedGemmTraits {
  using ElementA = tilelang::float_e4m3_t;
  using ElementB = tilelang::float_e4m3_t;
  using ElementC = void;
  using ElementD = OutType;
  using ElementAccumulator = float;
  using LayoutA = tilelang::layout::RowMajor;
  using LayoutB = tilelang::layout::ColumnMajor;
  using LayoutC = LayoutD;
  using LayoutSFA = typename PerfConfig::LayoutSFA;
  using LayoutSFB = typename PerfConfig::LayoutSFB;
  using ProblemShape = tilelang::gemm::GroupProblemShape<Shape<int, int, int>>;

  static constexpr int AlignmentA = 128 / tilelang::sizeof_bits<ElementA>::value;
  static constexpr int AlignmentB = 128 / tilelang::sizeof_bits<ElementB>::value;
  static constexpr int AlignmentC = 128 / tilelang::sizeof_bits<ElementD>::value;
  static constexpr int AlignmentD = 128 / tilelang::sizeof_bits<ElementD>::value;

  using ArchTag = tilelang::arch::Sm90;
  using OperatorClass = tilelang::arch::OpClassTensorOp;
  static constexpr auto RoundStyle = tilelang::FloatRoundStyle::round_to_nearest;
  using CustomEVTIdentity =  // acc
      tilelang::epilogue::fusion::Sm90EVT<
          tilelang::epilogue::fusion::
              Sm90Compute<tilelang::epilogue::thread::Identity, ElementD, ElementAccumulator, RoundStyle>,
          tilelang::epilogue::fusion::Sm90AccFetch>;

  using CollectiveEpilogue = typename tilelang::epilogue::collective::CollectiveBuilder<
      ArchTag,
      OperatorClass,
      typename PerfConfig::MmaTileShape,
      typename PerfConfig::ClusterShape,
      tilelang::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator,
      ElementAccumulator,
      ElementC,  // Use void to avoid load Matrix C
      LayoutC*,
      AlignmentC,
      ElementD,
      LayoutD*,
      AlignmentD,
      typename PerfConfig::EpilogueSchedule,
      CustomEVTIdentity>::CollectiveOp;

  using CollectiveMainloop = typename tilelang::gemm::collective::CollectiveBuilder<
      ArchTag,
      OperatorClass,
      ElementA,
      cute::tuple<LayoutA*, typename PerfConfig::LayoutSFA*>,
      AlignmentA,
      ElementB,
      cute::tuple<LayoutB*, typename PerfConfig::LayoutSFB*>,
      AlignmentB,
      ElementAccumulator,
      typename PerfConfig::MmaTileShape,
      typename PerfConfig::ClusterShape,
      tilelang::gemm::collective::StageCountAutoCarveout<static_cast<int>(
          sizeof(typename CollectiveEpilogue::SharedStorage))>,
      typename PerfConfig::KernelSchedule>::CollectiveOp;

  using GemmKernel = tilelang::gemm::kernel::GemmUniversal<ProblemShape, CollectiveMainloop, CollectiveEpilogue, void>;
  using Gemm = tilelang::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using UnderlyingProblemShape = ProblemShape::UnderlyingProblemShape;
  using StrideA = typename Gemm::GemmKernel::InternalStrideA;
  using StrideB = typename Gemm::GemmKernel::InternalStrideB;
  using StrideC = typename Gemm::GemmKernel::InternalStrideC;
  using StrideD = typename Gemm::GemmKernel::InternalStrideD;
};

}  // namespace expert_specialization
