#pragma once

// Misc
#include "cute/tensor.hpp"
#include "tilelang/arch/arch.h"
#include "tilelang/arch/mma.h"
#include "tilelang/tilelang.h"
#include "tilelang/detail/sm100_blockscaled_layout.hpp"
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

// Different configs for 1SM and 2SM MMA kernel
struct MMA1SMConfig {
  using MmaTileShape = Shape<_128, _128, _128>;
  using KernelSchedule = tilelang::gemm::KernelPtrArrayTmaWarpSpecialized1SmMxf8f6f4Sm100;
  using EpilogueSchedule = tilelang::epilogue::PtrArrayTmaWarpSpecialized1Sm;
  const static dim3 preferred_cluster;
  const static dim3 fallback_cluster;
};
const dim3 MMA1SMConfig::preferred_cluster(1, 4, 1);
const dim3 MMA1SMConfig::fallback_cluster(1, 2, 1);

template <typename _MMAConfig, typename OutputDtype>
struct ExpertSpecializationSm100MXFP8BlockscaledGroupedGemmTraits {
  using MMAConfig = _MMAConfig;
  using ElementInput = tilelang::float_e4m3_t;
  using ElementOutput = OutputDtype;
  using ProblemShape = tilelang::gemm::GroupProblemShape<Shape<int, int, int>>;

  // A matrix configuration
  using ElementA = tilelang::mx_float8_t<ElementInput>;
  using LayoutA = tilelang::layout::RowMajor;
  constexpr static int AlignmentA = 32;

  // B matrix configuration
  using ElementB = tilelang::mx_float8_t<ElementInput>;
  using LayoutB = tilelang::layout::ColumnMajor;
  constexpr static int AlignmentB = 32;

  // C/D matrix configuration
  using ElementC = void;
  using ElementD = ElementOutput;
  using LayoutC = tilelang::layout::RowMajor;
  using LayoutD = tilelang::layout::RowMajor;
  constexpr static int AlignmentC = 128 / tilelang::sizeof_bits<ElementD>::value;
  constexpr static int AlignmentD = 128 / tilelang::sizeof_bits<ElementD>::value;
  using ElementAccumulator = float;

  static constexpr auto RoundStyle = tilelang::FloatRoundStyle::round_to_nearest;
  using CustomEVTIdentity =  // acc
      tilelang::epilogue::fusion::Sm90EVT<
          tilelang::epilogue::fusion::
              Sm90Compute<tilelang::epilogue::thread::Identity, ElementD, ElementAccumulator, RoundStyle>,
          tilelang::epilogue::fusion::Sm90AccFetch>;

  // Core kernel configurations
  using ArchTag = tilelang::arch::Sm100;
  using OperatorClass = tilelang::arch::OpClassBlockScaledTensorOp;
  using StageCountType = tilelang::gemm::collective::StageCountAuto;

  // Runtime Cluster Shape
  using ClusterShape = Shape<int32_t, int32_t, _1>;

  // Define Epilogue
  using CollectiveEpilogue = typename tilelang::epilogue::collective::CollectiveBuilder<
      ArchTag,
      OperatorClass,
      typename MMAConfig::MmaTileShape,
      ClusterShape,
      Shape<_64, _64>,
      ElementAccumulator,
      ElementAccumulator,
      ElementC,
      LayoutC*,
      AlignmentC,
      ElementD,
      LayoutD*,
      AlignmentD,
      typename MMAConfig::EpilogueSchedule,
      CustomEVTIdentity>::CollectiveOp;

  // Define Mainloop
  using CollectiveMainloop = typename tilelang::gemm::collective::CollectiveBuilder<
      ArchTag,
      OperatorClass,
      ElementA,
      LayoutA*,
      AlignmentA,
      ElementB,
      LayoutB*,
      AlignmentB,
      ElementAccumulator,
      typename MMAConfig::MmaTileShape,
      ClusterShape,
      tilelang::gemm::collective::StageCountAutoCarveout<static_cast<int>(
          sizeof(typename CollectiveEpilogue::SharedStorage))>,
      typename MMAConfig::KernelSchedule>::CollectiveOp;

  // Define GemmKernel
  using GemmKernel = tilelang::gemm::kernel::GemmUniversal<ProblemShape, CollectiveMainloop, CollectiveEpilogue>;
  using Gemm = tilelang::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using ElementSF = typename Gemm::GemmKernel::ElementSF;
  using StrideA = typename Gemm::GemmKernel::InternalStrideA;
  using StrideB = typename Gemm::GemmKernel::InternalStrideB;
  using StrideC = typename Gemm::GemmKernel::InternalStrideC;
  using StrideD = typename Gemm::GemmKernel::InternalStrideD;
  using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::InternalLayoutSFA;
  using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::InternalLayoutSFB;
  using Sm1xxBlkScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
};

}  // namespace expert_specialization
