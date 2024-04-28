/* Copyright (c) 2024 VeriSilicon, INC.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
==============================================================================*/

#include "xla/stream_executor/vsi/vsi_compiler.h"

#include <memory>

#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/utils.h"
#include "xla/service/algebraic_simplifier.h"
#include "xla/service/batch_dot_simplification.h"
#include "xla/service/batchnorm_expander.h"
#include "xla/service/call_inliner.h"
#include "xla/service/conditional_simplifier.h"
#include "xla/service/dot_decomposer.h"
#include "xla/service/dynamic_dimension_simplifier.h"
#include "xla/service/dynamic_padder.h"
#include "xla/service/gather_simplifier.h"
#include "xla/service/hlo_constant_folding.h"
#include "xla/service/hlo_cse.h"
#include "xla/service/hlo_dce.h"
#include "xla/service/hlo_pass_pipeline.h"
#include "xla/service/reshape_mover.h"
#include "xla/service/topk_rewriter.h"
#include "xla/service/transpose_folding.h"
#include "xla/service/tuple_simplifier.h"
#include "xla/service/zero_sized_hlo_elimination.h"
#include "xla/shape.h"
#include "xla/stream_executor/platform/initialize.h"
#include "xla/stream_executor/vsi/vsi_executable.h"
#include "xla/stream_executor/vsi/vsi_executor.h"
#include "xla/stream_executor/vsi/vsi_platform_id.h"

namespace stream_executor {
namespace vsi {

xla::StatusOr<std::unique_ptr<xla::HloModule>> VsiCompiler::RunHloPasses(
    std::unique_ptr<xla::HloModule> hlo_module, StreamExecutor* executor,
    const CompileOptions& options) {
  TF_RETURN_IF_ERROR(RunHloOptimization(hlo_module.get()));
  return std::move(hlo_module);
}

xla::StatusOr<std::unique_ptr<xla::Executable>> VsiCompiler::RunBackend(
    std::unique_ptr<xla::HloModule> hlo_module, StreamExecutor* stream_exec,
    const CompileOptions& options) {
  std::shared_ptr<tim::vx::Graph> vx_graph;
  if (IsSyncLiveTensorsGraph(hlo_module.get())) {
    vx_graph = nullptr;
  } else {
    auto* vsi_executor =
        dynamic_cast<VsiExecutor*>(stream_exec->implementation());
    vx_graph = vsi_executor->GetVxContext()->CreateGraph();
  }

  auto executable =
      std::make_unique<VsiExecutable>(std::move(hlo_module), vx_graph);
  return std::move(executable);
}

xla::StatusOr<std::vector<std::unique_ptr<xla::Executable>>>
VsiCompiler::Compile(std::unique_ptr<xla::HloModuleGroup> module_group,
                     std::vector<std::vector<StreamExecutor*>> stream_execs,
                     const CompileOptions& options) {
  if (module_group->empty()) {
    return std::vector<std::unique_ptr<xla::Executable>>();
  }
  if (module_group->size() > 1) {
    return tsl::errors::Unimplemented(
        "Compilation of multiple HLO modules is not supported on vsi.");
  }
  if (stream_execs.size() != 1 || stream_execs[0].size() != 1) {
    return tsl::errors::Unimplemented("Unexpected number of StreamExecutor's.");
  }
  auto hlo_modules = module_group->ConsumeModules();
  TF_ASSIGN_OR_RETURN(auto module, RunHloPasses(std::move(hlo_modules[0]),
                                                stream_execs[0][0], options));
  TF_ASSIGN_OR_RETURN(auto executable, RunBackend(std::move(module),
                                                  stream_execs[0][0], options));
  std::vector<std::unique_ptr<xla::Executable>> ret;
  ret.push_back(std::move(executable));
  return std::move(ret);
}

xla::StatusOr<std::vector<std::unique_ptr<xla::AotCompilationResult>>>
VsiCompiler::CompileAheadOfTime(
    std::unique_ptr<xla::HloModuleGroup> module_group,
    const xla::AotCompilationOptions& aot_options) {
  return tsl::errors::Unimplemented("Not Implemented");
}

xla::HloCostAnalysis::ShapeSizeFunction VsiCompiler::ShapeSizeBytesFunction()
    const {
  return [](const xla::Shape& shape) -> int64_t {
    if (shape.IsOpaque()) {
      return sizeof(void*);
    }
    return xla::ShapeUtil::ByteSizeOf(shape, sizeof(void*));
  };
}

Platform::Id VsiCompiler::PlatformId() const { return kVsiPlatformId; }

bool VsiCompiler::IsSyncLiveTensorsGraph(const xla::HloModule* hlo_module) {
  const auto* entry_computation = hlo_module->entry_computation();
  const auto* root_instr = entry_computation->root_instruction();

  if (root_instr->opcode() == xla::HloOpcode::kParameter ||
      root_instr->opcode() == xla::HloOpcode::kConstant) {
    return true;
  }

  if (root_instr->opcode() == xla::HloOpcode::kTuple &&
      root_instr->operand_count() >= entry_computation->num_parameters()) {
    for (int64_t i = 0; i < root_instr->operand_count(); i++) {
      const auto* tuple_item = root_instr->operand(i);
      if (tuple_item->opcode() != xla::HloOpcode::kParameter &&
          tuple_item->opcode() != xla::HloOpcode::kConstant) {
        return false;
      }
    }
    return true;
  }

  return false;
}

tsl::Status VsiCompiler::RunHloOptimization(xla::HloModule* hlo_module) {
  xla::HloPassPipeline pipeline("vsi_optimizations");

  // Remove zero-sized HLO instructions.
  pipeline.AddPass<xla::ZeroSizedHloElimination>();

  // Inline functions.
  pipeline.AddPass<xla::CallInliner>(true);

  // Decompose BN.
  pipeline.AddPass<xla::BatchNormExpander>(
      /*rewrite_training_op=*/true,
      /*rewrite_inference_op=*/true,
      /*rewrite_grad_op=*/true);

  pipeline.AddPass<xla::DynamicDimensionSimplifier>();
  pipeline.AddPass<xla::DynamicPadder>();

  // BatchNormExpander can create zero-sized ops, so a ZeroSizedHloElimination
  // pass has to come after that pass.
  pipeline.AddPass<xla::ZeroSizedHloElimination>();
  pipeline.AddPass<xla::HloDCE>();

  // Canonicalize dot op.
  pipeline.AddPass<xla::BatchDotSimplification>();
  pipeline.AddPass<xla::DotDecomposer>();
  // Fold transpose + dot.
  pipeline.AddPass<xla::TransposeFolding>(
      [&](const xla::HloInstruction& dot,
          int64_t operand) -> tsl::StatusOr<bool> {
        return xla::TransposeFolding::IsRowColumnTransposeDotOperand(dot,
                                                                     operand);
      },
      xla::TransposeFolding::NeverFoldTranspose);

  pipeline.AddPass<xla::HloDCE>();
  pipeline.AddPass<xla::ReshapeMover>();
  pipeline.AddPass<xla::HloConstantFolding>();
  pipeline.AddPass<xla::ConditionalSimplifier>();
  pipeline.AddPass<xla::GatherSimplifier>();

  // Capture TopK pattern into custom call.
  pipeline.AddPass<xla::TopkDecomposer>([&](const xla::HloInstruction* topk) {
    return topk->opcode() == xla::HloOpcode::kTopK;
  });
  pipeline.AddPass<xla::TopkRewriter>([](const xla::HloSortInstruction* sort,
                                         int64_t) -> bool { return true; });

  pipeline.AddPass<xla::TupleSimplifier>();
  pipeline.AddPass<xla::HloCSE>(/*is_layout_sensitive=*/false);
  pipeline.AddPass<xla::HloDCE>();

  return pipeline.Run(hlo_module).status();
}

static void InitializeVsiCompiler() {
  xla::Compiler::RegisterCompilerFactory(
      kVsiPlatformId, []() { return std::make_unique<VsiCompiler>(); });

  xla::ComputationPlacer::RegisterComputationPlacer(kVsiPlatformId, []() {
    return std::make_unique<xla::ComputationPlacer>();
  });
}

}  // namespace vsi
}  // namespace stream_executor

STREAM_EXECUTOR_REGISTER_MODULE_INITIALIZER(
    vsi_compiler, stream_executor::vsi::InitializeVsiCompiler());
