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

#ifndef XLA_STREAM_EXECUTOR_VSI_VSI_COMPILER_H_
#define XLA_STREAM_EXECUTOR_VSI_VSI_COMPILER_H_

#include <memory>
#include <vector>

#include "tsl/platform/status.h"
#include "tsl/platform/statusor.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_module_group.h"
#include "xla/service/compiler.h"
#include "xla/service/executable.h"

namespace stream_executor {
namespace vsi {

class VsiCompiler final : public xla::Compiler {
 public:
  VsiCompiler() = default;
  ~VsiCompiler() override = default;

  tsl::StatusOr<std::unique_ptr<xla::HloModule>> RunHloPasses(
      std::unique_ptr<xla::HloModule> hlo_module, StreamExecutor* executor,
      const CompileOptions& options) override;
  tsl::StatusOr<std::unique_ptr<xla::Executable>> RunBackend(
      std::unique_ptr<xla::HloModule> hlo_module, StreamExecutor* stream_exec,
      const CompileOptions& options) override;
  tsl::StatusOr<std::vector<std::unique_ptr<xla::Executable>>> Compile(
      std::unique_ptr<xla::HloModuleGroup> module_group,
      std::vector<std::vector<StreamExecutor*>> stream_execs,
      const CompileOptions& options) override;

  tsl::StatusOr<std::vector<std::unique_ptr<xla::AotCompilationResult>>>
  CompileAheadOfTime(std::unique_ptr<xla::HloModuleGroup> module_group,
                     const xla::AotCompilationOptions& aot_options) override;

  xla::HloCostAnalysis::ShapeSizeFunction ShapeSizeBytesFunction()
      const override;

  Platform::Id PlatformId() const override;

 private:
  static bool IsSyncLiveTensorsGraph(const xla::HloModule* hlo_module);

  tsl::Status RunHloOptimization(xla::HloModule* hlo_module);

  TF_DISALLOW_COPY_AND_ASSIGN(VsiCompiler);
};

}  // namespace vsi
}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_VSI_VSI_COMPILER_H_