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

#ifndef XLA_STREAM_EXECUTOR_VSI_VSI_EXECUTABLE_H_
#define XLA_STREAM_EXECUTOR_VSI_VSI_EXECUTABLE_H_

#include <memory>

#include "absl/container/flat_hash_map.h"
#include "absl/synchronization/mutex.h"
#include "tim/vx/context.h"
#include "tim/vx/graph.h"
#include "tsl/platform/statusor.h"
#include "xla/service/executable.h"
#include "xla/stream_executor/stream_executor.h"

namespace stream_executor {
namespace vsi {

class VsiExecutable final : public xla::Executable {
 public:
  explicit VsiExecutable(std::unique_ptr<xla::HloModule> hlo_module,
                         std::shared_ptr<tim::vx::Graph> vx_graph)
      : xla::Executable(std::move(hlo_module)), vx_graph_(vx_graph) {}

  tsl::StatusOr<xla::ExecutionOutput> ExecuteAsyncOnStream(
      const xla::ServiceExecutableRunOptions* run_options,
      std::vector<xla::ExecutionInput> arguments,
      xla::HloExecutionProfile* hlo_execution_profile) override;

 private:
  tsl::StatusOr<xla::ExecutionOutput> ExecuteSyncLiveTensorsGraph(
      const xla::ServiceExecutableRunOptions* run_options,
      std::vector<xla::ExecutionInput> arguments);

  mutable absl::Mutex mu_;
  bool compiled_ ABSL_GUARDED_BY(mu_) = false;

  absl::flat_hash_map<int64_t, std::shared_ptr<tim::vx::Tensor>>
      input_vx_tensors_;
  std::shared_ptr<tim::vx::Graph> vx_graph_;
};

}  // namespace vsi
}  // namespace stream_executor

#endif