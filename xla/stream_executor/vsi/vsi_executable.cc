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

#include "xla/stream_executor/vsi/vsi_executable.h"

#include "tim/vx/tensor.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal.h"
#include "xla/service/maybe_owning_device_memory.h"
#include "xla/service/shaped_buffer.h"
#include "xla/service/transfer_manager.h"
#include "xla/shape_tree.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/vsi/vsi_executor.h"
#include "xla/stream_executor/vsi/vsi_graph_builder.h"

namespace stream_executor {
namespace vsi {

tsl::StatusOr<xla::ExecutionOutput> VsiExecutable::ExecuteAsyncOnStream(
    const xla::ServiceExecutableRunOptions* run_options,
    std::vector<xla::ExecutionInput> arguments,
    xla::HloExecutionProfile* hlo_execution_profile) {
  const xla::HloComputation* entry_comp = module().entry_computation();
  CHECK_EQ(entry_comp->num_parameters(), arguments.size())
      << "Wrong number of arguments passed when running executable";
  for (int64_t i = 0; i < entry_comp->num_parameters(); ++i) {
    const xla::Shape& expected_shape =
        entry_comp->parameter_instruction(i)->shape();
    const xla::Shape& actual_shape = arguments[i].Buffers().shape();
    CHECK_EQ(actual_shape, expected_shape)
        << "Shape mismatch on argument " << i << ", "
        << expected_shape.ToString(true) << " vs. "
        << actual_shape.ToString(true);
  }

  if (vx_graph_ == nullptr) {
    return ExecuteSyncLiveTensorsGraph(run_options, std::move(arguments));
  }

  {
    absl::MutexLock lock(&mu_);
    if (!compiled_) {
      auto graph_builder = std::make_unique<VsiGraphBuilder>(vx_graph_);

      TF_RETURN_IF_ERROR(
          module().entry_computation()->Accept(graph_builder.get()));

      // Map input parameters to their corresponding vx tensors.
      for (const auto* parameter :
           module().entry_computation()->parameter_instructions()) {
        auto input_vx_tensor = graph_builder->GetMappedVxTensor(parameter);
        input_vx_tensors_.insert(
            {parameter->parameter_number(), input_vx_tensor});
      }

      compiled_ = true;
      VLOG(1) << "TIM-VX graph compiled";
    }
  }

  auto* stream = run_options->stream();
  auto* stream_executor = stream->parent();
  const auto* platform = stream_executor->platform();
  TF_ASSIGN_OR_RETURN(xla::TransferManager * transfer_manager,
                      xla::TransferManager::GetForPlatform(platform));

  uint64_t start_micros = tsl::Env::Default()->NowMicros();

  // Allocate result buffer.
  auto root_hlo_instr = entry_comp->root_instruction();
  auto result_shape = root_hlo_instr->shape();

  TF_ASSIGN_OR_RETURN(xla::ScopedShapedBuffer result_buffer,
                      transfer_manager->AllocateScopedShapedBuffer(
                          result_shape, run_options->allocator(),
                          stream_executor->device_ordinal()));

  for (size_t i = 0; i < arguments.size(); i++) {
    auto input_vx_tensor = input_vx_tensors_.at(i);
    const auto& argument = arguments[i];
    const auto& arg_memory = argument.Buffer({}).AsDeviceMemoryBase();
    input_vx_tensor->CopyDataToTensor(arg_memory.opaque());
  }

  if (!vx_graph_->Run()) {
    return tsl::errors::Internal("Failed to run vx graph");
  }

  auto output_vx_tensors = vx_graph_->OutputsTensor();
  if (result_shape.IsTuple()) {
    for (auto& [index, result_memory] : result_buffer.buffers()) {
      if (index.empty()) {
        continue;
      }
      auto output_vx_tensor = output_vx_tensors[index[0]];
      output_vx_tensor->CopyDataFromTensor(result_memory.opaque());
    }
  } else {
    auto& result_memory =
        const_cast<DeviceMemoryBase&>(result_buffer.root_buffer());
    auto output_vx_tensor = output_vx_tensors[0];
    output_vx_tensor->CopyDataFromTensor(result_memory.opaque());
  }

  uint64_t end_micros = tsl::Env::Default()->NowMicros();

  xla::ExecutionProfile* profile =
      run_options->run_options().execution_profile();
  if (profile) {
    const double nanoseconds = (end_micros - start_micros) * 1000.0;
    profile->set_compute_time_ns(std::max(nanoseconds, 1.0));
  }
  xla::ExecutionOutput result(std::move(result_buffer));
  // MarkToBeReleasedArguments(absl::MakeSpan(arguments), result);
  return std::move(result);
}

tsl::StatusOr<xla::ExecutionOutput> VsiExecutable::ExecuteSyncLiveTensorsGraph(
    const xla::ServiceExecutableRunOptions* run_options,
    std::vector<xla::ExecutionInput> arguments) {
  const auto* root_instr = module().entry_computation()->root_instruction();
  const auto& result_shape = root_instr->shape();
  auto* stream = run_options->stream();
  auto* stream_executor = stream->parent();
  const auto* platform = stream_executor->platform();
  TF_ASSIGN_OR_RETURN(xla::TransferManager * transfer_manager,
                      xla::TransferManager::GetForPlatform(platform));

  TF_ASSIGN_OR_RETURN(xla::ScopedShapedBuffer result_buffer,
                      transfer_manager->AllocateScopedShapedBuffer(
                          result_shape, run_options->allocator(),
                          stream_executor->device_ordinal()));

  if (result_buffer.on_device_shape().IsTuple()) {
    for (auto& [index, result_memory] : result_buffer.buffers()) {
      if (index.empty()) {
        continue;
      }

      const auto* tuple_item = root_instr->operand(index[0]);
      if (tuple_item->IsConstant()) {
        const auto& literal = tuple_item->literal();
        std::memcpy(result_memory.opaque(), literal.untyped_data(),
                    literal.size_bytes());
      } else {
        const auto& arg_memory = arguments[tuple_item->parameter_number()]
                                     .Buffer({})
                                     .AsDeviceMemoryBase();
        std::memcpy(result_memory.opaque(), arg_memory.opaque(),
                    arg_memory.size());
      }
    }
  } else {
    const auto& arg_memory = arguments[0].Buffer({}).AsDeviceMemoryBase();
    auto& result_memory =
        const_cast<DeviceMemoryBase&>(result_buffer.root_buffer());
    std::memcpy(result_memory.opaque(), arg_memory.opaque(), arg_memory.size());
  }

  xla::ExecutionOutput result(std::move(result_buffer));
  return std::move(result);
}

}  // namespace vsi
}  // namespace stream_executor