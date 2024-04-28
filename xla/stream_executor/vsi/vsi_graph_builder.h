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

#ifndef XLA_STREAM_EXECUTOR_VSI_VSI_GRAPH_BUILDER_H_
#define XLA_STREAM_EXECUTOR_VSI_VSI_GRAPH_BUILDER_H_

#include <memory>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "tim/vx/graph.h"
#include "tsl/platform/status.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"

namespace stream_executor {
namespace vsi {

class VsiGraphBuilder final : public xla::ConstDfsHloVisitorWithDefault {
 public:
  explicit VsiGraphBuilder(std::shared_ptr<tim::vx::Graph> vx_graph)
      : vx_graph_(vx_graph) {}

  tsl::Status DefaultAction(
      const xla::HloInstruction* hlo_instruction) override;

  tsl::Status HandleParameter(const xla::HloInstruction* parameter) override;

  tsl::Status HandleConstant(const xla::HloInstruction* constant) override;

  tsl::Status HandleConvert(const xla::HloInstruction* convert) override;

  tsl::Status HandleTuple(const xla::HloInstruction* tuple) override;

  tsl::Status HandleGetTupleElement(
      const xla::HloInstruction* get_tuple_item) override;

  tsl::Status HandleReshape(const xla::HloInstruction* reshape) override;

  tsl::Status HandleBroadcast(const xla::HloInstruction* broadcast) override;

  tsl::Status HandleTranspose(const xla::HloInstruction* transpose) override;

  tsl::Status HandleSlice(const xla::HloInstruction* slice) override;

  tsl::Status HandleConcatenate(
      const xla::HloInstruction* concatenate) override;

  tsl::Status HandleElementwiseUnary(
      const xla::HloInstruction* elementwise_unary_op) override;

  tsl::Status HandleElementwiseBinary(
      const xla::HloInstruction* elementwise_binary_op) override;

  tsl::Status HandleCompare(const xla::HloInstruction* compare) override;

  tsl::Status HandleSelect(const xla::HloInstruction* select) override;

  tsl::Status HandleGather(const xla::HloInstruction* gather) override;

  tsl::Status HandleConvolution(
      const xla::HloInstruction* convolution) override;

  tsl::Status HandleDot(const xla::HloInstruction* dot) override;

  tsl::Status HandleReduceWindow(
      const xla::HloInstruction* reduce_window) override;

  tsl::Status HandleReduce(const xla::HloInstruction* reduce) override;

  tsl::Status HandlePad(const xla::HloInstruction* pad) override;

  tsl::Status HandleBatchNormInference(
      const xla::HloInstruction* batch_norm_inference) override;

  tsl::Status HandleClamp(const xla::HloInstruction* clamp) override;

  tsl::Status HandleCustomCall(const xla::HloInstruction* custom_call) override;

  // Invoked to inform the visitor that the traversal has completed, and that
  // the root was "root".
  tsl::Status FinishVisit(const xla::HloInstruction* /*root*/) override;

  std::shared_ptr<tim::vx::Tensor> GetMappedVxTensor(
      const xla::HloInstruction* hlo_instr) {
    return hlo_instr_to_vx_tensor_.at(hlo_instr);
  }

 private:
  tsl::Status HandleConv2d(const xla::HloInstruction* conv2d);
  tsl::Status HandleTransposedConv2d(
      const xla::HloInstruction* transposed_conv2d);

  tsl::Status HandleQuantize(const xla::HloInstruction* quantize);
  tsl::Status HandleDequantize(const xla::HloInstruction* dequantize);

  std::vector<std::shared_ptr<tim::vx::Tensor>> CreateOpOutputTensors(
      const xla::HloInstruction* hlo_instr);
  std::shared_ptr<tim::vx::Tensor> CreateOpOutputTensor(
      const xla::HloInstruction* hlo_instr);

  absl::flat_hash_map<const xla::HloInstruction*,
                      std::shared_ptr<tim::vx::Tensor>>
      hlo_instr_to_vx_tensor_;
  absl::flat_hash_map<const xla::HloInstruction*,
                      std::vector<std::shared_ptr<tim::vx::Tensor>>>
      hlo_instr_to_vx_tensors_;

  std::shared_ptr<tim::vx::Graph> vx_graph_;

  TF_DISALLOW_COPY_AND_ASSIGN(VsiGraphBuilder);
};

}  // namespace vsi
}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_VSI_VSI_GRAPH_BUILDER_H_