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

#include "xla/stream_executor/vsi/vsi_graph_builder.h"

#include <cstdint>
#include <memory>

#include "mlir/AsmParser/AsmParser.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/MLIRContext.h"
#include "tim/vx/ops.h"
#include "tsl/platform/logging.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"

namespace stream_executor {
namespace vsi {

namespace utils {

static inline tim::vx::DataType ConvertXlaDataType(xla::PrimitiveType dtype) {
  switch (dtype) {
    case xla::PrimitiveType::S8:
      return tim::vx::DataType::INT8;
    case xla::PrimitiveType::U8:
      return tim::vx::DataType::UINT8;
    case xla::PrimitiveType::S16:
      return tim::vx::DataType::INT16;
    case xla::PrimitiveType::U16:
      return tim::vx::DataType::UINT16;
    case xla::PrimitiveType::S32:
      return tim::vx::DataType::INT32;
    case xla::PrimitiveType::U32:
      return tim::vx::DataType::UINT32;
    case xla::PrimitiveType::S64:
      return tim::vx::DataType::INT64;
    case xla::PrimitiveType::F16:
      return tim::vx::DataType::FLOAT16;
    case xla::PrimitiveType::F32:
      return tim::vx::DataType::FLOAT32;
    case xla::PrimitiveType::PRED:
      return tim::vx::DataType::BOOL8;
    default:
      return tim::vx::DataType::UNKNOWN;
  }
}

static inline tim::vx::TensorSpec ConvertXlaShape(
    const xla::Shape& shape, tim::vx::TensorAttribute tensor_attr) {
  auto vx_dtype = ConvertXlaDataType(shape.element_type());
  auto vx_shape = tim::vx::ShapeType(shape.rank());

  for (int64_t i = 0; i < shape.rank(); i++) {
    vx_shape[shape.rank() - 1 - i] = static_cast<uint32_t>(shape.dimensions(i));
  }

  // TIM-VX does not support zero-ranked tensors.
  if (shape.rank() == 0) {
    vx_shape.push_back(1);
  }

  auto vx_spec = tim::vx::TensorSpec(vx_dtype, vx_shape, tensor_attr);
  return vx_spec;
}

static tim::vx::Quantization ConvertQuantParams(
    const mlir::DictionaryAttr& quant_attr, int64_t rank) {
  int32_t vx_quant_axis = -1;
  std::vector<float> scales;
  std::vector<int32_t> zero_points;

  auto quant_axis_attr = quant_attr.get("quantization_dimension");
  if (quant_axis_attr) {
    int64_t quant_axis = quant_axis_attr.cast<mlir::IntegerAttr>().getInt();
    vx_quant_axis =
        (quant_axis != -1) ? static_cast<int32_t>(rank - 1 - quant_axis) : -1;
  }

  auto scales_attr = quant_attr.get("scale");
  if (scales_attr) {
    for (auto scale_attr : scales_attr.cast<mlir::ArrayAttr>()) {
      scales.push_back(static_cast<float>(
          scale_attr.cast<mlir::FloatAttr>().getValueAsDouble()));
    }
  }

  auto zero_points_attr = quant_attr.get("zero_point");
  if (zero_points_attr) {
    for (auto zero_point_attr : zero_points_attr.cast<mlir::ArrayAttr>()) {
      zero_points.push_back(static_cast<int32_t>(
          zero_point_attr.cast<mlir::IntegerAttr>().getInt()));
    }
  }

  return (vx_quant_axis != -1)
             ? tim::vx::Quantization(tim::vx::QuantType::SYMMETRIC_PER_CHANNEL,
                                     vx_quant_axis, scales, zero_points)
             : tim::vx::Quantization(tim::vx::QuantType::ASYMMETRIC, scales[0],
                                     zero_points[0]);
}

}  // namespace utils

tsl::Status VsiGraphBuilder::DefaultAction(
    const xla::HloInstruction* hlo_instruction) {
  return tsl::errors::Unimplemented("Unsupported HLO OP: ",
                                    HloOpcodeString(hlo_instruction->opcode()));
}

tsl::Status VsiGraphBuilder::FinishVisit(const xla::HloInstruction* /*root*/) {
  bool is_compile_successful = vx_graph_->Compile();
  if (is_compile_successful) {
    return tsl::OkStatus();
  } else {
    return tsl::errors::Internal("Failed to compile TIM-VX graph.");
  }
}

tsl::Status VsiGraphBuilder::HandleParameter(
    const xla::HloInstruction* parameter) {
  auto vx_spec = utils::ConvertXlaShape(parameter->shape(),
                                        tim::vx::TensorAttribute::INPUT);
  auto vx_tensor = vx_graph_->CreateIOTensor(vx_spec);

  hlo_instr_to_vx_tensor_.insert({parameter, vx_tensor});

  return tsl::OkStatus();
}

tsl::Status VsiGraphBuilder::HandleConstant(
    const xla::HloInstruction* constant) {
  auto vx_spec = utils::ConvertXlaShape(constant->shape(),
                                        tim::vx::TensorAttribute::CONSTANT);

  const auto& literal = constant->literal();
  auto vx_tensor = vx_graph_->CreateTensor(vx_spec, literal.untyped_data());

  hlo_instr_to_vx_tensor_.insert({constant, vx_tensor});

  return tsl::OkStatus();
}

tsl::Status VsiGraphBuilder::HandleConvert(const xla::HloInstruction* convert) {
  const auto* input_operand = convert->operand(0);
  auto input_tensor = hlo_instr_to_vx_tensor_.at(input_operand);
  auto output_tensor = CreateOpOutputTensor(convert);

  auto vx_cast_op = vx_graph_->CreateOperation<tim::vx::ops::Cast>();
  vx_cast_op->BindInput(input_tensor);
  vx_cast_op->BindOutput(output_tensor);

  return tsl::OkStatus();
}

tsl::Status VsiGraphBuilder::HandleTuple(const xla::HloInstruction* tuple) {
  std::vector<std::shared_ptr<tim::vx::Tensor>> vx_tensors;
  for (const auto* tuple_item : tuple->operands()) {
    auto vx_tensor = hlo_instr_to_vx_tensor_.at(tuple_item);
    vx_tensors.push_back(vx_tensor);
  }

  hlo_instr_to_vx_tensors_.insert({tuple, vx_tensors});

  return tsl::OkStatus();
}

tsl::Status VsiGraphBuilder::HandleGetTupleElement(
    const xla::HloInstruction* get_tuple_item) {
  const auto* tuple = get_tuple_item->operand(0);
  auto vx_tensors = hlo_instr_to_vx_tensors_.at(tuple);
  auto vx_tensor = vx_tensors[get_tuple_item->tuple_index()];

  hlo_instr_to_vx_tensor_.insert({get_tuple_item, vx_tensor});

  return tsl::OkStatus();
}

tsl::Status VsiGraphBuilder::HandleReshape(const xla::HloInstruction* reshape) {
  const auto* input_operand = reshape->operand(0);
  auto input_tensor = hlo_instr_to_vx_tensor_.at(input_operand);
  auto output_tensor = CreateOpOutputTensor(reshape);

  auto vx_reshape_op = vx_graph_->CreateOperation<tim::vx::ops::Reshape>(
      output_tensor->GetShape());
  vx_reshape_op->BindInput(input_tensor);
  vx_reshape_op->BindOutput(output_tensor);

  return tsl::OkStatus();
}

tsl::Status VsiGraphBuilder::HandleBroadcast(
    const xla::HloInstruction* broadcast) {
  const auto* input_operand = broadcast->operand(0);
  auto input_tensor = hlo_instr_to_vx_tensor_.at(input_operand);
  auto output_tensor = CreateOpOutputTensor(broadcast);

  auto xla_dims = broadcast->dimensions();
  int64_t output_rank = broadcast->shape().rank();
  std::vector<int32_t> vx_dims(xla_dims.size());
  for (int64_t i = 0; i < xla_dims.size(); i++) {
    vx_dims[i] = static_cast<int32_t>(output_rank - 1 - xla_dims[i]);
  }

  auto vx_broadcast_op = vx_graph_->CreateOperation<tim::vx::ops::Broadcast>(
      output_tensor->GetShape(), vx_dims);
  vx_broadcast_op->BindInput(input_tensor);
  vx_broadcast_op->BindOutput(output_tensor);

  return tsl::OkStatus();
}

tsl::Status VsiGraphBuilder::HandleTranspose(
    const xla::HloInstruction* transpose) {
  const auto* input_operand = transpose->operand(0);
  auto input_tensor = hlo_instr_to_vx_tensor_.at(input_operand);
  auto output_tensor = CreateOpOutputTensor(transpose);

  auto xla_perm = transpose->dimensions();
  int64_t rank = transpose->shape().rank();
  std::vector<uint32_t> vx_perm(rank);
  for (int64_t i = 0; i < rank; i++) {
    vx_perm[rank - 1 - i] = static_cast<uint32_t>(rank - 1 - xla_perm[i]);
  }

  auto vx_transpose_op =
      vx_graph_->CreateOperation<tim::vx::ops::Transpose>(vx_perm);
  vx_transpose_op->BindInput(input_tensor);
  vx_transpose_op->BindOutput(output_tensor);

  return tsl::OkStatus();
}

tsl::Status VsiGraphBuilder::HandleSlice(const xla::HloInstruction* slice) {
  const auto* input_operand = slice->operand(0);
  auto input_tensor = hlo_instr_to_vx_tensor_.at(input_operand);
  auto output_tensor = CreateOpOutputTensor(slice);

  int64_t rank = input_operand->shape().rank();
  std::vector<int32_t> vx_begin(rank);
  std::vector<int32_t> vx_end(rank);
  std::vector<int32_t> vx_strides(rank);

  for (int64_t i = 0; i < rank; i++) {
    vx_begin[rank - i - 1] = static_cast<int32_t>(slice->slice_starts(i));
    vx_end[rank - i - 1] = static_cast<int32_t>(slice->slice_limits(i));
    vx_strides[rank - i - 1] = static_cast<int32_t>(slice->slice_strides(i));
  }

  auto vx_slice_op = vx_graph_->CreateOperation<tim::vx::ops::StridedSlice>(
      vx_begin, vx_end, vx_strides, 0, 0, 0);
  vx_slice_op->BindInput(input_tensor);
  vx_slice_op->BindOutput(output_tensor);

  return tsl::OkStatus();
}

tsl::Status VsiGraphBuilder::HandleConcatenate(
    const xla::HloInstruction* concatenate) {
  std::vector<std::shared_ptr<tim::vx::Tensor>> input_tensors;
  for (const auto* input_operand : concatenate->operands()) {
    auto input_tensor = hlo_instr_to_vx_tensor_.at(input_operand);
    input_tensors.push_back(input_tensor);
  }
  auto output_tensor = CreateOpOutputTensor(concatenate);

  int64_t xla_axis = concatenate->dimensions(0);
  int64_t rank = concatenate->shape().rank();
  uint32_t vx_axis = static_cast<uint32_t>(rank - 1 - xla_axis);

  auto vx_concat_op = vx_graph_->CreateOperation<tim::vx::ops::Concat>(
      vx_axis, input_tensors.size());
  vx_concat_op->BindInputs(input_tensors);
  vx_concat_op->BindOutput(output_tensor);

  return tsl::OkStatus();
}

tsl::Status VsiGraphBuilder::HandleElementwiseUnary(
    const xla::HloInstruction* elementwise_unary_op) {
  std::shared_ptr<tim::vx::Operation> vx_elementwise_unary_op;
  switch (elementwise_unary_op->opcode()) {
    case xla::HloOpcode::kAbs:
      vx_elementwise_unary_op = vx_graph_->CreateOperation<tim::vx::ops::Abs>();
      break;
    case xla::HloOpcode::kNegate:
      vx_elementwise_unary_op = vx_graph_->CreateOperation<tim::vx::ops::Neg>();
      break;
    case xla::HloOpcode::kSign:
      vx_elementwise_unary_op =
          vx_graph_->CreateOperation<tim::vx::ops::Sign>();
      break;
    case xla::HloOpcode::kLogistic:
      vx_elementwise_unary_op =
          vx_graph_->CreateOperation<tim::vx::ops::Sigmoid>();
      break;
    case xla::HloOpcode::kTanh:
      vx_elementwise_unary_op =
          vx_graph_->CreateOperation<tim::vx::ops::Tanh>();
      break;
    case xla::HloOpcode::kExp:
      vx_elementwise_unary_op = vx_graph_->CreateOperation<tim::vx::ops::Exp>();
      break;
    case xla::HloOpcode::kLog:
      vx_elementwise_unary_op = vx_graph_->CreateOperation<tim::vx::ops::Log>();
      break;
    case xla::HloOpcode::kSqrt:
      vx_elementwise_unary_op =
          vx_graph_->CreateOperation<tim::vx::ops::Sqrt>();
      break;
    case xla::HloOpcode::kRsqrt:
      vx_elementwise_unary_op =
          vx_graph_->CreateOperation<tim::vx::ops::Rsqrt>();
      break;
    case xla::HloOpcode::kRoundNearestAfz:
      vx_elementwise_unary_op =
          vx_graph_->CreateOperation<tim::vx::ops::Round>();
      break;
    case xla::HloOpcode::kCeil:
      vx_elementwise_unary_op =
          vx_graph_->CreateOperation<tim::vx::ops::Ceil>();
      break;
    case xla::HloOpcode::kFloor:
      vx_elementwise_unary_op =
          vx_graph_->CreateOperation<tim::vx::ops::Floor>();
      break;
    case xla::HloOpcode::kSin:
      vx_elementwise_unary_op = vx_graph_->CreateOperation<tim::vx::ops::Sin>();
      break;
    case xla::HloOpcode::kNot:
      vx_elementwise_unary_op =
          vx_graph_->CreateOperation<tim::vx::ops::LogicalNot>();
      break;
    default:
      return DefaultAction(elementwise_unary_op);
  }

  const auto* input_operand = elementwise_unary_op->operand(0);
  auto input_tensor = hlo_instr_to_vx_tensor_.at(input_operand);
  auto output_tensor = CreateOpOutputTensor(elementwise_unary_op);
  vx_elementwise_unary_op->BindInput(input_tensor);
  vx_elementwise_unary_op->BindOutput(output_tensor);

  return absl::OkStatus();
}

tsl::Status VsiGraphBuilder::HandleElementwiseBinary(
    const xla::HloInstruction* elementwise_binary_op) {
  std::shared_ptr<tim::vx::Operation> vx_elementwise_binary_op;
  switch (elementwise_binary_op->opcode()) {
    case xla::HloOpcode::kAdd:
      vx_elementwise_binary_op =
          vx_graph_->CreateOperation<tim::vx::ops::Add>();
      break;
    case xla::HloOpcode::kSubtract:
      vx_elementwise_binary_op =
          vx_graph_->CreateOperation<tim::vx::ops::Sub>();
      break;
    case xla::HloOpcode::kMultiply:
      vx_elementwise_binary_op =
          vx_graph_->CreateOperation<tim::vx::ops::Multiply>();
      break;
    case xla::HloOpcode::kDivide:
      vx_elementwise_binary_op =
          vx_graph_->CreateOperation<tim::vx::ops::Div>();
      break;
    case xla::HloOpcode::kPower:
      vx_elementwise_binary_op =
          vx_graph_->CreateOperation<tim::vx::ops::Pow>();
      break;
    case xla::HloOpcode::kMinimum:
      vx_elementwise_binary_op =
          vx_graph_->CreateOperation<tim::vx::ops::Minimum>();
      break;
    case xla::HloOpcode::kMaximum:
      vx_elementwise_binary_op =
          vx_graph_->CreateOperation<tim::vx::ops::Maximum>();
      break;
    case xla::HloOpcode::kAnd:
      vx_elementwise_binary_op =
          vx_graph_->CreateOperation<tim::vx::ops::LogicalAnd>();
      break;
    case xla::HloOpcode::kOr:
      vx_elementwise_binary_op =
          vx_graph_->CreateOperation<tim::vx::ops::LogicalOr>();
      break;
    default:
      return DefaultAction(elementwise_binary_op);
  }

  const auto* lhs_operand = elementwise_binary_op->operand(0);
  const auto* rhs_operand = elementwise_binary_op->operand(1);
  auto lhs_tensor = hlo_instr_to_vx_tensor_.at(lhs_operand);
  auto rhs_tensor = hlo_instr_to_vx_tensor_.at(rhs_operand);
  auto output_tensor = CreateOpOutputTensor(elementwise_binary_op);

  vx_elementwise_binary_op->BindInputs({lhs_tensor, rhs_tensor});
  vx_elementwise_binary_op->BindOutput(output_tensor);

  return absl::OkStatus();
}

tsl::Status VsiGraphBuilder::HandleCompare(const xla::HloInstruction* compare) {
  std::shared_ptr<tim::vx::Operation> vx_compare_op;
  switch (compare->comparison_direction()) {
    case xla::ComparisonDirection::kEq:
      vx_compare_op = vx_graph_->CreateOperation<tim::vx::ops::Equal>();
      break;
    case xla::ComparisonDirection::kNe:
      vx_compare_op = vx_graph_->CreateOperation<tim::vx::ops::NotEqual>();
      break;
    case xla::ComparisonDirection::kGt:
      vx_compare_op = vx_graph_->CreateOperation<tim::vx::ops::Greater>();
      break;
    case xla::ComparisonDirection::kGe:
      vx_compare_op =
          vx_graph_->CreateOperation<tim::vx::ops::GreaterOrEqual>();
      break;
    case xla::ComparisonDirection::kLt:
      vx_compare_op = vx_graph_->CreateOperation<tim::vx::ops::Less>();
      break;
    case xla::ComparisonDirection::kLe:
      vx_compare_op = vx_graph_->CreateOperation<tim::vx::ops::LessOrEqual>();
      break;
    default:
      return tsl::errors::Unimplemented(
          "Unsupported comparision direction: %s",
          xla::ComparisonDirectionToString(compare->comparison_direction()));
  }

  const auto* lhs_operand = compare->operand(0);
  const auto* rhs_operand = compare->operand(1);
  auto lhs_tensor = hlo_instr_to_vx_tensor_.at(lhs_operand);
  auto rhs_tensor = hlo_instr_to_vx_tensor_.at(rhs_operand);
  auto output_tensor = CreateOpOutputTensor(compare);

  vx_compare_op->BindInputs({lhs_tensor, rhs_tensor});
  vx_compare_op->BindOutput(output_tensor);

  return tsl::OkStatus();
}

tsl::Status VsiGraphBuilder::HandleSelect(const xla::HloInstruction* select) {
  const auto* pred_operand = select->operand(0);
  const auto* on_true_operand = select->operand(1);
  const auto* on_false_operand = select->operand(2);
  auto pred_tensor = hlo_instr_to_vx_tensor_.at(pred_operand);
  auto on_true_tensor = hlo_instr_to_vx_tensor_.at(on_true_operand);
  auto on_false_tensor = hlo_instr_to_vx_tensor_.at(on_false_operand);
  auto output_tensor = CreateOpOutputTensor(select);

  auto vx_select_op = vx_graph_->CreateOperation<tim::vx::ops::Select>();
  vx_select_op->BindInputs({pred_tensor, on_true_tensor, on_false_tensor});
  vx_select_op->BindOutput(output_tensor);

  return tsl::OkStatus();
}

tsl::Status VsiGraphBuilder::HandleGather(const xla::HloInstruction* gather) {
  const auto* input_operand = gather->operand(0);
  const auto* indices_operand = gather->operand(1);
  auto input_tensor = hlo_instr_to_vx_tensor_.at(input_operand);
  auto indices_tensor = hlo_instr_to_vx_tensor_.at(indices_operand);
  auto output_tensor = CreateOpOutputTensor(gather);

  const auto& dim_nums = gather->gather_dimension_numbers();
  // Only supports simple gather.
  CHECK_EQ(dim_nums.index_vector_dim(), 1);
  CHECK_EQ(dim_nums.collapsed_slice_dims_size(), 0);

  int64_t rank = input_operand->shape().rank();
  int64_t xla_axis = dim_nums.start_index_map(0);
  int32_t vx_axis = static_cast<int32_t>(rank - 1 - xla_axis);

  auto vx_gather_op = vx_graph_->CreateOperation<tim::vx::ops::Gather>(vx_axis);
  vx_gather_op->BindInputs({input_tensor, indices_tensor});
  vx_gather_op->BindOutput(output_tensor);

  return tsl::OkStatus();
}

tsl::Status VsiGraphBuilder::HandleConvolution(
    const xla::HloInstruction* convolution) {
  const auto& dim_nums = convolution->convolution_dimension_numbers();
  const auto& window = convolution->window();

  // Currently only supports Conv2d.
  CHECK_EQ(dim_nums.input_spatial_dimensions_size(), 2);
  CHECK_EQ(dim_nums.kernel_spatial_dimensions_size(), 2);
  CHECK_EQ(dim_nums.output_spatial_dimensions_size(), 2);
  CHECK_EQ(window.dimensions(0).base_dilation(), 1);
  CHECK_EQ(window.dimensions(1).base_dilation(), 1);

  return HandleConv2d(convolution);
}

tsl::Status VsiGraphBuilder::HandleConv2d(const xla::HloInstruction* conv2d) {
  const auto* lhs_operand = conv2d->operand(0);
  const auto* rhs_operand = conv2d->operand(1);
  auto lhs_tensor = hlo_instr_to_vx_tensor_.at(lhs_operand);
  auto rhs_tensor = hlo_instr_to_vx_tensor_.at(rhs_operand);
  auto output_tensor = CreateOpOutputTensor(conv2d);

  const auto& dim_nums = conv2d->convolution_dimension_numbers();
  const auto& window = conv2d->window();

  // Assume NCHW input/output and OIHW kernel layout.
  CHECK_EQ(dim_nums.input_batch_dimension(), 0);
  CHECK_EQ(dim_nums.input_feature_dimension(), 1);
  CHECK_EQ(dim_nums.kernel_output_feature_dimension(), 0);
  CHECK_EQ(dim_nums.kernel_input_feature_dimension(), 1);
  CHECK_EQ(dim_nums.output_batch_dimension(), 0);
  CHECK_EQ(dim_nums.output_feature_dimension(), 1);

  int64_t num_groups = conv2d->feature_group_count();
  const auto& xla_kernel_shape = rhs_operand->shape();
  int32_t vx_weights =
      xla_kernel_shape.dimensions(dim_nums.kernel_output_feature_dimension());
  int32_t vx_multiplier = (num_groups > 1) ? vx_weights / num_groups : 0;

  std::array<uint32_t, 2> vx_kernel_size = {
      static_cast<uint32_t>(window.dimensions(1).size()),
      static_cast<uint32_t>(window.dimensions(0).size()),
  };
  std::array<uint32_t, 4> vx_pad = {
      static_cast<uint32_t>(window.dimensions(1).padding_low()),
      static_cast<uint32_t>(window.dimensions(1).padding_high()),
      static_cast<uint32_t>(window.dimensions(0).padding_low()),
      static_cast<uint32_t>(window.dimensions(0).padding_high()),
  };
  std::array<uint32_t, 2> vx_strides = {
      static_cast<uint32_t>(window.dimensions(1).stride()),
      static_cast<uint32_t>(window.dimensions(0).stride()),
  };
  std::array<uint32_t, 2> vx_dilation = {
      static_cast<uint32_t>(window.dimensions(1).window_dilation()),
      static_cast<uint32_t>(window.dimensions(0).window_dilation()),
  };

  std::shared_ptr<tim::vx::Operation> vx_conv2d_op;
  if (num_groups > 1 && num_groups != vx_weights) {
    vx_conv2d_op = vx_graph_->CreateOperation<tim::vx::ops::GroupedConv2d>(
        vx_pad, vx_strides, vx_dilation, num_groups);
  } else {
    vx_conv2d_op = vx_graph_->CreateOperation<tim::vx::ops::Conv2d>(
        vx_weights, tim::vx::PadType::AUTO, vx_kernel_size, vx_strides,
        vx_dilation, vx_pad, vx_multiplier);
  }

  vx_conv2d_op->BindInputs({lhs_tensor, rhs_tensor});
  vx_conv2d_op->BindOutput(output_tensor);

  return tsl::OkStatus();
}

tsl::Status VsiGraphBuilder::HandleTransposedConv2d(
    const xla::HloInstruction* transposed_conv2d) {
  return DefaultAction(transposed_conv2d);
}

tsl::Status VsiGraphBuilder::HandleDot(const xla::HloInstruction* dot) {
  const auto* lhs_operand = dot->operand(0);
  const auto* rhs_operand = dot->operand(1);
  auto lhs_tensor = hlo_instr_to_vx_tensor_.at(lhs_operand);
  auto rhs_tensor = hlo_instr_to_vx_tensor_.at(rhs_operand);
  auto output_tensor = CreateOpOutputTensor(dot);

  const auto& dim_nums = dot->dot_dimension_numbers();

  // LHS and RHS must have the same batch size.
  CHECK_EQ(dim_nums.lhs_batch_dimensions_size(),
           dim_nums.rhs_batch_dimensions_size());
  // Only supports one contracting dimension.
  CHECK_EQ(dim_nums.lhs_contracting_dimensions_size(), 1);
  CHECK_EQ(dim_nums.rhs_contracting_dimensions_size(), 1);

  bool transpose_a = (dim_nums.lhs_contracting_dimensions(0) ==
                      dim_nums.lhs_batch_dimensions_size());
  bool transpose_b = (dim_nums.rhs_contracting_dimensions(0) ==
                      dim_nums.rhs_batch_dimensions_size() + 1);

  auto vx_matmul_op = vx_graph_->CreateOperation<tim::vx::ops::Matmul>(
      transpose_a, transpose_b);
  vx_matmul_op->BindInputs({lhs_tensor, rhs_tensor});
  vx_matmul_op->BindOutput(output_tensor);

  return tsl::OkStatus();
}

tsl::Status VsiGraphBuilder::HandleReduceWindow(
    const xla::HloInstruction* reduce_window) {
  const auto& window = reduce_window->window();
  auto reduce_op_code = reduce_window->to_apply()->root_instruction()->opcode();

  // Currently only supports MaxPool2d.
  CHECK_EQ(window.dimensions_size(), 4);
  CHECK_EQ(reduce_op_code, xla::HloOpcode::kMaximum);

  const auto* input_operand = reduce_window->operand(0);
  auto input_tensor = hlo_instr_to_vx_tensor_.at(input_operand);
  auto output_tensor = CreateOpOutputTensor(reduce_window);

  auto vx_pool_type = tim::vx::PoolType::MAX;
  std::array<uint32_t, 2> vx_pool_size = {
      static_cast<uint32_t>(window.dimensions(3).size()),
      static_cast<uint32_t>(window.dimensions(2).size()),
  };
  std::array<uint32_t, 4> vx_pad = {
      static_cast<uint32_t>(window.dimensions(3).padding_low()),
      static_cast<uint32_t>(window.dimensions(3).padding_high()),
      static_cast<uint32_t>(window.dimensions(2).padding_low()),
      static_cast<uint32_t>(window.dimensions(2).padding_high()),
  };
  std::array<uint32_t, 2> vx_strides = {
      static_cast<uint32_t>(window.dimensions(3).stride()),
      static_cast<uint32_t>(window.dimensions(2).stride()),
  };

  auto vx_pool2d_op = vx_graph_->CreateOperation<tim::vx::ops::Pool2d>(
      vx_pool_type, vx_pad, vx_pool_size, vx_strides);
  vx_pool2d_op->BindInput(input_tensor);
  vx_pool2d_op->BindOutput(output_tensor);

  return absl::OkStatus();
}

tsl::Status VsiGraphBuilder::HandleReduce(const xla::HloInstruction* reduce) {
  auto reduce_op_code = reduce->to_apply()->root_instruction()->opcode();

  const auto* input_operand = reduce->operand(0);
  int64_t input_rank = input_operand->shape().rank();
  auto xla_dims = reduce->dimensions();
  std::vector<int32_t> vx_axes(xla_dims.size());
  for (size_t i = 0; i < xla_dims.size(); i++) {
    vx_axes[i] = static_cast<int32_t>(input_rank - 1 - xla_dims[i]);
  }

  std::shared_ptr<tim::vx::Operation> vx_reduce_op;
  switch (reduce_op_code) {
    case xla::HloOpcode::kAdd:
      vx_reduce_op =
          vx_graph_->CreateOperation<tim::vx::ops::ReduceSum>(vx_axes, false);
      break;
    case xla::HloOpcode::kMultiply:
      vx_reduce_op =
          vx_graph_->CreateOperation<tim::vx::ops::ReduceProd>(vx_axes, false);
      break;
    case xla::HloOpcode::kMinimum:
      vx_reduce_op =
          vx_graph_->CreateOperation<tim::vx::ops::ReduceMin>(vx_axes, false);
      break;
    case xla::HloOpcode::kMaximum:
      vx_reduce_op =
          vx_graph_->CreateOperation<tim::vx::ops::ReduceMax>(vx_axes, false);
      break;
    case xla::HloOpcode::kAnd:
      vx_reduce_op =
          vx_graph_->CreateOperation<tim::vx::ops::ReduceAll>(vx_axes, false);
      break;
    case xla::HloOpcode::kOr:
      vx_reduce_op =
          vx_graph_->CreateOperation<tim::vx::ops::ReduceAny>(vx_axes, false);
      break;
    default:
      return tsl::errors::Unimplemented("Reduce opcode: %s is not supported",
                                        xla::HloOpcodeString(reduce_op_code));
  }

  auto input_tensor = hlo_instr_to_vx_tensor_.at(input_operand);
  auto output_tensor = CreateOpOutputTensor(reduce);
  vx_reduce_op->BindInput(input_tensor);
  vx_reduce_op->BindOutput(output_tensor);

  return tsl::OkStatus();
}

tsl::Status VsiGraphBuilder::HandlePad(const xla::HloInstruction* pad) {
  const auto* input_operand = pad->operand(0);
  const auto* pad_val_constant = pad->operand(1);
  auto input_tensor = hlo_instr_to_vx_tensor_.at(input_operand);
  auto output_tensor = CreateOpOutputTensor(pad);

  const auto& pad_val_literal = pad_val_constant->literal();
  auto pad_val_dtype = pad_val_constant->shape().element_type();

  int32_t pad_val_int32 = 0;
  float pad_val_fp32 = 0.0F;

  switch (pad_val_dtype) {
    case xla::PrimitiveType::F32:
      pad_val_fp32 = pad_val_literal.data<float>()[0];
      break;
    case xla::PrimitiveType::F16:
      pad_val_fp32 = pad_val_literal.data<Eigen::half>()[0];
      break;
    case xla::PrimitiveType::U8:
      pad_val_int32 = static_cast<int32_t>(pad_val_literal.data<uint8_t>()[0]);
      break;
    case xla::PrimitiveType::S8:
      pad_val_int32 = static_cast<int32_t>(pad_val_literal.data<int8_t>()[0]);
      break;
    case xla::PrimitiveType::U16:
      pad_val_int32 = static_cast<int32_t>(pad_val_literal.data<uint16_t>()[0]);
      break;
    case xla::PrimitiveType::S16:
      pad_val_int32 = static_cast<int32_t>(pad_val_literal.data<int16_t>()[0]);
      break;
    case xla::PrimitiveType::U32:
      pad_val_int32 = static_cast<int32_t>(pad_val_literal.data<uint32_t>()[0]);
      break;
    case xla::PrimitiveType::S32:
      pad_val_int32 = static_cast<int32_t>(pad_val_literal.data<int32_t>()[0]);
      break;
    default:
      return tsl::errors::Unimplemented("Unsupported pad value dtype: %s",
                                        xla::PrimitiveType_Name(pad_val_dtype));
  }

  int64_t rank = input_operand->shape().rank();
  const auto& padding_config = pad->padding_config();
  std::vector<uint32_t> vx_pad_low(rank);
  std::vector<uint32_t> vx_pad_high(rank);
  for (int64_t i = 0; i < rank; i++) {
    vx_pad_low[rank - 1 - i] =
        static_cast<uint32_t>(padding_config.dimensions(i).edge_padding_low());
    vx_pad_high[rank - 1 - i] =
        static_cast<uint32_t>(padding_config.dimensions(i).edge_padding_high());
  }

  std::shared_ptr<tim::vx::Operation> vx_pad_op;
  if (pad_val_fp32 != 0.0F) {
    vx_pad_op = vx_graph_->CreateOperation<tim::vx::ops::PadV2>(
        vx_pad_low, vx_pad_high, pad_val_fp32);
  } else {
    vx_pad_op = vx_graph_->CreateOperation<tim::vx::ops::Pad>(
        vx_pad_low, vx_pad_high, pad_val_int32);
  }

  vx_pad_op->BindInput(input_tensor);
  vx_pad_op->BindOutput(output_tensor);

  return tsl::OkStatus();
}

tsl::Status VsiGraphBuilder::HandleBatchNormInference(
    const xla::HloInstruction* batch_norm_inference) {
  const auto* input_operand = batch_norm_inference->operand(0);
  const auto* scale_operand = batch_norm_inference->operand(1);
  const auto* offset_operand = batch_norm_inference->operand(2);
  const auto* mean_operand = batch_norm_inference->operand(3);
  const auto* variance_operand = batch_norm_inference->operand(4);
  auto input_tensor = hlo_instr_to_vx_tensor_.at(input_operand);
  auto scale_tensor = hlo_instr_to_vx_tensor_.at(scale_operand);
  auto offset_tensor = hlo_instr_to_vx_tensor_.at(offset_operand);
  auto mean_tensor = hlo_instr_to_vx_tensor_.at(mean_operand);
  auto variance_tensor = hlo_instr_to_vx_tensor_.at(variance_operand);
  auto output_tensor = CreateOpOutputTensor(batch_norm_inference);

  // Currently only supports NCHW input.
  CHECK_EQ(batch_norm_inference->feature_index(), 1);
  float epsilon = batch_norm_inference->epsilon();

  auto vx_batch_norm_op =
      vx_graph_->CreateOperation<tim::vx::ops::BatchNorm>(epsilon);
  vx_batch_norm_op->BindInputs({input_tensor, mean_tensor, variance_tensor,
                                scale_tensor, offset_tensor});
  vx_batch_norm_op->BindOutput(output_tensor);
  return tsl::OkStatus();
}

tsl::Status VsiGraphBuilder::HandleClamp(const xla::HloInstruction* clamp) {
  const auto* min_operand = clamp->operand(0);
  const auto* input_operand = clamp->operand(1);
  const auto* max_operand = clamp->operand(2);
  auto input_tensor = hlo_instr_to_vx_tensor_.at(input_operand);
  auto min_tensor = hlo_instr_to_vx_tensor_.at(min_operand);
  auto max_tensor = hlo_instr_to_vx_tensor_.at(max_operand);
  auto temp_tensor =
      vx_graph_->CreateTensor(input_tensor->GetSpec().AsTransientSpec());
  auto output_tensor = CreateOpOutputTensor(clamp);

  auto vx_maximum_op = vx_graph_->CreateOperation<tim::vx::ops::Maximum>();
  vx_maximum_op->BindInputs({input_tensor, min_tensor});
  vx_maximum_op->BindOutput(temp_tensor);

  auto vx_minimum_op = vx_graph_->CreateOperation<tim::vx::ops::Minimum>();
  vx_minimum_op->BindInputs({temp_tensor, max_tensor});
  vx_minimum_op->BindOutput(output_tensor);

  return tsl::OkStatus();
}

tsl::Status VsiGraphBuilder::HandleCustomCall(
    const xla::HloInstruction* custom_call) {
  const auto& target = custom_call->custom_call_target();
  if (target == "stablehlo.uniform_quantize") {
    return HandleQuantize(custom_call);
  }

  if (target == "stablehlo.uniform_dequantize") {
    return HandleDequantize(custom_call);
  }

  return DefaultAction(custom_call);
}

tsl::Status VsiGraphBuilder::HandleQuantize(
    const xla::HloInstruction* quantize) {
  const auto& backend_config = quantize->raw_backend_config_string();
  mlir::MLIRContext mlir_context;
  auto quant_attr = mlir::parseAttribute(backend_config, &mlir_context)
                        .dyn_cast<mlir::DictionaryAttr>();
  if (!quant_attr) {
    return tsl::errors::Internal(
        "Couldn't parse backend config into a dictionary attribute");
  }

  int64_t rank = quantize->shape().rank();
  auto vx_quant = utils::ConvertQuantParams(quant_attr, rank);

  const auto* input_operand = quantize->operand(0);
  auto input_tensor = hlo_instr_to_vx_tensor_.at(input_operand);
  auto output_tensor = CreateOpOutputTensor(quantize);
  output_tensor->GetSpec().SetQuantization(vx_quant);

  auto vx_quantize_op = vx_graph_->CreateOperation<tim::vx::ops::DataConvert>();
  vx_quantize_op->BindInput(input_tensor);
  vx_quantize_op->BindOutput(output_tensor);

  return tsl::OkStatus();
}

tsl::Status VsiGraphBuilder::HandleDequantize(
    const xla::HloInstruction* dequantize) {
  const auto& backend_config = dequantize->raw_backend_config_string();
  mlir::MLIRContext mlir_context;
  auto quant_attr = mlir::parseAttribute(backend_config, &mlir_context)
                        .dyn_cast<mlir::DictionaryAttr>();
  if (!quant_attr) {
    return tsl::errors::Internal(
        "Couldn't parse backend config into a dictionary attribute");
  }

  int64_t rank = dequantize->shape().rank();
  auto vx_quant = utils::ConvertQuantParams(quant_attr, rank);

  const auto* input_operand = dequantize->operand(0);
  auto input_tensor = hlo_instr_to_vx_tensor_.at(input_operand);
  auto output_tensor = CreateOpOutputTensor(dequantize);
  input_tensor->GetSpec().SetQuantization(vx_quant);

  auto vx_dequantize_op =
      vx_graph_->CreateOperation<tim::vx::ops::DataConvert>();
  vx_dequantize_op->BindInput(input_tensor);
  vx_dequantize_op->BindOutput(output_tensor);

  return tsl::OkStatus();
}

std::vector<std::shared_ptr<tim::vx::Tensor>>
VsiGraphBuilder::CreateOpOutputTensors(const xla::HloInstruction* hlo_instr) {
  CHECK(hlo_instr->shape().IsTuple())
      << "Expect HLO instruction with tuple shape";

  std::vector<std::shared_ptr<tim::vx::Tensor>> vx_tensors;
  for (const auto& item_shape : hlo_instr->shape().tuple_shapes()) {
    auto tensor_attr = hlo_instr->IsRoot()
                           ? tim::vx::TensorAttribute::OUTPUT
                           : tim::vx::TensorAttribute::TRANSIENT;
    auto vx_spec = utils::ConvertXlaShape(item_shape, tensor_attr);
    auto vx_tensor = vx_graph_->CreateTensor(vx_spec);
    vx_tensors.push_back(std::move(vx_tensor));
  }

  hlo_instr_to_vx_tensors_.insert({hlo_instr, vx_tensors});
  return vx_tensors;
}

std::shared_ptr<tim::vx::Tensor> VsiGraphBuilder::CreateOpOutputTensor(
    const xla::HloInstruction* hlo_instr) {
  constexpr auto kIsGraphOutput = [](const xla::HloInstruction* hlo_instr) {
    if (hlo_instr->IsRoot()) {
      return true;
    }
    for (const auto* user : hlo_instr->users()) {
      if (user->opcode() == xla::HloOpcode::kTuple && user->IsRoot()) {
        return true;
      }
    }
    return false;
  };

  auto tensor_attr = kIsGraphOutput(hlo_instr)
                         ? tim::vx::TensorAttribute::OUTPUT
                         : tim::vx::TensorAttribute::TRANSIENT;
  auto vx_spec = utils::ConvertXlaShape(hlo_instr->shape(), tensor_attr);
  auto vx_tensor = vx_graph_->CreateTensor(vx_spec);

  hlo_instr_to_vx_tensor_.insert({hlo_instr, vx_tensor});
  return vx_tensor;
}

}  // namespace vsi
}  // namespace stream_executor