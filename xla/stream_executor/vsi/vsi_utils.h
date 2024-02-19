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

#ifndef XLA_STREAM_EXECUTOR_VSI_VSI_UTILS_H_
#define XLA_STREAM_EXECUTOR_VSI_VSI_UTILS_H_

#include "tim/vx/tensor.h"
#include "tim/vx/types.h"
#include "xla/shape.h"

namespace stream_executor {
namespace vsi {
namespace utils {

inline tim::vx::DataType ConvertXlaDataType(xla::PrimitiveType dtype) {
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

inline tim::vx::TensorSpec ConvertXlaShape(
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

}  // namespace utils
}  // namespace vsi
}  // namespace stream_executor

#endif