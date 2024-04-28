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

#ifndef XLA_STREAM_EXECUTOR_VSI_VSI_TRANSFER_MANAGER_H_
#define XLA_STREAM_EXECUTOR_VSI_VSI_TRANSFER_MANAGER_H_

#include "xla/service/generic_transfer_manager.h"

namespace stream_executor {
namespace vsi {

// An implementation of the XLA GenericTransferManager for vsi backend.
class VsiTransferManager final : public xla::GenericTransferManager {
 public:
  VsiTransferManager();
  ~VsiTransferManager() override = default;

  bool CanShapedBufferBeAccessedNow(
      StreamExecutor* executor,
      const xla::ShapedBuffer& device_buffer) const override {
    return true;
  }

  bool CanBufferBeAccessedNow(
      StreamExecutor* executor,
      const DeviceMemoryBase& device_buffer) const override {
    return true;
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(VsiTransferManager);
};

}  // namespace vsi
}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_VSI_VSI_TRANSFER_MANAGER_H_
