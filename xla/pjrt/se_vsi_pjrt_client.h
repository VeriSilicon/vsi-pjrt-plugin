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

#ifndef XLA_PJRT_SE_vsi_PJRT_CLIENT_H_
#define XLA_PJRT_SE_vsi_PJRT_CLIENT_H_

#include <memory>

#include "xla/pjrt/pjrt_stream_executor_client.h"
#include "xla/statusor.h"

namespace xla {

class StreamExecutorVsiDevice final : public PjRtStreamExecutorDevice {
 public:
  StreamExecutorVsiDevice(int id,
                          std::unique_ptr<LocalDeviceState> local_device_state);
};

class PjRtStreamExecutorVsiClient final : public PjRtStreamExecutorClient {
  using PjRtStreamExecutorClient::PjRtStreamExecutorClient;

  StatusOr<std::unique_ptr<PjRtLoadedExecutable>> Compile(
      mlir::ModuleOp mlir_module, CompileOptions options) override;
};

StatusOr<std::unique_ptr<PjRtClient>> GetStreamExecutorVsiClient();

}  // namespace xla

#endif  // XLA_PJRT_SE_vsi_PJRT_CLIENT_H_
