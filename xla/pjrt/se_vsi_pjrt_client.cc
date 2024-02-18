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

#include "xla/pjrt/se_vsi_pjrt_client.h"

#include <memory>
#include <string_view>
#include <utility>
#include <vector>

#include "tsl/platform/errors.h"
#include "xla/client/client_library.h"
#include "xla/pjrt/pjrt_stream_executor_client.h"
#include "xla/service/platform_util.h"

namespace xla {

static const char kVsiPlatformName[] = "vsi";

StreamExecutorVsiDevice::StreamExecutorVsiDevice(
    int id, std::unique_ptr<LocalDeviceState> local_device_state)
    : PjRtStreamExecutorDevice(id, std::move(local_device_state),
                               /*device_kind=*/kVsiPlatformName) {}

StatusOr<std::unique_ptr<PjRtLoadedExecutable>>
PjRtStreamExecutorVsiClient::Compile(mlir::ModuleOp mlir_module,
                                     CompileOptions options) {
  return PjRtStreamExecutorClient::Compile(mlir_module, options);
}

StatusOr<std::unique_ptr<PjRtClient>> GetStreamExecutorVsiClient() {
  TF_ASSIGN_OR_RETURN(se::Platform * platform,
                      PlatformUtil::GetPlatform(kVsiPlatformName));
  if (platform->VisibleDeviceCount() != 1) {
    return FailedPrecondition("%s platform must have exactly one device.",
                              kVsiPlatformName);
  }
  LocalClientOptions options;
  options.set_platform(platform);
  TF_ASSIGN_OR_RETURN(LocalClient * client,
                      ClientLibrary::GetOrCreateLocalClient(options));

  std::vector<std::unique_ptr<PjRtStreamExecutorDevice>> devices;
  se::StreamExecutor* executor = client->backend().stream_executor(0).value();
  auto device_state = std::make_unique<LocalDeviceState>(
      executor, client, LocalDeviceState::kSynchronous,
      /*max_inflight_computations=*/1,
      /*allow_event_reuse=*/false, /*use_callback_stream=*/false);
  auto device =
      std::make_unique<StreamExecutorVsiDevice>(0, std::move(device_state));
  devices.push_back(std::move(device));

  return std::unique_ptr<PjRtClient>(
      std::make_unique<PjRtStreamExecutorVsiClient>(
          kVsiPlatformName, client, std::move(devices), /*process_index=*/0,
          /*allocator=*/nullptr, /*host_memory_allocator=*/nullptr,
          /*should_stage_host_to_device_transfers=*/false,
          /*gpu_run_options=*/nullptr));
}

}  // namespace xla
