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

#include "xla/stream_executor/vsi/vsi_platform.h"

#include <memory>

#include "xla/stream_executor/platform/initialize.h"
#include "xla/stream_executor/vsi/vsi_executor.h"

namespace stream_executor {
namespace vsi {

VsiPlatform::VsiPlatform() { vx_context_ = tim::vx::Context::Create(); }

VsiPlatform::~VsiPlatform() = default;

Platform::Id VsiPlatform::id() const { return id_; }

int VsiPlatform::VisibleDeviceCount() const { return 1; }

const std::string& VsiPlatform::Name() const { return name_; }

tsl::StatusOr<std::unique_ptr<DeviceDescription>>
VsiPlatform::DescriptionForDevice(int ordinal) const {
  return tsl::errors::Unimplemented(
      "VsiPlatform::DescriptionForDevice not implemented.");
}

tsl::StatusOr<StreamExecutor*> VsiPlatform::ExecutorForDevice(int ordinal) {
  StreamExecutorConfig config;
  config.ordinal = ordinal;
  config.device_options = DeviceOptions::Default();
  return GetExecutor(config);
}

tsl::StatusOr<StreamExecutor*> VsiPlatform::GetExecutor(
    const StreamExecutorConfig& config) {
  return executor_cache_.GetOrCreate(
      config, [&]() { return GetUncachedExecutor(config); });
}

tsl::StatusOr<std::unique_ptr<StreamExecutor>> VsiPlatform::GetUncachedExecutor(
    const StreamExecutorConfig& config) {
  auto vsi_executor =
      std::make_unique<VsiExecutor>(vx_context_, config.ordinal);
  auto executor = std::make_unique<StreamExecutor>(
      this, std::move(vsi_executor), config.ordinal);
  auto init_status = executor->Init(config.device_options);
  if (!init_status.ok()) {
    return tsl::Status{
        absl::StatusCode::kInternal,
        absl::StrFormat(
            "Failed to initialize StreamExecutor for device ordinal %d: %s",
            config.ordinal, init_status.ToString())};
  }

  return std::move(executor);
}

static void InitializeVsiPlatform() {
  // Disabling leak checking, MultiPlatformManager does not destroy its
  // registered platforms.
  auto status = MultiPlatformManager::PlatformWithName("vsi");
  if (!status.ok()) {
    std::unique_ptr<VsiPlatform> platform(new VsiPlatform);
    TF_CHECK_OK(MultiPlatformManager::RegisterPlatform(std::move(platform)));
  }
}

}  // namespace vsi
}  // namespace stream_executor

REGISTER_MODULE_INITIALIZER(vsi_platform,
                            stream_executor::vsi::InitializeVsiPlatform());
