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

#ifndef XLA_STREAM_EXECUTOR_VSI_VSI_PLATFORM_H_
#define XLA_STREAM_EXECUTOR_VSI_VSI_PLATFORM_H_

#include <memory>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "tim/vx/context.h"
#include "tsl/platform/statusor.h"
#include "xla/stream_executor/executor_cache.h"
#include "xla/stream_executor/multi_platform_manager.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform/port.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/stream_executor_internal.h"
#include "xla/stream_executor/vsi/vsi_platform_id.h"

namespace stream_executor {
namespace vsi {

class VsiPlatform final : public Platform {
 public:
  VsiPlatform();
  ~VsiPlatform() override;

  Platform::Id id() const override;

  int VisibleDeviceCount() const override;

  const std::string& Name() const override;

  tsl::StatusOr<std::unique_ptr<DeviceDescription>> DescriptionForDevice(
      int ordinal) const override;

  tsl::StatusOr<StreamExecutor*> ExecutorForDevice(int ordinal) override;

  tsl::StatusOr<StreamExecutor*> GetExecutor(
      const StreamExecutorConfig& config) override;

  tsl::StatusOr<std::unique_ptr<StreamExecutor>> GetUncachedExecutor(
      const StreamExecutorConfig& config) override;

  std::shared_ptr<tim::vx::Context> GetContext() { return vx_context_; }

 private:
  // This platform's name.
  std::string name_ = "vsi";
  // This platform's id.
  Platform::Id id_ = kVsiPlatformId;

  // Cache of created StreamExecutors.
  ExecutorCache executor_cache_;

  std::shared_ptr<tim::vx::Context> vx_context_;

  SE_DISALLOW_COPY_AND_ASSIGN(VsiPlatform);
};

}  // namespace vsi
}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_VSI_VSI_PLATFORM_H_
