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

#ifndef XLA_STREAM_EXECUTOR_VSI_VSI_EXECUTOR_H_
#define XLA_STREAM_EXECUTOR_VSI_VSI_EXECUTOR_H_

#include "absl/synchronization/mutex.h"
#include "tim/vx/context.h"
#include "tsl/platform/status.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/stream_executor_internal.h"

namespace stream_executor {
namespace vsi {

class VsiExecutor final : public internal::StreamExecutorInterface {
 public:
  explicit VsiExecutor(std::shared_ptr<tim::vx::Context> vx_context,
                       int device_ordinal)
      : vx_context_(vx_context), device_ordinal_(device_ordinal) {}
  ~VsiExecutor() override = default;

  tsl::Status Init(int device_ordinal) override;

  int device_ordinal() const override { return device_ordinal_; }

  DeviceMemoryBase Allocate(uint64_t size, int64_t memory_space) override;

  void Deallocate(DeviceMemoryBase* mem) override;

  void* HostMemoryAllocate(uint64_t size) override;
  void HostMemoryDeallocate(void* mem) override;

  bool SynchronizeAllActivity() override;

  tsl::Status SynchronousMemZero(DeviceMemoryBase* location,
                                 uint64_t size) override;
  tsl::Status SynchronousMemSet(DeviceMemoryBase* location, int value,
                                uint64_t size) override;
  tsl::Status SynchronousMemcpy(DeviceMemoryBase* gpu_dst, const void* host_src,
                                uint64_t size) override;
  tsl::Status SynchronousMemcpy(void* host_dst, const DeviceMemoryBase& gpu_src,
                                uint64_t size) override;
  tsl::Status SynchronousMemcpyDeviceToDevice(DeviceMemoryBase* gpu_dst,
                                              const DeviceMemoryBase& gpu_src,
                                              uint64_t size) override;

  tsl::Status MemZero(Stream* stream, DeviceMemoryBase* location,
                      uint64_t size) override;

  tsl::Status Memset(Stream* stream, DeviceMemoryBase* location, uint8 pattern,
                     uint64_t size) override;
  tsl::Status Memset32(Stream* stream, DeviceMemoryBase* location,
                       uint32_t pattern, uint64_t size) override;
  tsl::Status Memcpy(Stream* stream, void* host_dst,
                     const DeviceMemoryBase& gpu_src, uint64_t size) override;
  tsl::Status Memcpy(Stream* stream, DeviceMemoryBase* gpu_dst,
                     const void* host_src, uint64_t size) override;
  bool MemcpyDeviceToDevice(Stream* stream, DeviceMemoryBase* gpu_dst,
                            const DeviceMemoryBase& gpu_src,
                            uint64_t size) override;

  bool HostMemoryRegister(void* mem, uint64_t size) override { return true; }
  bool HostMemoryUnregister(void* mem) override { return true; }
  bool HostCallback(Stream* stream,
                    absl::AnyInvocable<tsl::Status() &&> callback) override;
  tsl::Status AllocateEvent(Event* event) override;
  tsl::Status DeallocateEvent(Event* event) override;
  tsl::Status RecordEvent(Stream* stream, Event* event) override;
  tsl::Status WaitForEvent(Stream* stream, Event* event) override;
  Event::Status PollForEventStatus(Event* event) override;
  bool AllocateStream(Stream* stream) override;
  void DeallocateStream(Stream* stream) override;
  bool CreateStreamDependency(Stream* dependent, Stream* other) override;
  tsl::Status BlockHostUntilDone(Stream* stream) override;

  tsl::Status EnablePeerAccessTo(StreamExecutorInterface* other) override;
  bool CanEnablePeerAccessTo(StreamExecutorInterface* other) override;

  bool DeviceMemoryUsage(int64_t* free, int64_t* total) const override;

  // Creates a new DeviceDescription object. Ownership is transferred to the
  // caller.
  tsl::StatusOr<std::unique_ptr<DeviceDescription>> CreateDeviceDescription()
      const override;

  // Each call creates a new instance of the platform-specific implementation of
  // the corresponding interface type.
  std::unique_ptr<internal::EventInterface> CreateEventImplementation()
      override;
  std::unique_ptr<internal::StreamInterface> GetStreamImplementation() override;

  std::shared_ptr<tim::vx::Context> GetVxContext() { return vx_context_; }

 private:
  mutable absl::Mutex mu_;
  absl::once_flag compiled_;
  std::shared_ptr<tim::vx::Context> vx_context_;
  int device_ordinal_;

  TF_DISALLOW_COPY_AND_ASSIGN(VsiExecutor);
};

}  // namespace vsi
}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_VSI_VSI_EXECUTOR_H_