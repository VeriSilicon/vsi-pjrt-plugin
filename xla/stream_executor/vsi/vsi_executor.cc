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

#include "xla/stream_executor/vsi/vsi_executor.h"

#include "absl/synchronization/notification.h"
#include "tsl/platform/mem.h"
#include "xla/stream_executor/host/host_stream.h"
#include "xla/stream_executor/stream.h"

namespace stream_executor {
namespace vsi {

class HostEvent : public internal::EventInterface {
 public:
  HostEvent() : notification_(std::make_shared<absl::Notification>()) {}

  std::shared_ptr<absl::Notification>& GetNotification() {
    return notification_;
  }

 private:
  // We use a std::shared_ptr here because the client may delete the HostEvent
  // object while there are still RecordEvent and WaitForEvent callbacks pending
  // on a stream.
  std::shared_ptr<absl::Notification> notification_;
};

std::unique_ptr<internal::EventInterface>
VsiExecutor::CreateEventImplementation() {
  return std::unique_ptr<internal::EventInterface>(new HostEvent());
}

static HostEvent* AsHostEvent(Event* event) {
  DCHECK(event != nullptr);
  return static_cast<HostEvent*>(event->implementation());
}

static host::HostStream* AsHostStream(Stream* stream) {
  DCHECK(stream != nullptr);
  return dynamic_cast<host::HostStream*>(stream->implementation());
}

std::unique_ptr<internal::StreamInterface>
VsiExecutor::GetStreamImplementation() {
  return std::unique_ptr<internal::StreamInterface>(new host::HostStream(0));
}

tsl::Status VsiExecutor::Init(int device_ordinal,
                              DeviceOptions device_options) {
  return tsl::OkStatus();
}

DeviceMemoryBase VsiExecutor::Allocate(uint64_t size, int64_t memory_space) {
  void* data = tsl::port::AlignedMalloc(size, 64);
  return DeviceMemoryBase(data, size);
};

void VsiExecutor::Deallocate(DeviceMemoryBase* mem) {
  tsl::port::AlignedFree(mem->opaque());
}

void* VsiExecutor::HostMemoryAllocate(uint64_t size) {
  return tsl::port::Malloc(size);
}
void VsiExecutor::HostMemoryDeallocate(void* mem) { tsl::port::Free(mem); }

bool VsiExecutor::SynchronizeAllActivity() {
  // Not implemented for this platform for now.
  return true;
}

tsl::Status VsiExecutor::SynchronousMemZero(DeviceMemoryBase* location,
                                            uint64_t size) {
  memset(location->opaque(), 0, size);
  return tsl::OkStatus();
}
tsl::Status VsiExecutor::SynchronousMemSet(DeviceMemoryBase* location,
                                           int value, uint64_t size) {
  memset(location->opaque(), value, size);
  return tsl::OkStatus();
}
tsl::Status VsiExecutor::SynchronousMemcpy(DeviceMemoryBase* gpu_dst,
                                           const void* host_src,
                                           uint64_t size) {
  memcpy(gpu_dst->opaque(), host_src, size);
  return tsl::OkStatus();
}
tsl::Status VsiExecutor::SynchronousMemcpy(void* host_dst,
                                           const DeviceMemoryBase& gpu_src,
                                           uint64_t size) {
  memcpy(host_dst, gpu_src.opaque(), size);
  return tsl::OkStatus();
}
tsl::Status VsiExecutor::SynchronousMemcpyDeviceToDevice(
    DeviceMemoryBase* gpu_dst, const DeviceMemoryBase& gpu_src, uint64_t size) {
  memcpy(gpu_dst->opaque(), gpu_src.opaque(), size);
  return tsl::OkStatus();
}

tsl::Status VsiExecutor::MemZero(Stream* stream, DeviceMemoryBase* location,
                                 uint64_t size) {
  void* gpu_mem = location->opaque();
  // Enqueue the [asynchronous] memzero on the stream (HostStream) associated
  // with the HostExecutor.
  AsHostStream(stream)->EnqueueTask(
      [gpu_mem, size]() { memset(gpu_mem, 0, size); });
  return tsl::OkStatus();
}

tsl::Status VsiExecutor::Memset(Stream* stream, DeviceMemoryBase* location,
                                uint8 pattern, uint64_t size) {
  void* gpu_mem = location->opaque();
  // Enqueue the [asynchronous] memzero on the stream (HostStream) associated
  // with the HostExecutor.
  AsHostStream(stream)->EnqueueTask(
      [gpu_mem, size, pattern]() { memset(gpu_mem, pattern, size); });
  return tsl::OkStatus();
}

tsl::Status VsiExecutor::Memset32(Stream* stream, DeviceMemoryBase* location,
                                  uint32_t pattern, uint64_t size) {
  void* gpu_mem = location->opaque();
  // Enqueue the [asynchronous] memzero on the stream (HostStream) associated
  // with the HostExecutor.
  AsHostStream(stream)->EnqueueTask(
      [gpu_mem, size, pattern]() { memset(gpu_mem, pattern, size); });
  return tsl::OkStatus();
}

bool VsiExecutor::Memcpy(Stream* stream, void* host_dst,
                         const DeviceMemoryBase& gpu_src, uint64_t size) {
  // Enqueue the [asynchronous] memcpy on the stream (HostStream) associated
  // with the HostExecutor.
  void* src_mem = const_cast<void*>(gpu_src.opaque());
  AsHostStream(stream)->EnqueueTask(
      [host_dst, src_mem, size]() { memcpy(host_dst, src_mem, size); });
  return true;
}

bool VsiExecutor::Memcpy(Stream* stream, DeviceMemoryBase* gpu_dst,
                         const void* host_src, uint64_t size) {
  void* dst_mem = gpu_dst->opaque();
  // Enqueue the [asynchronous] memcpy on the stream (HostStream) associated
  // with the HostExecutor.
  AsHostStream(stream)->EnqueueTask(
      [dst_mem, host_src, size]() { memcpy(dst_mem, host_src, size); });
  return true;
}

bool VsiExecutor::MemcpyDeviceToDevice(Stream* stream,
                                       DeviceMemoryBase* gpu_dst,
                                       const DeviceMemoryBase& gpu_src,
                                       uint64_t size) {
  void* dst_mem = gpu_dst->opaque();
  void* src_mem = const_cast<void*>(gpu_src.opaque());
  // Enqueue this [asynchronous] "device-to-device" (i.e., host-to-host, given
  // the nature of the HostExecutor) memcpy  on the stream (HostStream)
  // associated with the HostExecutor.
  AsHostStream(stream)->EnqueueTask(
      [src_mem, dst_mem, size]() { memcpy(dst_mem, src_mem, size); });
  return true;
}

bool VsiExecutor::HostCallback(Stream* stream,
                               absl::AnyInvocable<tsl::Status() &&> callback) {
  AsHostStream(stream)->EnqueueTaskWithStatus(std::move(callback));
  return true;
}

tsl::Status VsiExecutor::AllocateEvent(Event* event) { return tsl::OkStatus(); }
tsl::Status VsiExecutor::DeallocateEvent(Event* event) {
  return tsl::OkStatus();
}
tsl::Status VsiExecutor::RecordEvent(Stream* stream, Event* event) {
  std::shared_ptr<absl::Notification> notification =
      AsHostEvent(event)->GetNotification();
  AsHostStream(stream)->EnqueueTask([notification]() {
    CHECK(!notification->HasBeenNotified());
    notification->Notify();
  });
  return tsl::OkStatus();
}

tsl::Status VsiExecutor::WaitForEvent(Stream* stream, Event* event) {
  std::shared_ptr<absl::Notification> notification =
      AsHostEvent(event)->GetNotification();
  AsHostStream(stream)->EnqueueTask(
      [notification]() { notification->WaitForNotification(); });
  return tsl::OkStatus();
}

Event::Status VsiExecutor::PollForEventStatus(Event* event) {
  absl::Notification& notification = *AsHostEvent(event)->GetNotification();
  return notification.HasBeenNotified() ? Event::Status::kComplete
                                        : Event::Status::kPending;
}

bool VsiExecutor::AllocateStream(Stream* stream) { return true; }

void VsiExecutor::DeallocateStream(Stream* stream) {}

bool VsiExecutor::CreateStreamDependency(Stream* dependent, Stream* other) {
  auto event = std::make_shared<absl::Notification>();
  AsHostStream(other)->EnqueueTask([event]() { event->Notify(); });
  AsHostStream(dependent)->EnqueueTask(
      [event]() { event->WaitForNotification(); });
  return true;
}

tsl::Status VsiExecutor::BlockHostUntilDone(Stream* stream) {
  return AsHostStream(stream)->BlockUntilDone();
}

tsl::Status VsiExecutor::EnablePeerAccessTo(StreamExecutorInterface* other) {
  return tsl::OkStatus();
}

bool VsiExecutor::CanEnablePeerAccessTo(StreamExecutorInterface* other) {
  return true;
}

bool VsiExecutor::DeviceMemoryUsage(int64_t* free, int64_t* total) const {
  tsl::port::MemoryInfo mem_info = tsl::port::GetMemoryInfo();
  *free = (mem_info.free != INT64_MAX) ? mem_info.free : -1;
  *total = (mem_info.total != INT64_MAX) ? mem_info.total : -1;
  return true;
}

tsl::StatusOr<std::unique_ptr<DeviceDescription>>
VsiExecutor::CreateDeviceDescription() const {
  internal::DeviceDescriptionBuilder builder;

  builder.set_device_address_bits(64);

  builder.set_name("vsi");
  builder.set_device_memory_size(static_cast<uint64_t>(4) * 1024 * 1024 * 1024);
  builder.set_platform_version("0.0.1");
  return builder.Build();
}

}  // namespace vsi
}  // namespace stream_executor