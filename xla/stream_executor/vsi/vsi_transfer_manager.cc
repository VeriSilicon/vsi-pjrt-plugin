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

#include "xla/stream_executor/vsi/vsi_transfer_manager.h"

#include <memory>

#include "xla/service/transfer_manager.h"
#include "xla/stream_executor/platform/initialize.h"
#include "xla/stream_executor/vsi/vsi_platform_id.h"

namespace stream_executor {
namespace vsi {

VsiTransferManager::VsiTransferManager()
    : GenericTransferManager(kVsiPlatformId,
                             /*pointer_size=*/sizeof(void*)) {}

static std::unique_ptr<xla::TransferManager> CreateVsiTransferManager() {
  return std::make_unique<VsiTransferManager>();
}

static void InitializeVsiTransferManager() {
  xla::TransferManager::RegisterTransferManager(kVsiPlatformId,
                                                &CreateVsiTransferManager);
}

}  // namespace vsi
}  // namespace stream_executor

REGISTER_MODULE_INITIALIZER(
    vsi_transfer_manager, stream_executor::vsi::InitializeVsiTransferManager());
