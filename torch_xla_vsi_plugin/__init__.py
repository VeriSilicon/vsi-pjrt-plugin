import os
from torch_xla.experimental.plugins import DevicePlugin


class VsiPlugin(DevicePlugin):
    def library_path(self) -> str:
        return os.path.join(os.path.dirname(__file__), "lib", "pjrt_c_api_vsi_plugin.so")

    def physical_chip_count(self) -> int:
        return 1
