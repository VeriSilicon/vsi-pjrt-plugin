from pathlib import Path
import sys
import subprocess
import shutil

import setuptools

bazel_argv = [
    "bazel",
    "build",
    "//xla:vsi_pjrt_plugin",
]
subprocess.check_call(bazel_argv, stdout=sys.stdout, stderr=sys.stderr)

lib_name = "pjrt_c_api_vsi_plugin.so"
src_dir = Path("bazel-bin/xla/pjrt/c")
dst_dir = Path("torch_xla_vsi_plugin/lib")
dst_dir.mkdir(parents=True, exist_ok=True)

shutil.copyfile(
    src=src_dir / lib_name,
    dst=dst_dir / lib_name
)

setuptools.setup()
