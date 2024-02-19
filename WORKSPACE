workspace(name = "vsi_pjrt_plugin")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

############################# OpenXLA Setup ###############################

# To update OpenXLA to a new revision,
# a) update URL and strip_prefix to the new git commit hash
# b) get the sha256 hash of the commit by running:
#    curl -L https://github.com/openxla/xla/archive/<git hash>.tar.gz | sha256sum
#    and update the sha256 with the result.
http_archive(
    name = "xla",
    patch_args = [
        "-l",
        "-p1",
    ],
    patch_tool = "patch",
    patches = [
        # "//patches/openxla:some_patch.patch",
    ],
    strip_prefix = "xla-b166243711f71b0a55daa1eda36b1dc745886784",
    urls = [
        "https://github.com/openxla/xla/archive/b166243711f71b0a55daa1eda36b1dc745886784.tar.gz",
    ],
)

# For development, one often wants to make changes to the OpenXLA repository as well
# as the PyTorch/XLA repository. You can override the pinned repository above with a
# local checkout by either:
# a) overriding the OpenXLA repository on the build.py command line by passing a flag
#    like:
#    bazel --override_repository=xla=/path/to/openxla
#    or
# b) by commenting out the http_archive above and uncommenting the following:
# local_repository(
#    name = "xla",
#    path = "/path/to/openxla",
# )

# Initialize OpenXLA's external dependencies.
load("@xla//:workspace4.bzl", "xla_workspace4")

xla_workspace4()

load("@xla//:workspace3.bzl", "xla_workspace3")

xla_workspace3()

load("@xla//:workspace2.bzl", "xla_workspace2")

xla_workspace2()

load("@xla//:workspace1.bzl", "xla_workspace1")

xla_workspace1()

load("@xla//:workspace0.bzl", "xla_workspace0")

xla_workspace0()

############################# TIM-VX Setup ###############################

http_archive(
    name = "tim_vx",
    patch_args = [
        "-l",
        "-p1",
    ],
    patch_tool = "patch",
    patches = [
        "//patches/tim_vx:relax_compiler_warning_as_error.patch",
    ],
    strip_prefix = "TIM-VX-b4b4f00f474e3b2cc2b33783b45a6a53e017580c",
    urls = [
        "https://github.com/VeriSilicon/TIM-VX/archive/b4b4f00f474e3b2cc2b33783b45a6a53e017580c.tar.gz",
    ],
)

# a) overriding the TIM-VX repository on the build.py command line by passing a flag
#    like:
#    bazel --override_repository=tim_vx=/path/to/tim_vx
#    or
# b) by commenting out the http_archive above and uncommenting the following:
# local_repository(
#    name = "tim_vx",
#    path = "/path/to/tim_vx",
# )

################# Compilation Database Extractor Setup ###################

# Hedron's Compile Commands Extractor for Bazel
# https://github.com/hedronvision/bazel-compile-commands-extractor
http_archive(
    name = "hedron_compile_commands",
    strip_prefix = "bazel-compile-commands-extractor-ed994039a951b736091776d677f324b3903ef939",
    url = "https://github.com/hedronvision/bazel-compile-commands-extractor/archive/ed994039a951b736091776d677f324b3903ef939.tar.gz",
)

load("@hedron_compile_commands//:workspace_setup.bzl", "hedron_compile_commands_setup")

hedron_compile_commands_setup()
