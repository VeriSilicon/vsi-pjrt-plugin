load("@xla//xla:xla.bzl", "xla_cc_binary", "xla_cc_test")

cc_library(
    name = "pjrt_c_api_vsi_internal",
    srcs = ["pjrt_c_api_vsi_internal.cc"],
    hdrs = ["pjrt_c_api_vsi_internal.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//xla/pjrt:se_vsi_pjrt_client",
        "@xla//xla/pjrt:pjrt_client",
        "@xla//xla/pjrt/c:pjrt_c_api_hdrs",
        "@xla//xla/pjrt/c:pjrt_c_api_helpers",
        "@xla//xla/pjrt/c:pjrt_c_api_wrapper_impl",
    ],
)

cc_library(
    name = "pjrt_c_api_vsi",
    srcs = ["pjrt_c_api_vsi.cc"],
    hdrs = ["pjrt_c_api_vsi.h"],
    linkstatic = True,
    visibility = ["//visibility:public"],
    deps = [
        ":pjrt_c_api_vsi_internal",
        "@xla//xla/pjrt/c:pjrt_c_api_hdrs",
    ],
    alwayslink = True,
)

xla_cc_binary(
    name = "pjrt_c_api_vsi_plugin.so",
    linkopts = [
        "-Wl,--version-script,$(location :pjrt_c_api_vsi_version_script.lds)",
        "-Wl,--no-undefined",
    ],
    linkshared = True,
    deps = [
        ":pjrt_c_api_vsi",
        ":pjrt_c_api_vsi_version_script.lds",
    ],
)

xla_cc_test(
    name = "pjrt_c_api_vsi_test",
    srcs = ["pjrt_c_api_vsi_test.cc"],
    deps = [
        ":pjrt_c_api_vsi",
        "@com_google_googletest//:gtest_main",
        "@xla//xla/pjrt/c:pjrt_c_api_test_common",
        "@xla//xla/pjrt/c:pjrt_c_api_wrapper_impl",
    ],
)
