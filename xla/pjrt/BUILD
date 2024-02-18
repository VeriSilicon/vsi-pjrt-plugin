cc_library(
    name = "se_vsi_pjrt_client",
    srcs = ["se_vsi_pjrt_client.cc"],
    hdrs = ["se_vsi_pjrt_client.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//xla/stream_executor/vsi:vsi_se_driver",
        "@xla//xla/client:client_library",
        "@xla//xla/client:local_client",
        "@xla//xla/pjrt:pjrt_stream_executor_client",
        "@xla//xla/pjrt/c:pjrt_c_api_wrapper_impl",
    ],
)