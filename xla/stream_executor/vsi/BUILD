cc_library(
    name = "vsi_platform_id",
    srcs = ["vsi_platform_id.cc"],
    hdrs = ["vsi_platform_id.h"],
    visibility = ["//visibility:public"],
    deps = ["@xla//xla/stream_executor:platform"],
)

cc_library(
    name = "vsi_transfer_manager",
    srcs = ["vsi_transfer_manager.cc"],
    hdrs = ["vsi_transfer_manager.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":vsi_platform_id",
        "@xla//xla/service:generic_transfer_manager",
        "@xla//xla/service:transfer_manager",
        "@xla//xla/stream_executor/platform:dso_loader",
    ],
    alwayslink = True,  # Contains transfer manager registration
)

cc_library(
    name = "vsi_compiler",
    srcs = ["vsi_compiler.cc"],
    hdrs = ["vsi_compiler.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":vsi_executable",
        ":vsi_executor",
        ":vsi_platform_id",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:status",
        "@tsl//tsl/platform:statusor",
        "@xla//xla/pjrt:pjrt_compiler",
        "@xla//xla/pjrt:utils",
        "@xla//xla/service:algebraic_simplifier",
        "@xla//xla/service:batch_dot_simplification",
        "@xla//xla/service:batchnorm_expander",
        "@xla//xla/service:call_inliner",
        "@xla//xla/service:compiler",
        "@xla//xla/service:conditional_simplifier",
        "@xla//xla/service:dot_decomposer",
        "@xla//xla/service:dynamic_dimension_inference",
        "@xla//xla/service:dynamic_dimension_simplifier",
        "@xla//xla/service:dynamic_padder",
        "@xla//xla/service:gather_simplifier",
        "@xla//xla/service:hlo_constant_folding",
        "@xla//xla/service:hlo_cse",
        "@xla//xla/service:hlo_dce",
        "@xla//xla/service:hlo_pass_pipeline",
        "@xla//xla/service:reshape_mover",
        "@xla//xla/service:topk_rewriter",
        "@xla//xla/service:transpose_folding",
        "@xla//xla/service:tuple_simplifier",
        "@xla//xla/service:zero_sized_hlo_elimination",
        "@xla//xla/stream_executor/platform:dso_loader",
    ],
    alwayslink = True,  # Contains compiler registration
)

cc_library(
    name = "vsi_executable",
    srcs = [
        "vsi_executable.cc",
        "vsi_graph_builder.cc",
    ],
    hdrs = [
        "vsi_executable.h",
        "vsi_graph_builder.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":vsi_executor",
        "@com_google_absl//absl/synchronization",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AsmParser",
        "@llvm-project//mlir:IR",
        "@tim_vx//:tim-vx_interface",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:status",
        "@tsl//tsl/platform:statusor",
        "@xla//xla:literal",
        "@xla//xla:shape_tree",
        "@xla//xla:shape_util",
        "@xla//xla:status_macros",
        "@xla//xla:statusor",
        "@xla//xla:types",
        "@xla//xla:xla_data_proto_cc",
        "@xla//xla:xla_proto_cc",
        "@xla//xla/hlo/evaluator:hlo_evaluator",
        "@xla//xla/hlo/ir:hlo",
        "@xla//xla/service:executable",
        "@xla//xla/service:hlo_cost_analysis",
        "@xla//xla/service:hlo_execution_profile",
        "@xla//xla/service:hlo_module_config",
        "@xla//xla/service:maybe_owning_device_memory",
        "@xla//xla/service:shaped_buffer",
        "@xla//xla/service:transfer_manager",
        "@xla//xla/stream_executor",
    ],
    alwayslink = True,
)

cc_library(
    name = "vsi_executor",
    srcs = ["vsi_executor.cc"],
    hdrs = ["vsi_executor.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_absl//absl/synchronization",
        "@tim_vx//:tim-vx_interface",
        "@xla//xla:shape_util",
        "@xla//xla:status_macros",
        "@xla//xla:xla_data_proto_cc",
        "@xla//xla/stream_executor",
        "@xla//xla/stream_executor:stream_executor_internal",
        "@xla//xla/stream_executor/host:host_stream",
    ],
    alwayslink = True,
)

cc_library(
    name = "vsi_platform",
    srcs = [
        "vsi_platform.cc",
    ],
    hdrs = [
        "vsi_platform.h",
    ],
    linkstatic = True,
    visibility = ["//visibility:public"],
    deps = [
        ":vsi_executor",
        ":vsi_platform_id",
        "@tsl//tsl/platform:status",
        "@tsl//tsl/platform:statusor",
        "@xla//xla/stream_executor/platform",
        "@xla//xla/stream_executor/platform:dso_loader",
    ],
    alwayslink = True,
)

cc_library(
    name = "vsi_se_driver",
    visibility = ["//visibility:public"],
    deps = [
        ":vsi_compiler",
        ":vsi_executable",
        ":vsi_executor",
        ":vsi_platform",
        ":vsi_platform_id",
        ":vsi_transfer_manager",
    ],
)
