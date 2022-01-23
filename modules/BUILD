# Description:
# Example TensorFlow models for comet_dnn

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

py_library(
    name = "comet_dnn_input",
    srcs = ["comet_dnn_input.py"],
    srcs_version = "PY2AND3",
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "comet_dnn_input_test",
    size = "small",
    srcs = ["comet_dnn_input_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":comet_dnn_input",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
)

py_library(
    name = "comet_dnn",
    srcs = ["comet_dnn.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":comet_dnn_input",
        "//tensorflow:tensorflow_py",
    ],
)

py_binary(
    name = "comet_dnn_eval",
    srcs = [
        "comet_dnn_eval.py",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//tensorflow:__subpackages__"],
    deps = [
        ":comet_dnn",
    ],
)

py_binary(
    name = "comet_dnn_train",
    srcs = [
        "comet_dnn_train.py",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//tensorflow:__subpackages__"],
    deps = [
        ":comet_dnn",
    ],
)

py_binary(
    name = "comet_dnn_multi_gpu_train",
    srcs = [
        "comet_dnn_multi_gpu_train.py",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//tensorflow:__subpackages__"],
    deps = [
        ":comet_dnn",
    ],
)

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)
