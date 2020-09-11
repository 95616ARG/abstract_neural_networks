load("@bazel_python//:bazel_python.bzl", "bazel_python_coverage_report", "bazel_python_interpreter")

bazel_python_interpreter(
    name = "bazel_python_venv",
    python_version = "3.7.4",
    requirements_file = "requirements.txt",
)

bazel_python_coverage_report(
    name = "coverage_report",
    code_paths = ["*.py"],
    test_paths = ["test_abstract_intervals"],
)

py_library(
    name = "abstract",
    srcs = ["abstract.py"],
)

py_library(
    name = "interval_domain",
    srcs = ["interval_domain.py"],
)

py_test(
    name = "test_abstract_intervals",
    size = "small",
    srcs = ["test_abstract_intervals.py"],
    deps = [
        ":abstract",
        ":interval_domain",
        "@bazel_python//:pytest_helper",
    ],
)
