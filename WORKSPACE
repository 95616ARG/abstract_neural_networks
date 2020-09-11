workspace(name = "SyReNN")

load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

git_repository(
    name = "bazel_python",
    commit = "f99ab8738dced7257c97dc719457f50a601ed84c",
    remote = "https://github.com/95616ARG/bazel_python.git",
)

load("@bazel_python//:bazel_python.bzl", "bazel_python")

bazel_python()
