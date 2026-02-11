# This copyright notice applies to this file only
#
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

"""

"""

import sys
import os
from distutils.version import LooseVersion
import types


def _ensure_setuptools(min_version):
    try:
        import setuptools
    except ImportError:
        try:
            import ensurepip
        except Exception as exc:
            print("Error: setuptools is required and ensurepip is unavailable.")
            print(exc)
            sys.exit(1)
        ensurepip.bootstrap()
        import setuptools

    if LooseVersion(setuptools.__version__) < LooseVersion(min_version):
        print("Error: version of setuptools is too old (<%s)!" % min_version)
        sys.exit(1)

    return setuptools


_ensure_setuptools("42")


def _install_pkg_resources_shim():
    try:
        import pkg_resources  # noqa: F401
        return
    except Exception:
        pass

    try:
        from importlib import metadata
    except Exception:
        try:
            import importlib_metadata as metadata
        except Exception:
            return

    def load_entry_point(spec, group, name):
        dist_name, _, _ = spec.partition("==")
        try:
            dist = metadata.distribution(dist_name)
        except Exception as exc:
            raise ImportError("Distribution not found: %s" % dist_name) from exc
        for ep in dist.entry_points:
            if ep.group == group and ep.name == name:
                return ep.load()
        raise ImportError(
            "Entry point not found: %s %s %s" % (dist_name, group, name)
        )

    shim = types.ModuleType("pkg_resources")
    shim.load_entry_point = load_entry_point
    sys.modules["pkg_resources"] = shim


_install_pkg_resources_shim()

if __name__ == "__main__":
    import skbuild

    skbuild.setup(
        name="PyNvVideoCodec",
        version="2.0.2",
        description="PyNvVideoCodec is NVIDIA's Python based video codec library for hardware accelerated video encode and decode on NVIDIA GPUs.",
        author="NVIDIA",
        license="MIT",
        packages=["PyNvVideoCodec", "PyNvVideoCodec.decoders", "PyNvVideoCodec.transcoder", "PyNvVideoCodec.utils", "samples", "benchmarks"],
        package_dir={"": "src", "samples": "samples", "benchmarks": "benchmarks"},
        include_package_data=True,
        cmake_install_dir="src",
        # Ensure compatibility with newer CMake versions when configuring
        # vendored dependencies such as pybind11.
        cmake_args=["-DCMAKE_POLICY_VERSION_MINIMUM=3.5"],
    )
