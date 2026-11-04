from __future__ import annotations

import sys
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class BuildExt(build_ext):
    def build_extensions(self) -> None:
        import pybind11

        if ".mm" not in self.compiler.src_extensions:
            self.compiler.src_extensions.append(".mm")
        for ext in self.extensions:
            ext.include_dirs.append(pybind11.get_include())
            ext.extra_compile_args = (["/std:c++17"] if self.compiler.compiler_type == "msvc"
                                      else ["-std=c++17"])
            if ext.name == "softDOE._native" and self.compiler.compiler_type != "msvc":
                ext.extra_compile_args.append("-ffp-contract=off")
        super().build_extensions()


ext_modules = [
    Extension(
        "softDOE._core",
        ["src/softDOE/_core.cpp"],
        depends=["src/softDOE/_kernel.hpp"],
        language="c++",
    ),
    Extension("softDOE._native", ["src/softDOE/_native.cpp"] +
              (["src/softDOE/_native_metal.mm"] if sys.platform == "darwin" else []) +
              (["src/softDOE/_native_cuda.cpp"] if sys.platform.startswith("linux") else []),
              depends=["src/softDOE/_native.hpp", "src/softDOE/_kernel.hpp"], language="c++",
              define_macros=([("SOFTDOE_NATIVE_METAL", "1")] if sys.platform == "darwin" else
                             [("SOFTDOE_NATIVE_CUDA", "1")] if sys.platform.startswith("linux") else []),
              extra_link_args=["-weak_framework", "Metal", "-framework", "Foundation",
                               "-framework", "Accelerate"]
              if sys.platform == "darwin" else ["-ldl"] if sys.platform.startswith("linux") else []),
]


setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExt},
)
