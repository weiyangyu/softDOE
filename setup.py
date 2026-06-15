from __future__ import annotations

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class BuildExt(build_ext):
    def build_extensions(self) -> None:
        import pybind11

        for ext in self.extensions:
            ext.include_dirs.append(pybind11.get_include())
            ext.extra_compile_args = ["-std=c++17"]
        super().build_extensions()


ext_modules = [
    Extension(
        "softDOE._core",
        ["src/softDOE/_core.cpp"],
        language="c++",
    )
]


setup(
    name="softDOE",
    version="0.1.0",
    description="Space-filling one-factor-at-a-time designs.",
    author="Wei-Yang Yu and V. Roshan Joseph",
    package_dir={"": "src"},
    packages=["softDOE"],
    python_requires=">=3.9",
    install_requires=["numpy", "scipy", "pyDOE"],
    extras_require={"test": ["pytest"]},
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExt},
)
