from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "maxflow_cpp",
        ["maxflow.cpp"],
        extra_compile_args=['-O3', '-std=c++11'],
    ),
]

setup(
    name="maxflow_cpp",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)

"""
ext_modules = [
    Pybind11Extension(
        "maxflow_bk_cpp",
        ["maxflow_bk.cpp"],
        extra_compile_args=['-O3', '-std=c++11'],
    ),
]

setup(
    name="maxflow_bk_cpp",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
"""