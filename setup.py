"""Packaging configuration for dual_cbf_compiler."""

from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="dual_cbf_compiler",
    version="0.1.0",
    author="Mohammadreza Kamaldar",
    author_email="mkamaldar@southalabama.edu",
    description=(
        "Ahead-of-time compiler that translates trained neural control barrier "
        "functions into bare-metal C++ headers evaluating exact Lie derivatives "
        "via dual algebra, with zero dynamic memory allocation on embedded targets."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mkamaldar/dual_cbf_compiler",
    license="MIT",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: C++",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development :: Compilers",
    ],
    keywords=(
        "control-barrier-functions, neural-networks, automatic-differentiation, "
        "embedded-systems, code-generation, dual-numbers, lie-derivatives, "
        "real-time, safety-critical"
    ),
    packages=find_packages(exclude=["tests", "tests.*"]),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.21",
    ],
    extras_require={
        "torch": ["torch>=1.13"],
        "onnx": ["onnx>=1.12"],
        "test": ["pytest>=7.0", "torch>=1.13", "onnx>=1.12"],
        "dev": ["pytest>=7.0", "torch>=1.13", "onnx>=1.12", "ruff", "mypy"],
    },
    entry_points={
        "console_scripts": [
            "dual-cbf-compile=dual_cbf_compiler.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
