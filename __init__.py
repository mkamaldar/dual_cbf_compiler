"""dual_cbf_compiler — AOT compiler for neural CBFs.

Translates trained PyTorch / ONNX feedforward neural networks into
bare-metal C++ headers that evaluate exact Lie derivatives using
forward-mode dual or hyper-dual arithmetic.
"""

from __future__ import annotations

from .emitter import emit_cpp_header
from .ir import ActivationLayer, LinearLayer, ParsedNetwork
from .parser import parse_onnx, parse_pytorch

__version__ = "0.1.0"

__all__ = [
    "ActivationLayer",
    "LinearLayer",
    "ParsedNetwork",
    "emit_cpp_header",
    "parse_onnx",
    "parse_pytorch",
]
