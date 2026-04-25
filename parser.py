"""Dual-ingestion parsers: PyTorch and ONNX into a unified IR.

Both parsers reject convolutional, recurrent, and batch-normalization
architectures. The resulting :class:`ParsedNetwork` is a flat sequence
of :class:`LinearLayer` and :class:`ActivationLayer` entries.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

from .ir import ActivationKind, ActivationLayer, LinearLayer, ParsedNetwork

if TYPE_CHECKING:
    import torch.nn as nn

# ---------------------------------------------------------------------------
# Disallowed module / op classifications
# ---------------------------------------------------------------------------

_DISALLOWED_TORCH_TYPES = {
    "Conv1d", "Conv2d", "Conv3d",
    "ConvTranspose1d", "ConvTranspose2d", "ConvTranspose3d",
    "BatchNorm1d", "BatchNorm2d", "BatchNorm3d",
    "LayerNorm", "GroupNorm", "InstanceNorm1d", "InstanceNorm2d",
    "RNN", "LSTM", "GRU", "RNNCell", "LSTMCell", "GRUCell",
    "MultiheadAttention", "Transformer", "TransformerEncoder", "TransformerDecoder",
    "Dropout", "Dropout2d", "Dropout3d", "AlphaDropout",
}

_DISALLOWED_ONNX_OPS = {
    "Conv", "ConvTranspose",
    "BatchNormalization", "LayerNormalization", "InstanceNormalization",
    "RNN", "LSTM", "GRU",
    "MaxPool", "AveragePool", "GlobalAveragePool", "GlobalMaxPool",
    "Dropout",
    "Attention",
}

_TORCH_ACTIVATION_MAP: dict[str, ActivationKind] = {
    "ReLU": "relu",
    "Softplus": "softplus",
    "Tanh": "tanh",
    "Sigmoid": "sigmoid",
    "Identity": "identity",
}

_ONNX_ACTIVATION_MAP: dict[str, ActivationKind] = {
    "Relu": "relu",
    "Softplus": "softplus",
    "Tanh": "tanh",
    "Sigmoid": "sigmoid",
    "Identity": "identity",
}


# ---------------------------------------------------------------------------
# PyTorch ingestion
# ---------------------------------------------------------------------------

def parse_pytorch(model: "nn.Sequential") -> ParsedNetwork:
    """Parse a torch.nn.Sequential CBF model into the unified IR.

    Args:
        model: A torch.nn.Sequential whose modules are Linear layers
            interleaved with supported activations (ReLU, Softplus, Tanh,
            Sigmoid, Identity). The terminal layer must be Linear with
            output dimension 1.

    Returns:
        Validated ParsedNetwork.

    Raises:
        TypeError: If ``model`` is not torch.nn.Sequential.
        ValueError: If any module is unsupported (Conv, BN, RNN, etc.) or
            the resulting topology violates feedforward-CBF assumptions.
    """
    import torch.nn as nn  # local import keeps torch optional at top level

    if not isinstance(model, nn.Sequential):
        raise TypeError(
            f"parse_pytorch requires torch.nn.Sequential, got {type(model).__name__}"
        )

    parsed = ParsedNetwork()
    state = model.state_dict()

    linear_idx = 0
    for module in model:
        cls_name = type(module).__name__

        if cls_name in _DISALLOWED_TORCH_TYPES:
            raise ValueError(
                f"Architecture rejected: {cls_name} is not supported. "
                f"Networks must be strictly feedforward (Linear + activation only)."
            )

        if isinstance(module, nn.Linear):
            W = module.weight.detach().cpu().numpy().astype(np.float32, copy=True)
            if module.bias is not None:
                b = module.bias.detach().cpu().numpy().astype(np.float32, copy=True)
            else:
                b = np.zeros((W.shape[0],), dtype=np.float32)

            # Per spec: weights and biases are stored flattened (row-major)
            W = np.ascontiguousarray(W.reshape(W.shape[0], W.shape[1]))
            b = np.ascontiguousarray(b.reshape(-1))

            parsed.layers.append(LinearLayer(W=W, b=b))
            linear_idx += 1

        elif cls_name in _TORCH_ACTIVATION_MAP:
            kind = _TORCH_ACTIVATION_MAP[cls_name]
            parsed.layers.append(ActivationLayer(kind=kind))

        elif isinstance(module, nn.Flatten):
            # Acceptable but emits no operation; skip.
            continue

        else:
            raise ValueError(
                f"Unsupported PyTorch module: {cls_name}. "
                f"Allowed: Linear, ReLU, Softplus, Tanh, Sigmoid, Identity, Flatten."
            )

    parsed.validate()
    _ = state  # state_dict referenced for symmetry with the ONNX path
    return parsed


# ---------------------------------------------------------------------------
# ONNX ingestion
# ---------------------------------------------------------------------------

def parse_onnx(file_path: str) -> ParsedNetwork:
    """Parse an ONNX model file into the unified IR.

    Walks the ONNX computational graph and pattern-matches sequences of
    Gemm or (MatMul + Add) followed by activation nodes. Initializer
    tensors carry the weight and bias data.

    Args:
        file_path: Path to a .onnx model file.

    Returns:
        Validated ParsedNetwork.

    Raises:
        ImportError: If onnx is not installed.
        ValueError: If the graph contains disallowed ops (Conv, BN, RNN,
            attention, pooling, dropout) or cannot be reduced to an
            alternating Linear/activation sequence.
    """
    try:
        import onnx
        from onnx import numpy_helper
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "parse_onnx requires the 'onnx' package. Install with: pip install onnx"
        ) from exc

    model = onnx.load(file_path)
    graph = model.graph

    # Reject blacklisted ops upfront
    for node in graph.node:
        if node.op_type in _DISALLOWED_ONNX_OPS:
            raise ValueError(
                f"Architecture rejected: ONNX op '{node.op_type}' is not supported. "
                f"Networks must be strictly feedforward."
            )

    # Build initializer lookup
    initializers: dict[str, np.ndarray] = {}
    for init in graph.initializer:
        initializers[init.name] = numpy_helper.to_array(init).astype(np.float32, copy=True)

    parsed = ParsedNetwork()
    nodes = list(graph.node)
    i = 0

    while i < len(nodes):
        node = nodes[i]
        op = node.op_type

        if op == "Gemm":
            # Gemm(A=x, B=W, C=b) => x @ W^T + b (transB=1) or x @ W + b (transB=0)
            attrs = {a.name: a for a in node.attribute}
            transB = bool(attrs["transB"].i) if "transB" in attrs else False
            transA = bool(attrs["transA"].i) if "transA" in attrs else False
            alpha = float(attrs["alpha"].f) if "alpha" in attrs else 1.0
            beta = float(attrs["beta"].f) if "beta" in attrs else 1.0

            if transA:
                raise ValueError("Gemm with transA=1 is not supported")

            B_name = node.input[1]
            C_name = node.input[2] if len(node.input) > 2 else None

            if B_name not in initializers:
                raise ValueError(
                    f"Gemm operand B='{B_name}' is not a static initializer; "
                    f"only fixed-weight feedforward networks are supported."
                )

            B = initializers[B_name]
            # We want W with shape (n_out, n_in) so that y = W x + b matches IR
            if transB:
                W = (alpha * B).copy()  # already (n_out, n_in)
            else:
                W = (alpha * B.T).copy()  # transpose (n_in, n_out) to (n_out, n_in)

            if C_name is not None and C_name in initializers:
                b = (beta * initializers[C_name]).reshape(-1).copy()
            else:
                b = np.zeros((W.shape[0],), dtype=np.float32)

            parsed.layers.append(LinearLayer(W=W.astype(np.float32), b=b.astype(np.float32)))
            i += 1

        elif op == "MatMul":
            # Pattern: MatMul(x, W) -> Add(., b) [-> activation]
            B_name = node.input[1]
            if B_name not in initializers:
                raise ValueError(
                    f"MatMul operand '{B_name}' is not a static initializer."
                )
            B = initializers[B_name]
            # x @ W with W shape (n_in, n_out); IR expects (n_out, n_in)
            W = B.T.copy()

            # Look ahead for Add
            b = np.zeros((W.shape[0],), dtype=np.float32)
            mm_out = node.output[0]
            consumed = 1
            if i + 1 < len(nodes) and nodes[i + 1].op_type == "Add":
                add_node = nodes[i + 1]
                if mm_out in add_node.input:
                    other = [name for name in add_node.input if name != mm_out][0]
                    if other in initializers:
                        b = initializers[other].reshape(-1).copy()
                        consumed = 2

            parsed.layers.append(LinearLayer(W=W.astype(np.float32), b=b.astype(np.float32)))
            i += consumed

        elif op in _ONNX_ACTIVATION_MAP:
            parsed.layers.append(ActivationLayer(kind=_ONNX_ACTIVATION_MAP[op]))
            i += 1

        elif op in {"Reshape", "Flatten", "Identity"}:
            i += 1  # structurally inert

        elif op == "Add":
            # Add appearing on its own, not paired with a MatMul we consumed
            raise ValueError(
                f"Stray Add node '{node.name or node.output[0]}' has no matching MatMul."
            )

        else:
            raise ValueError(
                f"Unsupported ONNX op: '{op}'. "
                f"Allowed: Gemm, MatMul+Add, Relu, Softplus, Tanh, Sigmoid, Identity, "
                f"Reshape, Flatten."
            )

    parsed.validate()
    return parsed
