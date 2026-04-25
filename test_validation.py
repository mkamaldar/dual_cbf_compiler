"""Tests for parser validation (no torch required)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from dual_cbf_compiler.ir import ActivationLayer, LinearLayer, ParsedNetwork


def test_validate_empty():
    try:
        ParsedNetwork(layers=[]).validate()
    except ValueError as e:
        assert "Empty" in str(e)
        return
    raise AssertionError("expected ValueError")


def test_validate_must_start_linear():
    net = ParsedNetwork(layers=[ActivationLayer("relu")])
    try:
        net.validate()
    except ValueError as e:
        assert "linear" in str(e).lower()
        return
    raise AssertionError("expected ValueError")


def test_validate_dim_mismatch():
    net = ParsedNetwork(layers=[
        LinearLayer(W=np.zeros((4, 3), dtype=np.float32), b=np.zeros(4, dtype=np.float32)),
        ActivationLayer("relu"),
        LinearLayer(W=np.zeros((1, 5), dtype=np.float32), b=np.zeros(1, dtype=np.float32)),  # 5 != 4
    ])
    try:
        net.validate()
    except ValueError as e:
        assert "mismatch" in str(e).lower()
        return
    raise AssertionError("expected ValueError")


def test_validate_scalar_output():
    net = ParsedNetwork(layers=[
        LinearLayer(W=np.zeros((4, 3), dtype=np.float32), b=np.zeros(4, dtype=np.float32)),
        ActivationLayer("relu"),
        LinearLayer(W=np.zeros((2, 4), dtype=np.float32), b=np.zeros(2, dtype=np.float32)),
    ])
    try:
        net.validate()
    except ValueError as e:
        assert "output dimension" in str(e).lower()
        return
    raise AssertionError("expected ValueError")


def test_validate_alternating():
    net = ParsedNetwork(layers=[
        LinearLayer(W=np.zeros((4, 3), dtype=np.float32), b=np.zeros(4, dtype=np.float32)),
        LinearLayer(W=np.zeros((1, 4), dtype=np.float32), b=np.zeros(1, dtype=np.float32)),
    ])
    try:
        net.validate()
    except ValueError as e:
        assert "activation" in str(e).lower() or "alternat" in str(e).lower() or "expected" in str(e).lower()
        return
    raise AssertionError("expected ValueError")


def test_emitter_rejects_relu_relative_degree_2():
    """Hard error when emitting hyper-dual code for a ReLU network."""
    from dual_cbf_compiler.emitter import emit_cpp_header
    net = ParsedNetwork(layers=[
        LinearLayer(W=np.zeros((4, 3), dtype=np.float32), b=np.zeros(4, dtype=np.float32)),
        ActivationLayer("relu"),
        LinearLayer(W=np.zeros((1, 4), dtype=np.float32), b=np.zeros(1, dtype=np.float32)),
    ])
    try:
        emit_cpp_header(net, relative_degree=2)
    except ValueError as e:
        msg = str(e).lower()
        assert "relu" in msg
        assert "relative_degree=2" in msg or "second derivative" in msg
        return
    raise AssertionError("expected ValueError for ReLU + relative_degree=2")


def test_emitter_accepts_softplus_relative_degree_2():
    """Smooth activations are fine for hyper-dual."""
    from dual_cbf_compiler.emitter import emit_cpp_header
    net = ParsedNetwork(layers=[
        LinearLayer(W=np.zeros((4, 3), dtype=np.float32), b=np.zeros(4, dtype=np.float32)),
        ActivationLayer("softplus"),
        LinearLayer(W=np.zeros((1, 4), dtype=np.float32), b=np.zeros(1, dtype=np.float32)),
    ])
    header = emit_cpp_header(net, relative_degree=2)
    assert "evaluate_cbf_2nd_order" in header
    assert "ROW-MAJOR" in header  # documentation reminder is present


def test_emitter_documents_row_major_layout():
    """Generated header explains the row-major G layout prominently."""
    from dual_cbf_compiler.emitter import emit_cpp_header
    net = ParsedNetwork(layers=[
        LinearLayer(W=np.zeros((4, 3), dtype=np.float32), b=np.zeros(4, dtype=np.float32)),
        ActivationLayer("relu"),
        LinearLayer(W=np.zeros((1, 4), dtype=np.float32), b=np.zeros(1, dtype=np.float32)),
    ])
    header = emit_cpp_header(net, relative_degree=1)
    assert "ROW-MAJOR" in header
    assert "G[i * m + j]" in header
    assert "G2D" in header  # the worked-example block is present


def test_widths_and_depth():
    net = ParsedNetwork(layers=[
        LinearLayer(W=np.zeros((8, 4), dtype=np.float32), b=np.zeros(8, dtype=np.float32)),
        ActivationLayer("relu"),
        LinearLayer(W=np.zeros((8, 8), dtype=np.float32), b=np.zeros(8, dtype=np.float32)),
        ActivationLayer("relu"),
        LinearLayer(W=np.zeros((1, 8), dtype=np.float32), b=np.zeros(1, dtype=np.float32)),
    ])
    net.validate()
    assert net.widths == [4, 8, 8, 1]
    assert net.depth == 3
    assert net.input_dim == 4
    assert net.output_dim == 1


def main():
    fns = [
        test_validate_empty,
        test_validate_must_start_linear,
        test_validate_dim_mismatch,
        test_validate_scalar_output,
        test_validate_alternating,
        test_widths_and_depth,
        test_emitter_rejects_relu_relative_degree_2,
        test_emitter_accepts_softplus_relative_degree_2,
        test_emitter_documents_row_major_layout,
    ]
    n_pass = 0
    for fn in fns:
        try:
            fn()
            print(f"[PASS] {fn.__name__}")
            n_pass += 1
        except Exception as e:
            print(f"[FAIL] {fn.__name__}: {e}")
    print(f"\n{n_pass}/{len(fns)} validation tests passed.")
    return 0 if n_pass == len(fns) else 1


if __name__ == "__main__":
    raise SystemExit(main())
