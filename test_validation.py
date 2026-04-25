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
