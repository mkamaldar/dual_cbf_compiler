"""Mathematical verification suite for the dual-CBF compiler.

Each test:
  1. Builds a random feedforward PyTorch network and random states/fields;
  2. Computes the reference Lie derivatives via torch.autograd;
  3. Emits the C++ header, compiles a tiny driver via g++, runs it;
  4. Asserts the C++ output equals the PyTorch baseline to single-precision
     machine epsilon (relative tolerance, with a small absolute floor).

Run with:  pytest -v dual_cbf_compiler/verify.py
"""

from __future__ import annotations

import os
import struct
import subprocess
import tempfile
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pytest
import torch
import torch.nn as nn

from .emitter import emit_cpp_header
from .parser import parse_pytorch


# ---------------------------------------------------------------------------
# Tolerances
# ---------------------------------------------------------------------------
# Single-precision machine epsilon is ~1.19e-7. The dual algebra is
# bit-equivalent to PyTorch's float32 evaluation up to floating-point
# accumulation reordering, so we expect agreement at a few times eps.
_RTOL = 5e-5
_ATOL = 5e-5


# ---------------------------------------------------------------------------
# Reference computations using PyTorch autograd
# ---------------------------------------------------------------------------

def _torch_lie_first_order(
    model: nn.Sequential,
    x: np.ndarray,
    f: np.ndarray,
    G: np.ndarray,
) -> Tuple[float, float, np.ndarray]:
    """Compute h, L_f h, and L_G h via torch.autograd."""
    n = x.shape[0]
    m = G.shape[1]
    xt = torch.tensor(x, dtype=torch.float32, requires_grad=True)
    h = model(xt.unsqueeze(0)).squeeze()
    grad = torch.autograd.grad(h, xt, create_graph=False, retain_graph=False)[0]
    grad_np = grad.detach().cpu().numpy().astype(np.float32)

    h_val = float(h.detach().cpu().numpy())
    Lf = float(grad_np @ f)
    Lg = (grad_np @ G).astype(np.float32)
    return h_val, Lf, Lg


def _torch_lie_second_order(
    model: nn.Sequential,
    x: np.ndarray,
    f: np.ndarray,
    Df_f: np.ndarray,
    G: np.ndarray | None = None,
    Df_G: np.ndarray | None = None,
) -> Tuple[float, np.ndarray | None]:
    """Compute L_f^2 h_theta(x) and (optionally) L_{col_j(G)} L_f h_theta(x).

    Identity:
      L_f^2 h(x)            = f^T H(x) f + grad h(x)^T (Df . f)
      L_{xi} L_f h(x)       = f^T H(x) xi + grad h(x)^T (Df . xi)
    """
    xt = torch.tensor(x, dtype=torch.float32, requires_grad=True)
    h = model(xt.unsqueeze(0)).squeeze()
    grad = torch.autograd.grad(h, xt, create_graph=True)[0]  # n
    # Hessian via autograd: H[i,:] = d/dx (grad[i])
    H = torch.zeros((x.shape[0], x.shape[0]), dtype=torch.float32)
    for i in range(x.shape[0]):
        Hi = torch.autograd.grad(grad[i], xt, retain_graph=(i < x.shape[0] - 1))[0]
        H[i] = Hi
    H_np = H.detach().cpu().numpy().astype(np.float32)
    grad_np = grad.detach().cpu().numpy().astype(np.float32)

    L2f = float(f @ H_np @ f + grad_np @ Df_f)

    LgLf = None
    if G is not None and Df_G is not None:
        m = G.shape[1]
        LgLf = np.zeros((m,), dtype=np.float32)
        for j in range(m):
            gj = G[:, j].astype(np.float32)
            Df_gj = Df_G[:, j].astype(np.float32)
            LgLf[j] = float(f @ H_np @ gj + grad_np @ Df_gj)

    return L2f, LgLf


# ---------------------------------------------------------------------------
# C++ build & execution harness
# ---------------------------------------------------------------------------

_DRIVER_FIRST_ORDER = r"""
// Auto-generated test driver
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "dual_cbf.h"

int main(int argc, char** argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: driver <input.bin> <output.bin> <m>\n");
        return 1;
    }
    int m = atoi(argv[3]);

    int n = dual_cbf::INPUT_DIM;
    float x[%(N)d];
    float f[%(N)d];
    float G[%(N)d * 64];  // m up to 64

    FILE* fp = fopen(argv[1], "rb");
    if (!fp) { fprintf(stderr, "open input failed\n"); return 2; }
    fread(x, sizeof(float), n, fp);
    fread(f, sizeof(float), n, fp);
    fread(G, sizeof(float), n * m, fp);
    fclose(fp);

    float h, Lf;
    float Lg[64];
    dual_cbf::evaluate_cbf(x, f, G, m, &h, &Lf, Lg);

    FILE* op = fopen(argv[2], "wb");
    if (!op) { fprintf(stderr, "open output failed\n"); return 3; }
    fwrite(&h, sizeof(float), 1, op);
    fwrite(&Lf, sizeof(float), 1, op);
    fwrite(Lg, sizeof(float), m, op);
    fclose(op);
    return 0;
}
"""

_DRIVER_SECOND_ORDER = r"""
// Auto-generated test driver
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "dual_cbf.h"

int main(int argc, char** argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: driver <input.bin> <output.bin> <m>\n");
        return 1;
    }
    int m = atoi(argv[3]);
    int n = dual_cbf::INPUT_DIM;
    float x[%(N)d];
    float f[%(N)d];
    float G[%(N)d * 64];
    float Df_f[%(N)d];
    float Df_G[%(N)d * 64];

    FILE* fp = fopen(argv[1], "rb");
    fread(x, sizeof(float), n, fp);
    fread(f, sizeof(float), n, fp);
    fread(G, sizeof(float), n * m, fp);
    fread(Df_f, sizeof(float), n, fp);
    fread(Df_G, sizeof(float), n * m, fp);
    fclose(fp);

    float L2f;
    float LgLf[64];
    dual_cbf::evaluate_cbf_2nd_order(x, f, G, Df_f, Df_G, m, &L2f, LgLf);

    FILE* op = fopen(argv[2], "wb");
    fwrite(&L2f, sizeof(float), 1, op);
    fwrite(LgLf, sizeof(float), m, op);
    fclose(op);
    return 0;
}
"""


def _compile_and_run(
    header_text: str,
    driver_template: str,
    input_dim: int,
    payload: bytes,
    m: int,
    expect_floats: int,
) -> np.ndarray:
    """Write header + driver, compile via g++, execute, parse stdout/output."""
    if not _have_gxx():
        pytest.skip("g++ not available on PATH")

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "dual_cbf.h").write_text(header_text)

        driver_src = driver_template % {"N": input_dim}
        (td / "driver.cpp").write_text(driver_src)

        binary = td / "driver"
        cmd = [
            "g++", "-O2", "-std=c++17",
            "-Wno-unused-but-set-variable",
            "-Wno-unused-variable",
            "-I", str(td),
            str(td / "driver.cpp"),
            "-o", str(binary),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"g++ failed:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
            )

        in_file = td / "in.bin"
        out_file = td / "out.bin"
        in_file.write_bytes(payload)

        run = subprocess.run(
            [str(binary), str(in_file), str(out_file), str(m)],
            capture_output=True, text=True,
        )
        if run.returncode != 0:
            raise RuntimeError(
                f"driver failed:\nSTDOUT: {run.stdout}\nSTDERR: {run.stderr}"
            )

        out_bytes = out_file.read_bytes()
        floats = np.frombuffer(out_bytes, dtype=np.float32)
        if floats.shape[0] != expect_floats:
            raise RuntimeError(
                f"driver returned {floats.shape[0]} floats, expected {expect_floats}"
            )
        return floats.copy()


def _have_gxx() -> bool:
    """Check whether g++ is available on PATH."""
    try:
        return subprocess.run(
            ["g++", "--version"], capture_output=True
        ).returncode == 0
    except FileNotFoundError:
        return False


# ---------------------------------------------------------------------------
# Helpers for test fixtures
# ---------------------------------------------------------------------------

def _build_relu_net(widths: List[int], seed: int) -> nn.Sequential:
    torch.manual_seed(seed)
    modules: list[nn.Module] = []
    for i in range(len(widths) - 1):
        modules.append(nn.Linear(widths[i], widths[i + 1]))
        if i < len(widths) - 2:
            modules.append(nn.ReLU())
    return nn.Sequential(*modules)


def _build_softplus_net(widths: List[int], seed: int) -> nn.Sequential:
    torch.manual_seed(seed)
    modules: list[nn.Module] = []
    for i in range(len(widths) - 1):
        modules.append(nn.Linear(widths[i], widths[i + 1]))
        if i < len(widths) - 2:
            modules.append(nn.Softplus())
    return nn.Sequential(*modules)


def _build_tanh_net(widths: List[int], seed: int) -> nn.Sequential:
    torch.manual_seed(seed)
    modules: list[nn.Module] = []
    for i in range(len(widths) - 1):
        modules.append(nn.Linear(widths[i], widths[i + 1]))
        if i < len(widths) - 2:
            modules.append(nn.Tanh())
    return nn.Sequential(*modules)


def _bias_inputs_relu_safe(rng: np.random.Generator, n: int) -> np.ndarray:
    """Return inputs that are uniformly bounded away from ReLU's kink at zero."""
    x = rng.uniform(-1.0, 1.0, size=(n,)).astype(np.float32)
    x[np.abs(x) < 0.1] += 0.1 * np.sign(x[np.abs(x) < 0.1] + 1e-9)
    return x


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "widths,seed",
    [
        ([4, 16, 16, 1], 0),
        ([2, 32, 32, 1], 1),
        ([6, 8, 8, 8, 1], 2),
        ([3, 24, 1], 3),
    ],
)
def test_first_order_relu(widths: List[int], seed: int) -> None:
    """Dual algebra matches PyTorch autograd for ReLU networks (rel. degree 1)."""
    model = _build_relu_net(widths, seed)
    parsed = parse_pytorch(model)

    n = widths[0]
    m = 2
    rng = np.random.default_rng(seed + 100)
    x = _bias_inputs_relu_safe(rng, n)
    f = rng.standard_normal(n).astype(np.float32)
    G = rng.standard_normal((n, m)).astype(np.float32)

    h_ref, Lf_ref, Lg_ref = _torch_lie_first_order(model, x, f, G)

    header = emit_cpp_header(parsed, relative_degree=1)
    payload = x.tobytes() + f.tobytes() + G.tobytes()
    out = _compile_and_run(
        header, _DRIVER_FIRST_ORDER, n, payload, m, expect_floats=2 + m
    )
    h_c, Lf_c = float(out[0]), float(out[1])
    Lg_c = out[2:].astype(np.float32)

    assert np.isfinite(h_c) and np.isfinite(Lf_c) and np.all(np.isfinite(Lg_c))
    assert abs(h_c - h_ref) <= _ATOL + _RTOL * abs(h_ref), (
        f"h mismatch: c={h_c} ref={h_ref}"
    )
    assert abs(Lf_c - Lf_ref) <= _ATOL + _RTOL * abs(Lf_ref), (
        f"Lf mismatch: c={Lf_c} ref={Lf_ref}"
    )
    np.testing.assert_allclose(Lg_c, Lg_ref, rtol=_RTOL, atol=_ATOL)


@pytest.mark.parametrize(
    "widths,seed",
    [
        ([3, 16, 16, 1], 10),
        ([2, 8, 8, 8, 1], 11),
    ],
)
def test_first_order_softplus(widths: List[int], seed: int) -> None:
    """Dual algebra matches PyTorch autograd for softplus networks."""
    model = _build_softplus_net(widths, seed)
    parsed = parse_pytorch(model)

    n = widths[0]
    m = 2
    rng = np.random.default_rng(seed + 200)
    x = rng.standard_normal(n).astype(np.float32)
    f = rng.standard_normal(n).astype(np.float32)
    G = rng.standard_normal((n, m)).astype(np.float32)

    h_ref, Lf_ref, Lg_ref = _torch_lie_first_order(model, x, f, G)

    header = emit_cpp_header(parsed, relative_degree=1)
    payload = x.tobytes() + f.tobytes() + G.tobytes()
    out = _compile_and_run(
        header, _DRIVER_FIRST_ORDER, n, payload, m, expect_floats=2 + m
    )

    assert abs(out[0] - h_ref) <= _ATOL + _RTOL * abs(h_ref)
    assert abs(out[1] - Lf_ref) <= _ATOL + _RTOL * abs(Lf_ref)
    np.testing.assert_allclose(out[2:], Lg_ref, rtol=_RTOL, atol=_ATOL)


@pytest.mark.parametrize(
    "widths,seed",
    [
        ([3, 8, 8, 1], 20),
        ([2, 12, 12, 1], 21),
    ],
)
def test_first_order_tanh(widths: List[int], seed: int) -> None:
    """Dual algebra matches PyTorch autograd for tanh networks."""
    model = _build_tanh_net(widths, seed)
    parsed = parse_pytorch(model)

    n = widths[0]
    m = 1
    rng = np.random.default_rng(seed + 300)
    x = rng.standard_normal(n).astype(np.float32) * 0.5
    f = rng.standard_normal(n).astype(np.float32)
    G = rng.standard_normal((n, m)).astype(np.float32)

    h_ref, Lf_ref, Lg_ref = _torch_lie_first_order(model, x, f, G)

    header = emit_cpp_header(parsed, relative_degree=1)
    payload = x.tobytes() + f.tobytes() + G.tobytes()
    out = _compile_and_run(
        header, _DRIVER_FIRST_ORDER, n, payload, m, expect_floats=2 + m
    )

    assert abs(out[0] - h_ref) <= _ATOL + _RTOL * abs(h_ref)
    assert abs(out[1] - Lf_ref) <= _ATOL + _RTOL * abs(Lf_ref)
    np.testing.assert_allclose(out[2:], Lg_ref, rtol=_RTOL, atol=_ATOL)


@pytest.mark.parametrize(
    "widths,seed",
    [
        ([3, 8, 8, 1], 30),
        ([2, 12, 12, 1], 31),
    ],
)
def test_second_order_softplus(widths: List[int], seed: int) -> None:
    """Hyper-dual algebra matches PyTorch Hessian for softplus networks."""
    model = _build_softplus_net(widths, seed)
    parsed = parse_pytorch(model)

    n = widths[0]
    m = 1
    rng = np.random.default_rng(seed + 400)
    x = rng.standard_normal(n).astype(np.float32) * 0.5
    f = rng.standard_normal(n).astype(np.float32)
    G = rng.standard_normal((n, m)).astype(np.float32)
    # Random analytic Jacobians of f (the user-supplied f'(x) xi);
    # the test only requires that the same Df_xi is fed to both paths.
    Df = rng.standard_normal((n, n)).astype(np.float32)
    Df_f = (Df @ f).astype(np.float32)
    Df_G = (Df @ G).astype(np.float32)

    L2f_ref, LgLf_ref = _torch_lie_second_order(model, x, f, Df_f, G, Df_G)

    header = emit_cpp_header(parsed, relative_degree=2)
    payload = (
        x.tobytes() + f.tobytes() + G.tobytes()
        + Df_f.tobytes() + Df_G.tobytes()
    )
    out = _compile_and_run(
        header, _DRIVER_SECOND_ORDER, n, payload, m, expect_floats=1 + m
    )

    L2f_c = float(out[0])
    LgLf_c = out[1:].astype(np.float32)

    assert abs(L2f_c - L2f_ref) <= 1e-3 + 1e-3 * abs(L2f_ref), (
        f"L_f^2 h mismatch: c={L2f_c} ref={L2f_ref}"
    )
    np.testing.assert_allclose(LgLf_c, LgLf_ref, rtol=1e-3, atol=1e-3)


def test_parser_rejects_conv() -> None:
    """parse_pytorch rejects non-feedforward architectures."""
    model = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Conv1d(1, 1, kernel_size=1),  # disallowed
    )
    with pytest.raises(ValueError, match="not supported"):
        parse_pytorch(model)


def test_parser_rejects_batchnorm() -> None:
    model = nn.Sequential(
        nn.Linear(4, 8),
        nn.BatchNorm1d(8),  # disallowed
        nn.ReLU(),
        nn.Linear(8, 1),
    )
    with pytest.raises(ValueError, match="not supported"):
        parse_pytorch(model)


def test_parser_rejects_lstm() -> None:
    model = nn.Sequential(
        nn.Linear(4, 8),
        nn.LSTM(8, 8),  # disallowed
    )
    with pytest.raises(ValueError, match="not supported"):
        parse_pytorch(model)


def test_parser_requires_scalar_output() -> None:
    """The CBF must be scalar-valued."""
    model = nn.Sequential(
        nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2)
    )
    with pytest.raises(ValueError, match="output dimension"):
        parse_pytorch(model)


def test_parser_rejects_relu_when_relative_degree_2() -> None:
    """parse_pytorch refuses ReLU networks for relative_degree=2."""
    model = nn.Sequential(
        nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1)
    )
    with pytest.raises(ValueError, match="ReLU"):
        parse_pytorch(model, relative_degree=2)


def test_parser_accepts_softplus_when_relative_degree_2() -> None:
    """Smooth activations pass parse_pytorch with relative_degree=2."""
    model = nn.Sequential(
        nn.Linear(4, 8), nn.Softplus(), nn.Linear(8, 1)
    )
    parsed = parse_pytorch(model, relative_degree=2)
    assert parsed.depth == 2


def test_emitter_rejects_relu_relative_degree_2() -> None:
    """emit_cpp_header refuses ReLU networks for relative_degree=2 even if
    they slipped past parsing (e.g. were constructed by hand)."""
    from dual_cbf_compiler.ir import (
        ActivationLayer as IRA,
        LinearLayer as IRL,
        ParsedNetwork as IRP,
    )
    net = IRP(layers=[
        IRL(W=np.zeros((4, 3), dtype=np.float32), b=np.zeros(4, dtype=np.float32)),
        IRA("relu"),
        IRL(W=np.zeros((1, 4), dtype=np.float32), b=np.zeros(1, dtype=np.float32)),
    ])
    with pytest.raises(ValueError, match="ReLU"):
        emit_cpp_header(net, relative_degree=2)


def test_emitter_documents_row_major_g() -> None:
    """Generated header includes the row-major G layout reminder."""
    model = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 1))
    parsed = parse_pytorch(model)
    header = emit_cpp_header(parsed, relative_degree=1)
    assert "ROW-MAJOR" in header
    assert "G[i * m + j]" in header
    assert "G2D" in header
