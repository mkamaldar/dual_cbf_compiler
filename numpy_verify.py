"""Numpy-only verification: end-to-end exact derivative test.

This standalone harness reproduces the spirit of verify.py without
requiring PyTorch. It builds a random feedforward network, computes the
exact spatial gradient analytically through the chain rule (which is
straightforward for ReLU/softplus/tanh networks), then compares against
the dual-algebraic C++ output.
"""

from __future__ import annotations

import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

# Make the local package importable
sys.path.insert(0, str(Path(__file__).parent))

from dual_cbf_compiler.emitter import emit_cpp_header
from dual_cbf_compiler.ir import ActivationLayer, LinearLayer, ParsedNetwork


# ---------- analytic chain-rule reference ----------

def _activation(kind: str, x: np.ndarray) -> np.ndarray:
    if kind == "relu":
        return np.maximum(x, 0.0)
    if kind == "softplus":
        out = np.where(x > 20.0, x, np.log1p(np.exp(x)))
        return out.astype(np.float32)
    if kind == "tanh":
        return np.tanh(x).astype(np.float32)
    if kind == "sigmoid":
        return (1.0 / (1.0 + np.exp(-x))).astype(np.float32)
    if kind == "identity":
        return x
    raise ValueError(kind)


def _activation_p(kind: str, x: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    if kind == "relu":
        return (x > 0.0).astype(np.float32)
    if kind == "softplus":
        return (1.0 / (1.0 + np.exp(-x))).astype(np.float32)
    if kind == "tanh":
        return (1.0 - sigma * sigma).astype(np.float32)
    if kind == "sigmoid":
        return (sigma * (1.0 - sigma)).astype(np.float32)
    if kind == "identity":
        return np.ones_like(x, dtype=np.float32)
    raise ValueError(kind)


def reference_h_and_grad(network: ParsedNetwork, x: np.ndarray) -> tuple[float, np.ndarray]:
    """Forward + reverse pass through the IR to compute h and gradient."""
    a = x.astype(np.float32).copy()
    pre_acts: list[np.ndarray] = []
    sig_primes: list[np.ndarray] = []
    activations: list[str] = []

    for layer in network.layers:
        if isinstance(layer, LinearLayer):
            a = (layer.W @ a + layer.b).astype(np.float32)
            pre_acts.append(a.copy())
        else:
            sig = _activation(layer.kind, a).astype(np.float32)
            sig_primes.append(_activation_p(layer.kind, a, sig).astype(np.float32))
            activations.append(layer.kind)
            a = sig

    h = float(a[0])

    # Reverse pass
    delta = np.array([1.0], dtype=np.float32)
    linears = network.linear_layers
    activation_layers = network.activation_layers

    n_linear = len(linears)
    for i in range(n_linear - 1, -1, -1):
        if i < n_linear - 1:
            # multiply by sigma'(a_{i+1}) (the activation following linear i)
            delta = delta * sig_primes[i]
        delta = linears[i].W.T @ delta

    return h, delta.astype(np.float32)


# ---------- harness ----------

def build_random_network(widths, kind: str, seed: int) -> ParsedNetwork:
    rng = np.random.default_rng(seed)
    layers = []
    n_lin = len(widths) - 1
    for i in range(n_lin):
        W = (rng.standard_normal((widths[i+1], widths[i])) * np.sqrt(2.0 / widths[i])).astype(np.float32)
        b = rng.standard_normal(widths[i+1]).astype(np.float32) * 0.1
        layers.append(LinearLayer(W=W, b=b))
        if i < n_lin - 1:
            layers.append(ActivationLayer(kind=kind))
    net = ParsedNetwork(layers=layers)
    net.validate()
    return net


_DRIVER_TEMPLATE = r"""
#include <cstdio>
#include <cstdlib>
#include "dual_cbf.h"

int main(int argc, char** argv) {
    int m = atoi(argv[3]);
    int n = dual_cbf::INPUT_DIM;
    float x[%(N)d], f[%(N)d];
    float G[%(N)d * 64];
    FILE* fp = fopen(argv[1], "rb");
    fread(x, sizeof(float), n, fp);
    fread(f, sizeof(float), n, fp);
    fread(G, sizeof(float), n * m, fp);
    fclose(fp);
    float h, Lf;
    float Lg[64];
    dual_cbf::evaluate_cbf(x, f, G, m, &h, &Lf, Lg);
    FILE* op = fopen(argv[2], "wb");
    fwrite(&h, 4, 1, op);
    fwrite(&Lf, 4, 1, op);
    fwrite(Lg, 4, m, op);
    fclose(op);
    return 0;
}
"""


def compile_and_run(network: ParsedNetwork, x, f, G):
    n = network.input_dim
    m = G.shape[1]
    header = emit_cpp_header(network, relative_degree=1)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "dual_cbf.h").write_text(header)
        (td / "driver.cpp").write_text(_DRIVER_TEMPLATE % {"N": n})
        binary = td / "driver"
        cmd = ["g++", "-O2", "-std=c++17", "-I", str(td),
               str(td / "driver.cpp"), "-o", str(binary)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(r.stderr)
        in_file = td / "in.bin"
        out_file = td / "out.bin"
        in_file.write_bytes(x.tobytes() + f.tobytes() + G.tobytes())
        run = subprocess.run([str(binary), str(in_file), str(out_file), str(m)],
                             capture_output=True, text=True)
        if run.returncode != 0:
            raise RuntimeError(run.stderr)
        out = np.frombuffer(out_file.read_bytes(), dtype=np.float32).copy()
    return float(out[0]), float(out[1]), out[2:].copy()


def run_test(name: str, widths, kind: str, seed: int, m: int = 2):
    net = build_random_network(widths, kind, seed)
    rng = np.random.default_rng(seed + 999)
    x = rng.standard_normal(net.input_dim).astype(np.float32)
    if kind == "relu":
        x[np.abs(x) < 0.1] += 0.2 * np.sign(x[np.abs(x) < 0.1] + 1e-6)
    elif kind == "tanh":
        x *= 0.5
    f = rng.standard_normal(net.input_dim).astype(np.float32)
    G = rng.standard_normal((net.input_dim, m)).astype(np.float32)

    h_ref, grad = reference_h_and_grad(net, x)
    Lf_ref = float(grad @ f)
    Lg_ref = (grad @ G).astype(np.float32)

    h_c, Lf_c, Lg_c = compile_and_run(net, x, f, G)

    rtol, atol = 5e-5, 5e-5
    ok_h = abs(h_c - h_ref) <= atol + rtol * abs(h_ref)
    ok_lf = abs(Lf_c - Lf_ref) <= atol + rtol * abs(Lf_ref)
    ok_lg = bool(np.all(np.abs(Lg_c - Lg_ref) <= atol + rtol * np.abs(Lg_ref)))
    status = "PASS" if (ok_h and ok_lf and ok_lg) else "FAIL"
    print(f"[{status}] {name:32s} widths={widths} act={kind:8s} "
          f"h:({h_ref:+.6e} vs {h_c:+.6e})  "
          f"Lf:({Lf_ref:+.6e} vs {Lf_c:+.6e})  "
          f"max|dLg|={float(np.max(np.abs(Lg_c-Lg_ref))):.2e}")
    return status == "PASS"


def main():
    tests = [
        ("relu_tiny",     [3, 8, 8, 1],     "relu",     0),
        ("relu_medium",   [4, 32, 32, 1],   "relu",     1),
        ("relu_deep",     [3, 16, 16, 16, 1], "relu",   2),
        ("softplus_med",  [3, 16, 16, 1],   "softplus", 10),
        ("softplus_deep", [2, 8, 8, 8, 1],  "softplus", 11),
        ("tanh_med",      [3, 12, 12, 1],   "tanh",     20),
        ("tanh_deep",     [2, 8, 8, 8, 1],  "tanh",     21),
        ("sigmoid_med",   [3, 16, 16, 1],   "sigmoid",  30),
    ]
    results = [run_test(name, widths, kind, seed) for name, widths, kind, seed in tests]
    n_pass = sum(results)
    n_total = len(results)
    print(f"\n{n_pass}/{n_total} tests passed.")
    return 0 if n_pass == n_total else 1


if __name__ == "__main__":
    raise SystemExit(main())
