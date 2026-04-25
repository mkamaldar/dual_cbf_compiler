"""Numpy-only verification of the relative-degree-2 hyper-dual emitter."""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from dual_cbf_compiler.emitter import emit_cpp_header
from dual_cbf_compiler.ir import ActivationLayer, LinearLayer, ParsedNetwork
from numpy_verify import build_random_network


def _activation(kind, x):
    if kind == "softplus":
        return np.where(x > 20.0, x, np.log1p(np.exp(x))).astype(np.float64)
    if kind == "tanh":
        return np.tanh(x).astype(np.float64)
    raise ValueError(kind)

def _activation_p(kind, x, sigma):
    if kind == "softplus":
        return (1.0 / (1.0 + np.exp(-x))).astype(np.float64)
    if kind == "tanh":
        return (1.0 - sigma * sigma).astype(np.float64)
    raise ValueError(kind)

def _activation_pp(kind, x, sigma, sig_p):
    if kind == "softplus":
        # softplus'' = sig * (1 - sig)
        return (sig_p * (1.0 - sig_p)).astype(np.float64)
    if kind == "tanh":
        return (-2.0 * sigma * sig_p).astype(np.float64)
    raise ValueError(kind)


def reference_grad_and_hessian(network: ParsedNetwork, x: np.ndarray):
    """Compute h, grad h, and Hessian H by analytic forward+reverse-over-forward."""
    a = x.astype(np.float64).copy()
    n_in = a.shape[0]

    pre_acts = []
    sigs = []
    sig_primes = []
    sig_pps = []
    activations = []

    for layer in network.layers:
        if isinstance(layer, LinearLayer):
            a = layer.W.astype(np.float64) @ a + layer.b.astype(np.float64)
            pre_acts.append(a.copy())
        else:
            sig = _activation(layer.kind, a)
            sp = _activation_p(layer.kind, a, sig)
            spp = _activation_pp(layer.kind, a, sig, sp)
            sigs.append(sig)
            sig_primes.append(sp)
            sig_pps.append(spp)
            activations.append(layer.kind)
            a = sig

    h = float(a[0])

    # Compute Jacobian h'(x) = d h/d x by hand using chain rule:
    # h'(x) = W_d * D_{d-1} * W_{d-1} * D_{d-2} * ... * D_1 * W_1
    # where D_i = diag(sig'_i(a_i))
    linears = network.linear_layers
    # accumulate Jacobian J = h'(x) row-vector of shape (1, n_in)
    J = linears[-1].W.astype(np.float64)  # (1, n_{d-1})
    for i in range(len(linears) - 2, -1, -1):
        D = np.diag(sig_primes[i])
        J = J @ D @ linears[i].W.astype(np.float64)
    grad = J.reshape(-1)

    # Hessian via finite difference of the gradient (we already have an exact
    # gradient evaluator -- reuse it for numerical Hessian as ground truth).
    eps = 1e-3
    H = np.zeros((n_in, n_in), dtype=np.float64)
    for k in range(n_in):
        xp = x.copy(); xp[k] += eps
        xm = x.copy(); xm[k] -= eps
        gp = _grad_only(network, xp)
        gm = _grad_only(network, xm)
        H[k, :] = (gp - gm) / (2 * eps)
    H = 0.5 * (H + H.T)

    return h, grad, H


def _grad_only(network, x):
    a = x.astype(np.float64).copy()
    pre_acts = []
    sig_primes = []
    for layer in network.layers:
        if isinstance(layer, LinearLayer):
            a = layer.W.astype(np.float64) @ a + layer.b.astype(np.float64)
            pre_acts.append(a.copy())
        else:
            sig = _activation(layer.kind, a)
            sp = _activation_p(layer.kind, a, sig)
            sig_primes.append(sp)
            a = sig
    linears = network.linear_layers
    J = linears[-1].W.astype(np.float64)
    for i in range(len(linears) - 2, -1, -1):
        D = np.diag(sig_primes[i])
        J = J @ D @ linears[i].W.astype(np.float64)
    return J.reshape(-1)


_DRIVER_TEMPLATE = r"""
#include <cstdio>
#include <cstdlib>
#include "dual_cbf.h"
int main(int argc, char** argv) {
    int m = atoi(argv[3]);
    int n = dual_cbf::INPUT_DIM;
    float x[%(N)d], f[%(N)d], Df_f[%(N)d];
    float G[%(N)d * 64], Df_G[%(N)d * 64];
    FILE* fp = fopen(argv[1], "rb");
    fread(x, 4, n, fp);
    fread(f, 4, n, fp);
    fread(G, 4, n*m, fp);
    fread(Df_f, 4, n, fp);
    fread(Df_G, 4, n*m, fp);
    fclose(fp);
    float L2f, LgLf[64];
    dual_cbf::evaluate_cbf_2nd_order(x, f, G, Df_f, Df_G, m, &L2f, LgLf);
    FILE* op = fopen(argv[2], "wb");
    fwrite(&L2f, 4, 1, op);
    fwrite(LgLf, 4, m, op);
    fclose(op);
    return 0;
}
"""


def compile_and_run_2nd(network, x, f, G, Df_f, Df_G):
    n = network.input_dim
    m = G.shape[1]
    header = emit_cpp_header(network, relative_degree=2)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "dual_cbf.h").write_text(header)
        (td / "driver.cpp").write_text(_DRIVER_TEMPLATE % {"N": n})
        binary = td / "driver"
        r = subprocess.run(
            ["g++", "-O2", "-std=c++17", "-I", str(td),
             str(td / "driver.cpp"), "-o", str(binary)],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            raise RuntimeError(r.stderr)
        in_file = td / "in.bin"
        out_file = td / "out.bin"
        in_file.write_bytes(
            x.astype(np.float32).tobytes() + f.astype(np.float32).tobytes()
            + G.astype(np.float32).tobytes()
            + Df_f.astype(np.float32).tobytes() + Df_G.astype(np.float32).tobytes()
        )
        run = subprocess.run([str(binary), str(in_file), str(out_file), str(m)],
                             capture_output=True, text=True)
        if run.returncode != 0:
            raise RuntimeError(run.stderr)
        out = np.frombuffer(out_file.read_bytes(), dtype=np.float32).copy()
    return float(out[0]), out[1:].copy()


def run_2nd_order_test(name, widths, kind, seed, m=2):
    rng = np.random.default_rng(seed)
    net = build_random_network(widths, kind, seed)
    x = rng.standard_normal(net.input_dim).astype(np.float32) * 0.5
    f = rng.standard_normal(net.input_dim).astype(np.float32)
    G = rng.standard_normal((net.input_dim, m)).astype(np.float32)
    # User-supplied Jacobian-vector products of f
    Df = rng.standard_normal((net.input_dim, net.input_dim)).astype(np.float32)
    Df_f = (Df @ f).astype(np.float32)
    Df_G = (Df @ G).astype(np.float32)

    h, grad, H = reference_grad_and_hessian(net, x.astype(np.float64))
    L2f_ref = float(f.astype(np.float64) @ H @ f.astype(np.float64) + grad @ Df_f.astype(np.float64))
    LgLf_ref = np.array([
        float(f.astype(np.float64) @ H @ G[:, j].astype(np.float64) + grad @ Df_G[:, j].astype(np.float64))
        for j in range(m)
    ], dtype=np.float64)

    L2f_c, LgLf_c = compile_and_run_2nd(net, x, f, G, Df_f, Df_G)

    # Hessian comes from finite-difference reference, so tolerance is wider
    err_l2f = abs(L2f_c - L2f_ref) / (1.0 + abs(L2f_ref))
    err_lglf = float(np.max(np.abs(LgLf_c - LgLf_ref) / (1.0 + np.abs(LgLf_ref))))
    ok = err_l2f < 1e-3 and err_lglf < 1e-3
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name:24s} widths={widths} act={kind:8s} "
          f"L2f:({L2f_ref:+.4e} vs {L2f_c:+.4e}) "
          f"max_rel_LgLf={err_lglf:.2e}")
    return ok


def main():
    tests = [
        ("hyper_softplus_3x16", [3, 16, 16, 1], "softplus", 100),
        ("hyper_softplus_2x12", [2, 12, 12, 1], "softplus", 101),
        ("hyper_tanh_3x12",     [3, 12, 12, 1], "tanh",     200),
        ("hyper_tanh_2x8x8",    [2, 8, 8, 8, 1],"tanh",     201),
    ]
    results = [run_2nd_order_test(*t) for t in tests]
    print(f"\n{sum(results)}/{len(results)} 2nd-order tests passed.")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
