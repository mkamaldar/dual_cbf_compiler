"""End-to-end example: compile a small CBF and assemble the QP constraint.

This mirrors how the package is intended to be used in practice:

  1. Build (or load) a trained feedforward neural CBF.
  2. Parse it into the unified IR.
  3. Emit a self-contained C++ header.
  4. Compile the header against a tiny driver and run.

Run directly:  python3 example_pipeline.py
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np

from dual_cbf_compiler import emit_cpp_header
from dual_cbf_compiler.ir import ActivationLayer, LinearLayer, ParsedNetwork


def build_demo_cbf() -> ParsedNetwork:
    """Build a deterministic 4-32-32-1 ReLU CBF (kinematic-bicycle scale)."""
    rng = np.random.default_rng(42)
    layers = []
    widths = [4, 32, 32, 1]
    for i in range(len(widths) - 1):
        W = (rng.standard_normal((widths[i + 1], widths[i]))
             * np.sqrt(2.0 / widths[i])).astype(np.float32)
        b = (rng.standard_normal(widths[i + 1]) * 0.05).astype(np.float32)
        layers.append(LinearLayer(W=W, b=b))
        if i < len(widths) - 2:
            layers.append(ActivationLayer("relu"))
    net = ParsedNetwork(layers=layers)
    net.validate()
    return net


_DRIVER = r"""
#include <cstdio>
#include <cstdlib>
#include "dual_cbf.h"

int main() {
    // Kinematic bicycle state (px, py, psi, v) and dynamics
    float x[4]   = {1.0f, 2.0f, 0.3f, 5.0f};
    float f[4]   = {x[3] * cosf(x[2]), x[3] * sinf(x[2]), 0.0f, 0.0f};
    // G(x) column-major: 4x2
    //   col 0 = [0, 0, 0, 1]^T  (acceleration input)
    //   col 1 = [0, 0, v/L, 0]^T  (steering input, L = 2.5 m)
    int m = 2;
    float G[4 * 2] = {
        0.0f, 0.0f,        // row 0
        0.0f, 0.0f,        // row 1
        0.0f, x[3] / 2.5f, // row 2
        1.0f, 0.0f,        // row 3
    };

    float h, Lf, Lg[2];
    dual_cbf::evaluate_cbf(x, f, G, m, &h, &Lf, Lg);

    printf("h    = %+e\n", h);
    printf("L_f  = %+e\n", Lf);
    printf("L_G  = [%+e, %+e]\n", Lg[0], Lg[1]);
    // CBF QP constraint: L_f h + L_G h * u >= -alpha(h)
    // For a linear class-K function alpha(s) = k * s with k = 1:
    float k = 1.0f;
    float rhs = -k * h;
    printf("constraint:  L_f + L_G * u >= %+e\n", rhs);
    printf("(at u = 0):  %+e >= %+e -> %s\n",
           Lf, rhs, (Lf >= rhs ? "feasible" : "infeasible"));
    return 0;
}
"""


def main() -> int:
    print("[1/4] Building demo CBF (4-32-32-1 ReLU)...")
    network = build_demo_cbf()
    print(f"      depth={network.depth} widths={network.widths}")

    print("[2/4] Emitting C++ header...")
    header = emit_cpp_header(network, relative_degree=1)
    n_lines = header.count("\n")
    print(f"      generated {len(header)} chars, {n_lines} lines")

    print("[3/4] Compiling and running...")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "dual_cbf.h").write_text(header)
        (td / "main.cpp").write_text(_DRIVER)
        binary = td / "demo"
        cc = subprocess.run(
            ["g++", "-O2", "-std=c++17", "-I", str(td),
             str(td / "main.cpp"), "-o", str(binary)],
            capture_output=True, text=True,
        )
        if cc.returncode != 0:
            print("g++ failed:", cc.stderr)
            return 1
        run = subprocess.run([str(binary)], capture_output=True, text=True)
        if run.returncode != 0:
            print("driver failed:", run.stderr)
            return 1
        print("[4/4] Driver output:")
        for line in run.stdout.splitlines():
            print(f"      {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
