# dual_cbf_compiler

Ahead-of-time compiler that translates trained neural Control Barrier Functions
into bare-metal C++ headers evaluating exact Lie derivatives via dual algebra.
The generated code performs zero dynamic memory allocation and is suitable for
deployment on resource-constrained microcontrollers running kilohertz-rate
safety filters.

## Status and intended use

This is research software released to accompany peer-reviewed publications.
It is **not** certified to ISO 26262, DO-178C, IEC 61508, or any other
functional-safety standard, and it ships without warranty (see License).
The generated headers are meant for research, teaching, and prototyping on
embedded targets. Anyone deploying them on a system that can injure people
or damage property is responsible for their own verification, validation,
and certification.

Please record the exact release (`pip show dual_cbf_compiler`, or the version
string written into the header banner) alongside any generated `dual_cbf.h`
so that results are reproducible.

## Install

```bash
pip install -e ".[test]"
```

## Recommended workflow: ONNX-first

The cleanest path is to **export your trained PyTorch model to ONNX**. ONNX
captures the network topology along with the weights, so no manual architecture
declaration is required:

```python
import torch
torch.onnx.export(
    model,
    sample_input,
    "model.onnx",
    input_names=["x"],
    output_names=["h"],
    opset_version=14,
)
```

then either

```bash
dual-cbf-compile model.onnx -o dual_cbf.h --relative-degree 1
```

or programmatically

```python
from dual_cbf_compiler import parse_onnx, emit_cpp_header

network = parse_onnx("model.onnx")
header = emit_cpp_header(network, relative_degree=1)
open("dual_cbf.h", "w").write(header)
```

## Direct PyTorch ingestion (programmatic, recommended)

If you have an in-memory `torch.nn.Sequential` whose modules describe the
topology, parse it directly — no string declaration required:

```python
import torch.nn as nn
from dual_cbf_compiler import parse_pytorch, emit_cpp_header

model = nn.Sequential(
    nn.Linear(4, 32), nn.ReLU(),
    nn.Linear(32, 32), nn.ReLU(),
    nn.Linear(32, 1),
)
# (load trained weights into model here)

network = parse_pytorch(model)            # relative_degree=1 by default
header  = emit_cpp_header(network, relative_degree=1)
open("dual_cbf.h", "w").write(header)
```

## PyTorch state_dict from the command line (fallback only)

PyTorch state_dicts (`.pt`/`.pth`) only save tensor weights, not the
computational graph, so the CLI must be told the topology via `--torch-arch`.
**This is error-prone — a typo silently maps weights to the wrong dimensions.**
Prefer the ONNX export path above.

```bash
# Only if ONNX export is unavailable
dual-cbf-compile model.pt -o dual_cbf.h \
    --torch-arch "4,relu,32,relu,32,linear,1"
```

## Use the generated C++ header

```cpp
#include "dual_cbf.h"

float x[4]   = {/* state */};
float f[4]   = {/* drift  */};

// IMPORTANT: G(x) is ROW-MAJOR with shape (n x m).
// Element G(i, j) lives at G[i * m + j].
// If you keep G as a 2D C array float G2D[4][2], populate the flat
// layout as:
//   for (int i = 0; i < 4; ++i)
//       for (int j = 0; j < 2; ++j)
//           G[i * 2 + j] = G2D[i][j];
float G[4 * 2] = {
    /* row 0 */ 0.0f,    0.0f,
    /* row 1 */ 0.0f,    0.0f,
    /* row 2 */ 0.0f,    0.4f,
    /* row 3 */ 1.0f,    0.0f,
};

float h, Lf, Lg[2];
dual_cbf::evaluate_cbf(x, f, G, /*m=*/2, &h, &Lf, Lg);

// CBF QP constraint:  L_f h + L_G h * u >= -alpha(h)
```

The generated header repeats this row-major layout reminder in its file-level
comment so the convention is impossible to miss at the call site.

## Relative-degree-2 (hyper-dual)

For mechanical systems with acceleration inputs:

```python
network = parse_onnx("softplus_model.onnx", relative_degree=2)
header  = emit_cpp_header(network, relative_degree=2)
```

```cpp
float L2f, LgLf[m];
dual_cbf::evaluate_cbf_2nd_order(x, f, G, Df_f, Df_G, m, &L2f, LgLf);
```

`Df_f` is the user-supplied Jacobian-vector product `f'(x) f(x)` (length `n`),
and `Df_G` is the row-major Jacobian-vector product `f'(x) G(x)` (shape `n×m`).

**ReLU is rejected for relative_degree=2.** ReLU's second derivative vanishes
almost everywhere, so the Hessian contribution to `L_f^2 h_theta` and
`L_{col_j(G)} L_f h_theta` would be silently lost. Both `parse_pytorch` and
`parse_onnx` raise a `ValueError` at parse time if a ReLU layer is detected
and `relative_degree=2` was requested. Retrain with **Softplus**, **Tanh**,
or **Sigmoid** instead.

## Verify

```bash
# Numpy-only verification (no torch needed)
python3 numpy_verify.py
python3 numpy_verify_2nd.py

# Full suite (requires torch)
pytest -v dual_cbf_compiler/verify.py
```

The pytest suite builds random PyTorch networks, evaluates Lie derivatives via
`torch.autograd`, compiles the generated C++ via `g++`, runs the binary, and
asserts agreement to single-precision machine epsilon.

## Design

* **Zero dynamic allocation.** All weights and biases are embedded as
  `static const float` arrays. A single shared static scratch buffer of
  `2 * max(n_i)` floats (relative-degree 1) or `4 * max(n_i)` floats
  (relative-degree 2) is destructively overwritten as the forward pass
  advances layer by layer.
* **Forward-only.** The chain rule is collapsed into a memoryless forward
  recurrence by encoding the system state as the real part and a vector
  field as the dual part of an input dual vector. No reverse-mode AD,
  no computational graph.
* **Algebraic mode selection.** `relative_degree=1` emits standard dual
  algebra (eps^2 = 0). `relative_degree=2` emits hyper-dual arithmetic
  with cross term eps_12, allowing exact extraction of the second-order
  Lie derivatives required for acceleration-controlled mechanical systems.
* **Hard guards on common footguns.** ReLU + relative_degree=2 is a
  `ValueError`, not a passive warning. The CLI declines to silently map
  weights to the wrong topology — when `--torch-arch` and the state_dict
  disagree, it refuses with a hint to use ONNX.

## Layout

## Assumptions and limitations

- **Relative degree 1.** The network may use ReLU. At a ReLU kink the emitted
  code returns the one-sided derivative selected by the forward pass; see the
  non-smooth paper below for the conditions under which this preserves the
  barrier certificate.
- **Relative degree 2 (hyper-dual).** Requires twice-differentiable
  activations (e.g. softplus, tanh). The compiler rejects ReLU in this mode
  on purpose.
- **Dynamics are supplied by you.** The header evaluates `h`, `L_f h`, and
  `L_g h` (and second-order terms) for the `f`, `G` you pass in. Model error,
  `G` layout (row-major, `n x m`), and units are the caller's responsibility.
- **The QP is not included.** Feasibility of the CBF constraint, the choice of
  `alpha(h)`, sampling-rate effects, and actuator limits must be handled in
  the caller's control loop.
- **Numerics.** Emitted code uses `float` (32-bit). Use `double` on targets
  that need it by editing the generated typedef.





## License


Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) and
[NOTICE](NOTICE). 
