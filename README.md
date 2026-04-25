# dual_cbf_compiler

Ahead-of-time compiler that translates trained neural Control Barrier Functions
into bare-metal C++ headers evaluating exact Lie derivatives via dual algebra.
The generated code performs zero dynamic memory allocation and is suitable for
deployment on resource-constrained microcontrollers running kilohertz-rate
safety filters.

## Install

```bash
pip install -e ".[test]"
```

## Use

### From PyTorch

```python
import torch.nn as nn
from dual_cbf_compiler import parse_pytorch, emit_cpp_header

model = nn.Sequential(
    nn.Linear(4, 32), nn.ReLU(),
    nn.Linear(32, 32), nn.ReLU(),
    nn.Linear(32, 1),
)
# (load trained weights into model here)

network = parse_pytorch(model)
header = emit_cpp_header(network, relative_degree=1)
open("dual_cbf.h", "w").write(header)
```

### From ONNX

```python
from dual_cbf_compiler import parse_onnx, emit_cpp_header

network = parse_onnx("trained_cbf.onnx")
header = emit_cpp_header(network, relative_degree=2)  # 2nd-order Lie derivatives
open("dual_cbf.h", "w").write(header)
```

### From the command line

```bash
dual-cbf-compile model.onnx -o dual_cbf.h --relative-degree 1
```

## Use the generated C++ header

```cpp
#include "dual_cbf.h"

float x[4]   = {/* state */};
float f[4]   = {/* drift  */};
float G[4*2] = {/* input field, column-major */};

float h, Lf, Lg[2];
dual_cbf::evaluate_cbf(x, f, G, /*m=*/2, &h, &Lf, Lg);

// Now use (h, Lf, Lg) to assemble the CBF QP constraint:
//     L_f h + L_G h * u >= -alpha(h)
```

## Verify

```bash
pytest -v dual_cbf_compiler/verify.py
```

The suite builds random PyTorch networks, evaluates Lie derivatives via
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

## Layout

```
dual_cbf_compiler/
  ir.py        — Internal representation (LinearLayer, ActivationLayer)
  parser.py    — parse_pytorch, parse_onnx
  emitter.py   — emit_cpp_header
  verify.py    — Pytest suite (PyTorch autograd vs g++-compiled binary)
  cli.py       — dual-cbf-compile entry point
setup.py
```
