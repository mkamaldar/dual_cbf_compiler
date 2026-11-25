# dual-cbf-compiler

[![PyPI](https://img.shields.io/pypi/v/dual-cbf-compiler.svg)](https://pypi.org/project/dual-cbf-compiler/)
[![Python](https://img.shields.io/pypi/pyversions/dual-cbf-compiler.svg)](https://pypi.org/project/dual-cbf-compiler/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

An ahead-of-time compiler that translates a trained neural control barrier
function (CBF) into one self-contained C++ header. The header evaluates the
barrier and its Lie derivatives by forward-mode differentiation with dual
numbers, with buffers of fixed size and without dynamic memory allocation.

This repository accompanies the paper

> M. Kamaldar, "[Extracting exact Lie derivatives without backpropagation: A dual compiler for neural control barrier functions][paper]," 2026.

The measurements of the paper on an ESP32-S3 microcontroller are produced by
a separate repository, [`dual_cbf_esp32_experiments`][experiments].

[paper]: https://arxiv.org/pdf/2604.23995
[experiments]: https://github.com/mkamaldar/dual_cbf_esp32_experiments
[pypi]: https://pypi.org/project/dual-cbf-compiler/

## Contents

- [What the compiler does](#what-the-compiler-does)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Entry points of the emitted header](#entry-points-of-the-emitted-header)
- [Relative degree two](#relative-degree-two)
- [Command-line reference](#command-line-reference)
- [Python interface](#python-interface)
- [Memory of the emitted code](#memory-of-the-emitted-code)
- [Tests](#tests)
- [Assumptions and limitations](#assumptions-and-limitations)
- [Status and intended use](#status-and-intended-use)
- [Repository layout](#repository-layout)
- [Citation](#citation)
- [License](#license)

## What the compiler does

Consider the control-affine system `dx/dt = f(x) + G(x) u` with state
`x` in `R^n` and input `u` in `R^m`, and a barrier `h` that is a feedforward
neural network from `R^n` to `R`. A CBF safety filter solves, at each control
cycle, a quadratic program whose constraint is

```
L_f h(x) + L_G h(x) u + alpha(h(x)) >= 0,
```

where `L_f h(x) = h'(x) f(x)` and `L_G h(x) = h'(x) G(x)` are the Lie
derivatives of `h` along `f` and `G`. The filter therefore needs `h(x)`,
`L_f h(x)`, and `L_G h(x)` at each cycle.

A dual number `a + b eps` has a real part `a` and a dual part `b`, where
`eps^2 = 0`. Evaluating the network at the dual input `x + xi eps` returns

```
h(x + xi eps) = h(x) + (h'(x) xi) eps,
```

so that one forward pass with `xi = f(x)` returns `h(x)` and `L_f h(x)`, and
one forward pass with `xi` equal to column `j` of `G(x)` returns component
`j` of `L_G h(x)`. No intermediate value of the forward pass is stored for a
backward pass, and the working memory depends on the largest layer width and
not on the depth of the network.

The compiler performs this construction ahead of time:

| | |
| --- | --- |
| **Input** | A feedforward network with scalar output, as an ONNX file or a `torch.nn.Sequential`: affine layers that alternate with ReLU, softplus, hyperbolic tangent, or sigmoid activations. |
| **Output** | One C++ header, `dual_cbf.h`, that includes only `<math.h>`. The weights are `static const float` arrays, all buffers have a size fixed at compile time, and the code has no recursion and no call to an allocator. |
| **Relative degree one** | `h(x)`, `L_f h(x)`, and `L_G h(x)` by dual numbers. |
| **Relative degree two** | `h(x)`, `L_f h(x)`, `L_f^2 h(x)`, and `L_G L_f h(x)` by hyper-dual numbers. |

## Installation

Python 3.9 or later is required.

```bash
pip install "dual-cbf-compiler[onnx]"
```

The base package depends on NumPy alone. The extra `onnx` adds the reader
of ONNX files, and the extra `torch` adds PyTorch for the direct reading of a
`torch.nn.Sequential`:

```bash
pip install dual-cbf-compiler            # compiler core, NumPy only
pip install "dual-cbf-compiler[onnx]"    # + ONNX files
pip install "dual-cbf-compiler[torch]"   # + torch.nn.Sequential
```

From a clone of this repository, for development:

```bash
git clone https://github.com/mkamaldar/dual_cbf_compiler.git
cd dual_cbf_compiler
pip install -e ".[test]"
```

## Quick start

**1. Export the trained network to ONNX.** An ONNX file stores the layer
structure together with the weights, so that the compiler needs no
description of the architecture.

```python
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(4, 32), nn.ReLU(),
    nn.Linear(32, 32), nn.ReLU(),
    nn.Linear(32, 1),
)
# ... train the model, or load the trained weights ...

torch.onnx.export(
    model, torch.zeros(1, 4), "cbf.onnx",
    input_names=["x"], output_names=["h"],
    opset_version=14, dynamo=False,
)
```

The argument `dynamo=False` selects the exporter that writes one `Gemm` node
per layer, which is the form that the compiler reads. PyTorch releases that
do not have this argument use that exporter by default; omit the argument
there.

**2. Compile.**

```bash
dual-cbf-compile cbf.onnx -o dual_cbf.h --inputs 2
```

```
Wrote dual_cbf.h: depth=3, input_dim=4, max_width=32, relative_degree=1
```

**3. Call the header from the control loop.**

```cpp
#include "dual_cbf.h"

// Kinematic bicycle: x = (p_x, p_y, psi, v), u = (acceleration, steering).
float x[4] = {1.0f, 2.0f, 0.3f, 5.0f};
float f[4] = {x[3] * cosf(x[2]), x[3] * sinf(x[2]), 0.0f, 0.0f};

// G(x) is passed flattened in ROW-MAJOR order: element (i, j) of the
// n-by-m matrix is G[i * m + j].
float G[4 * 2] = {
    /* row 0 */ 0.0f, 0.0f,
    /* row 1 */ 0.0f, 0.0f,
    /* row 2 */ 0.0f, x[3] / 2.5f,
    /* row 3 */ 1.0f, 0.0f,
};

float h, Lf, Lg[2];
dual_cbf::evaluate_cbf(x, f, G, /* m = */ 2, &h, &Lf, Lg);

// Constraint of the CBF quadratic program, for the user's alpha:
//   Lf + Lg[0] * u[0] + Lg[1] * u[1] + alpha(h) >= 0
```

The header does not contain the dynamics, the function `alpha`, or the
quadratic program; these belong to the calling code.

## Entry points of the emitted header

All entry points are in the namespace `dual_cbf` (option `--namespace`).
`N` denotes the largest layer width, input layer included, which the header
defines as `MAX_WIDTH`.

| Entry point | Emitted for | Returns | Forward passes |
| --- | --- | --- | --- |
| `evaluate_cbf(x, f, G, m, &h, &Lf, Lg)` | relative degree 1 | `h`, `L_f h`, `L_G h` | `m + 1` dual passes |
| `evaluate_cbf_shared(x, f, G, &h, &Lf, Lg)` | relative degree 1 with `--inputs m` | `h`, `L_f h`, `L_G h` | one sweep: the real part once, with `m + 1` dual parts |
| `evaluate_cbf_2nd_order(x, f, G, Df_f, Df_G, m, &L2f, LgLf)` | relative degree 2 | `L_f^2 h`, `L_G L_f h` | `m + 1` hyper-dual passes |
| `evaluate_hocbf(x, f, G, Df_f, Df_G, m, &h, &Lf, &L2f, LgLf)` | relative degree 2 | `h`, `L_f h`, `L_f^2 h`, `L_G L_f h` | the same `m + 1` hyper-dual passes |

Arguments:

| Argument | Length | Meaning |
| --- | --- | --- |
| `x` | `n` | state |
| `f` | `n` | drift `f(x)` |
| `G` | `n * m` | input matrix `G(x)`, row-major: element `(i, j)` at `G[i * m + j]` |
| `Df_f` | `n` | Jacobian-vector product `f'(x) f(x)` |
| `Df_G` | `n * m` | `f'(x) G(x)`, row-major |
| `Lg`, `LgLf` | `m` | outputs |

**Sequential passes and the shared sweep.** `evaluate_cbf` recomputes the
real part of the network in each of its `m + 1` passes.
`evaluate_cbf_shared` propagates the real part once, together with the
`m + 1` dual parts, and thus executes fewer operations. Its working memory
grows with `m`, whereas that of `evaluate_cbf` does not. The number of inputs
of the shared sweep is fixed at compile time and is available as
`dual_cbf::N_INPUTS`.

## Relative degree two

If `L_G h` vanishes identically, as for a position constraint on a mechanical
system whose input is a force or a torque, the input appears in the second
derivative of `h` along the trajectories. The constraint of the high-order
CBF then contains `L_f^2 h(x)` and `L_G L_f h(x)`, which involve the Hessian
of the network. Hyper-dual numbers `a + b eps_1 + c eps_2 + d eps_1 eps_2`,
with `eps_1^2 = eps_2^2 = 0`, carry this second-order information in one
forward pass.

```bash
dual-cbf-compile cbf.onnx -o dual_cbf.h --relative-degree 2
```

```cpp
float h, Lf, L2f, LgLf[1];
dual_cbf::evaluate_hocbf(x, f, G, Df_f, Df_G, /* m = */ 1,
                         &h, &Lf, &L2f, LgLf);
```

The calling code supplies the two Jacobian-vector products `Df_f = f'(x) f(x)`
and `Df_G = f'(x) G(x)`. For dynamics in closed form, these products are
short closed-form expressions.

The compiler rejects ReLU networks for relative degree two with a
`ValueError`. The second derivative of the ReLU is zero wherever it exists,
so that the Hessian of a ReLU network is zero at all points of
differentiability, and the second-order Lie derivatives would lose the
contribution of the Hessian. Networks for relative degree two use the
softplus, hyperbolic tangent, or sigmoid activation.

## Command-line reference

```
dual-cbf-compile MODEL [-o OUTPUT] [-r {1,2}] [-m INPUTS] [-n NAMESPACE]
                       [--torch-arch SPEC]
```

| Option | Default | Meaning |
| --- | --- | --- |
| `MODEL` | | `.onnx` file, or a PyTorch `state_dict` file (`.pt`, `.pth`) together with `--torch-arch` |
| `-o`, `--output` | `dual_cbf.h` | path of the emitted header |
| `-r`, `--relative-degree` | `1` | `1`: dual numbers; `2`: hyper-dual numbers |
| `-m`, `--inputs` | none | number of inputs `m`; with relative degree one, the header then also contains `evaluate_cbf_shared` |
| `-n`, `--namespace` | `dual_cbf` | C++ namespace of the emitted code |
| `--torch-arch` | none | layer structure of a `state_dict` file, for example `4,relu,32,relu,32,linear,1` |

A `state_dict` file stores the weights without the layer structure, and the
structure given with `--torch-arch` is not checked against the training
script. The ONNX path is therefore the recommended one.

## Python interface

```python
from dual_cbf_compiler import parse_onnx, parse_pytorch, emit_cpp_header

network = parse_onnx("cbf.onnx", relative_degree=1)   # or parse_pytorch(model)
header = emit_cpp_header(network, relative_degree=1, n_inputs=2)

with open("dual_cbf.h", "w") as file:
    file.write(header)
```

| Function | Purpose |
| --- | --- |
| `parse_onnx(file_path, relative_degree=1)` | reads an ONNX file into a `ParsedNetwork` |
| `parse_pytorch(model, relative_degree=1)` | reads a `torch.nn.Sequential` into a `ParsedNetwork` |
| `emit_cpp_header(network, relative_degree=1, namespace="dual_cbf", n_inputs=None)` | returns the text of the header |

`ParsedNetwork` exposes `depth`, `widths`, `input_dim`, and `validate()`.
The script [`example_pipeline.py`](example_pipeline.py) builds a
4-32-32-1 ReLU network, emits its header, compiles it with `g++`, and prints
the constraint.

## Memory of the emitted code

The weights and biases are constants, which a microcontroller toolchain
places in flash memory. The working memory has a size fixed at compile time.
The emitted code holds the state of the preceding layer in one static array
and the state of the current layer in an automatic array on the stack, which
exists only while the code of that layer runs. With `N` the largest layer
width and `m` the number of inputs:

| Entry point | Static array (floats) | Automatic array of a layer (floats) |
| --- | --- | --- |
| `evaluate_cbf` | `2 N` | at most `2 N` |
| `evaluate_cbf_shared` | `(m + 2) N` | at most `(m + 2) N` |
| `evaluate_cbf_2nd_order`, `evaluate_hocbf` | `4 N` | at most `4 N` |

The entry points with sequential passes also hold the current column of `G`,
and of `Df_G` for relative degree two, in automatic arrays of `n` floats.
These sizes do not depend on the depth of the network. Measured static,
stack, and heap usage on an ESP32-S3, together with execution times and a
comparison with statically allocated reverse-mode code, are reported in the
paper and reproduced by [`dual_cbf_esp32_experiments`][experiments].

## Tests

The tests emit headers for random networks, compile them with `g++`, run the
binaries, and compare the outputs with references. They need a C++17
compiler on the path.

```bash
pip install -e ".[test]"
pytest dual_cbf_compiler/verify.py tests   # reference: torch.autograd
python numpy_verify.py                     # first order; reference: NumPy chain rule
python numpy_verify_2nd.py                 # second order; reference: NumPy chain rule
```

The suite asserts agreement between the emitted single-precision code and
`torch.autograd` to a relative and absolute tolerance of `5e-5` for the
first-order quantities and of `1e-3` for the second-order quantities.

## Assumptions and limitations

- **Architecture.** Fully connected feedforward networks with scalar output:
  affine layers that alternate with elementwise activations, and an affine
  last layer. Convolutional, recurrent, normalization, dropout, and attention
  layers are rejected with a `ValueError`.
- **Activations.** ReLU, softplus with `beta = 1`, hyperbolic tangent, and
  sigmoid. Softplus is evaluated as `x` for `x > 20`, which is the default
  threshold of PyTorch.
- **ReLU.** At a preactivation equal to zero, the emitted code uses the
  derivative value zero. At such states the network need not be
  differentiable, and the returned value is the derivative of the affine
  piece that this convention selects. The paper states the set of states at
  which the returned values are the Lie derivatives.
- **Arithmetic.** The emitted code uses single precision (`float`, `expf`,
  `logf`, `tanhf`).
- **One translation unit, one caller.** The weights and the workspace have
  internal linkage; include the header in one source file. The entry points
  share a static workspace and are not reentrant; call them from one task,
  and not from an interrupt that can preempt a call in progress.
- **Dynamics.** The header evaluates the quantities for the `f`, `G`,
  `Df_f`, and `Df_G` that the caller passes. Model error, the row-major
  layout of `G`, and units are the responsibility of the caller.
- **No certificate.** The compiler differentiates the network that it is
  given. It does not verify that the network satisfies the condition of a
  CBF, and it does not check the feasibility of the quadratic program.

## Status and intended use

This is research software that accompanies a paper. It is not certified to
ISO 26262, DO-178C, IEC 61508, or any other functional-safety standard, and
it is provided without warranty (see [LICENSE](LICENSE)). Anyone who deploys
the emitted headers on a system that can injure people or damage property is
responsible for the verification, validation, and certification of that
system.

Each emitted header records the compiler version in its first line. Record
this version together with the results obtained from a header.

## Repository layout

```
dual_cbf_compiler/
├── dual_cbf_compiler/
│   ├── __init__.py          public interface and version
│   ├── cli.py               command dual-cbf-compile
│   ├── parser.py            readers for ONNX and torch.nn.Sequential
│   ├── ir.py                intermediate representation (ParsedNetwork)
│   ├── emitter.py           C++ code generation
│   └── verify.py            tests against torch.autograd
├── tests/test_validation.py tests of the input validation
├── numpy_verify.py          first-order check without PyTorch
├── numpy_verify_2nd.py      second-order check without PyTorch
├── example_pipeline.py      end-to-end example
├── CHANGELOG.md
├── setup.py, pyproject.toml, MANIFEST.in
└── LICENSE, NOTICE
```

## Citation

```bibtex
@misc{kamaldar2026dual,
  author = {Kamaldar, Mohammadreza},
  title  = {Extracting exact {L}ie derivatives without backpropagation:
            {A} dual compiler for neural control barrier functions},
  year   = {2026},
  url    = {https://arxiv.org/pdf/2604.23995}
}
```

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) and
[NOTICE](NOTICE).
