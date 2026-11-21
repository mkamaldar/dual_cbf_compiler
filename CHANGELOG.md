# Changelog

## 0.2.0

- Shared sweep for relative degree one. With the number of inputs given at
  compile time (`--inputs m`, or `n_inputs=m` in `emit_cpp_header`), the
  header also contains `evaluate_cbf_shared`, which propagates the real part
  once together with the `m + 1` dual parts.
- `evaluate_hocbf` for relative degree two, which returns `h`, `L_f h`,
  `L_f^2 h`, and `L_G L_f h` from the same `m + 1` hyper-dual passes.
- The emitted header records the compiler version in its first line.
- `parse_pytorch` rejects `Softplus` with `beta != 1`, which the emitted code
  does not implement.
- The package is laid out as an installable Python package
  (`dual_cbf_compiler/`), with the tests in `dual_cbf_compiler/verify.py` and
  `tests/`.
- The measurements of the paper were obtained with headers emitted by this
  version.

## 0.1.0

- First release: dual-number code for relative degree one
  (`evaluate_cbf`) and hyper-dual code for relative degree two
  (`evaluate_cbf_2nd_order`), from ONNX files and `torch.nn.Sequential`
  models.
