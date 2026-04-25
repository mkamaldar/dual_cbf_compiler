"""Command-line interface: dual-cbf-compile <model> -o dual_cbf.h.

ONNX is the recommended input format: it captures the network topology
along with the weights, so no manual architecture string is required.
PyTorch state_dicts (.pt/.pth) carry only tensors, so the user must
supply --torch-arch to declare the topology; this is error-prone and is
provided only as a fallback. Convert your trained PyTorch model to ONNX
with::

    torch.onnx.export(model, sample_input, 'model.onnx',
                      input_names=['x'], output_names=['h'])

and then invoke::

    dual-cbf-compile model.onnx -o dual_cbf.h
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .emitter import emit_cpp_header
from .parser import parse_onnx, parse_pytorch


def main(argv: list[str] | None = None) -> int:
    """Entry point invoked by the ``dual-cbf-compile`` console script."""
    p = argparse.ArgumentParser(
        prog="dual-cbf-compile",
        description=(
            "Compile a trained neural CBF (ONNX recommended; PyTorch "
            "state_dict supported as a fallback) into a self-contained "
            "bare-metal C++ header."
        ),
        epilog=(
            "TIP: Prefer ONNX. Export your trained PyTorch model with "
            "torch.onnx.export(model, sample_input, 'model.onnx'), then "
            "run 'dual-cbf-compile model.onnx -o dual_cbf.h'. ONNX captures "
            "the topology, eliminating the error-prone --torch-arch flag."
        ),
    )
    p.add_argument(
        "model",
        type=Path,
        help="Path to a .onnx model file (recommended) or a .pt/.pth state-dict.",
    )
    p.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("dual_cbf.h"),
        help="Output header path (default: ./dual_cbf.h).",
    )
    p.add_argument(
        "-r", "--relative-degree",
        type=int,
        choices=(1, 2),
        default=1,
        help=(
            "1: dual algebra; 2: hyper-dual (acceleration-controlled "
            "systems). NOTE: relative_degree=2 requires a smooth activation "
            "(Softplus, Tanh, or Sigmoid); ReLU networks are rejected."
        ),
    )
    p.add_argument(
        "-n", "--namespace",
        type=str,
        default="dual_cbf",
        help="C++ namespace for the emitted code.",
    )
    p.add_argument(
        "--torch-arch",
        type=str,
        default=None,
        help=(
            "FALLBACK ONLY (PyTorch .pt/.pth files). "
            "Comma-separated topology spec, e.g. '4,relu,32,relu,32,linear,1' "
            "for a 4-32-32-1 ReLU network with a final linear output. "
            "Strongly prefer ONNX export instead -- a typo here will silently "
            "map weights to the wrong dimensions."
        ),
    )

    args = p.parse_args(argv)

    suffix = args.model.suffix.lower()
    try:
        if suffix == ".onnx":
            network = parse_onnx(str(args.model), relative_degree=args.relative_degree)
        elif suffix in {".pt", ".pth"}:
            if args.torch_arch is None:
                print(
                    "error: PyTorch state_dict requires --torch-arch to declare topology.\n"
                    "       This is error-prone -- a typo silently maps weights to the\n"
                    "       wrong dimensions. RECOMMENDED: export your model to ONNX:\n"
                    "           torch.onnx.export(model, x_sample, 'model.onnx')\n"
                    "       then re-run with the .onnx file.",
                    file=sys.stderr,
                )
                return 2
            network = _load_pt_state_dict(
                args.model, args.torch_arch, relative_degree=args.relative_degree
            )
        else:
            print(
                f"error: unsupported model extension '{suffix}'. "
                f"Use .onnx (recommended) or .pt/.pth.",
                file=sys.stderr,
            )
            return 2
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    header = emit_cpp_header(
        network,
        relative_degree=args.relative_degree,
        namespace=args.namespace,
    )
    args.output.write_text(header)
    print(
        f"Wrote {args.output}: depth={network.depth}, "
        f"input_dim={network.input_dim}, max_width={max(network.widths)}, "
        f"relative_degree={args.relative_degree}"
    )
    return 0


def _load_pt_state_dict(path: Path, arch: str, relative_degree: int = 1):
    """Reconstruct a torch.nn.Sequential from a state_dict given a topology spec."""
    import torch
    import torch.nn as nn

    from .parser import parse_pytorch

    tokens = [t.strip() for t in arch.split(",")]
    modules: list[nn.Module] = []
    i = 0
    while i < len(tokens):
        try:
            n_in = int(tokens[i])
        except ValueError:
            raise SystemExit(
                f"--torch-arch: expected integer width at position {i}, got '{tokens[i]}'. "
                f"Hint: prefer ONNX export -- it captures the topology automatically."
            )
        if i + 1 >= len(tokens):
            raise SystemExit(f"--torch-arch: dangling width at position {i}")
        nxt = tokens[i + 1]
        if nxt.lower() in {"relu", "softplus", "tanh", "sigmoid", "linear"}:
            if i + 2 >= len(tokens):
                raise SystemExit(f"--torch-arch: missing output width after '{nxt}'")
            n_out = int(tokens[i + 2])
            modules.append(nn.Linear(n_in, n_out))
            if nxt.lower() == "relu":
                modules.append(nn.ReLU())
            elif nxt.lower() == "softplus":
                modules.append(nn.Softplus())
            elif nxt.lower() == "tanh":
                modules.append(nn.Tanh())
            elif nxt.lower() == "sigmoid":
                modules.append(nn.Sigmoid())
            i += 2
        else:
            raise SystemExit(
                f"--torch-arch: unknown activation '{nxt}' at position {i+1}. "
                f"Allowed: relu, softplus, tanh, sigmoid, linear. "
                f"Hint: prefer ONNX export to avoid this entirely."
            )

    model = nn.Sequential(*modules)
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    try:
        model.load_state_dict(state)
    except RuntimeError as exc:
        raise SystemExit(
            f"--torch-arch: declared topology does not match the state_dict.\n"
            f"  underlying error: {exc}\n"
            f"  Hint: a typo in --torch-arch silently maps weights to wrong "
            f"dimensions; prefer ONNX export."
        ) from exc
    return parse_pytorch(model, relative_degree=relative_degree)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
