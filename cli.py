"""Command-line interface: dual-cbf-compile <model> -o dual_cbf.h."""

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
            "Compile a trained neural CBF (PyTorch state_dict or ONNX) "
            "into a self-contained bare-metal C++ header."
        ),
    )
    p.add_argument(
        "model",
        type=Path,
        help="Path to a .pt/.pth state-dict or .onnx model file.",
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
        help="1: dual algebra; 2: hyper-dual (acceleration-controlled systems).",
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
            "Comma-separated layer widths and activations for loading a "
            "PyTorch state_dict, e.g. '4,relu,32,relu,32,linear,1'. "
            "Required for .pt/.pth files."
        ),
    )

    args = p.parse_args(argv)

    suffix = args.model.suffix.lower()
    if suffix == ".onnx":
        network = parse_onnx(str(args.model))
    elif suffix in {".pt", ".pth"}:
        if args.torch_arch is None:
            print(
                "error: PyTorch state_dict requires --torch-arch to declare topology",
                file=sys.stderr,
            )
            return 2
        network = _load_pt_state_dict(args.model, args.torch_arch)
    else:
        print(f"error: unsupported model extension '{suffix}'", file=sys.stderr)
        return 2

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


def _load_pt_state_dict(path: Path, arch: str):
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
            raise SystemExit(f"--torch-arch: expected integer at position {i}, got '{tokens[i]}'")
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
            raise SystemExit(f"--torch-arch: unknown activation '{nxt}'")

    model = nn.Sequential(*modules)
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    return parse_pytorch(model)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
