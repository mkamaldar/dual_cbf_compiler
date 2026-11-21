"""Internal representation for a parsed feedforward network."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Union

import numpy as np

ActivationKind = Literal["relu", "softplus", "tanh", "sigmoid", "identity"]


@dataclass
class LinearLayer:
    """An affine layer y = W x + b."""

    W: np.ndarray  # shape (n_out, n_in), row-major float32
    b: np.ndarray  # shape (n_out,), float32

    @property
    def n_in(self) -> int:
        return int(self.W.shape[1])

    @property
    def n_out(self) -> int:
        return int(self.W.shape[0])


@dataclass
class ActivationLayer:
    """A componentwise activation layer."""

    kind: ActivationKind


Layer = Union[LinearLayer, ActivationLayer]


@dataclass
class ParsedNetwork:
    """Topology and parameters of a feedforward neural CBF.

    The layers list alternates LinearLayer and ActivationLayer entries,
    starting and ending with LinearLayer. The terminal layer is a single-
    output linear layer (n_d = 1) per the CBF parameterization.
    """

    layers: List[Layer] = field(default_factory=list)

    @property
    def linear_layers(self) -> List[LinearLayer]:
        return [layer for layer in self.layers if isinstance(layer, LinearLayer)]

    @property
    def activation_layers(self) -> List[ActivationLayer]:
        return [layer for layer in self.layers if isinstance(layer, ActivationLayer)]

    @property
    def widths(self) -> List[int]:
        """Layer widths (n_0, n_1, ..., n_d)."""
        if not self.layers:
            return []
        linears = self.linear_layers
        return [linears[0].n_in] + [layer.n_out for layer in linears]

    @property
    def depth(self) -> int:
        """Number of linear layers d."""
        return len(self.linear_layers)

    @property
    def input_dim(self) -> int:
        return self.widths[0]

    @property
    def output_dim(self) -> int:
        return self.widths[-1]

    def validate(self) -> None:
        """Validate structural assumptions of the CBF compiler."""
        if not self.layers:
            raise ValueError("Empty network")

        if not isinstance(self.layers[0], LinearLayer):
            raise ValueError("Network must start with a linear layer")
        if not isinstance(self.layers[-1], LinearLayer):
            raise ValueError("Network must end with a linear layer")

        # Alternating pattern: Linear, Activation, Linear, Activation, ..., Linear
        for i, layer in enumerate(self.layers):
            expected_linear = i % 2 == 0
            is_linear = isinstance(layer, LinearLayer)
            if expected_linear != is_linear:
                raise ValueError(
                    f"Layer {i}: expected {'linear' if expected_linear else 'activation'}, "
                    f"got {'linear' if is_linear else 'activation'}"
                )

        # Dimension consistency
        linears = self.linear_layers
        for i in range(len(linears) - 1):
            if linears[i].n_out != linears[i + 1].n_in:
                raise ValueError(
                    f"Layer dim mismatch: layer {i} outputs {linears[i].n_out} "
                    f"but layer {i+1} expects {linears[i+1].n_in}"
                )

        # CBF requires scalar output
        if self.output_dim != 1:
            raise ValueError(
                f"Network output dimension must be 1 (CBF is scalar-valued), "
                f"got {self.output_dim}"
            )
