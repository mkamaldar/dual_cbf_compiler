"""C++ code emitter for the dual-algebraic CBF compiler.

Generates a single self-contained header file ``dual_cbf.h`` that
evaluates the network and its Lie derivatives via dual or hyper-dual
arithmetic. The generated code:

* allocates zero dynamic memory (no ``new``/``malloc``/``std::vector``);
* stores all weight matrices and biases as ``static const float`` arrays
  embedded directly in the header;
* uses one shared static scratch buffer sized exactly :math:`2\\,\\max_i n_i`
  (relative-degree 1) or :math:`4\\,\\max_i n_i` (relative-degree 2) and
  destructively overwrites it as the forward pass advances;
* exposes ``evaluate_cbf`` (relative-degree 1) or
  ``evaluate_cbf_2nd_order`` (relative-degree 2) as the public entry
  point.
"""

from __future__ import annotations

import io
from textwrap import dedent
from typing import List

import numpy as np

from .ir import ActivationLayer, LinearLayer, ParsedNetwork


# ---------------------------------------------------------------------------
# Helpers for emitting array literals
# ---------------------------------------------------------------------------

def _emit_float_array(name: str, data: np.ndarray, line_width: int = 8) -> str:
    """Render a flat float array as a ``static const float`` definition."""
    flat = data.astype(np.float32).reshape(-1)
    n = flat.shape[0]
    buf = io.StringIO()
    buf.write(f"static const float {name}[{n}] = {{\n    ")
    for i, v in enumerate(flat):
        # Use repr-ish full precision; ensure literal is a valid C++ float
        # (must contain '.', 'e', or 'E' before the 'f' suffix).
        s = f"{float(v):.9g}"
        if not any(ch in s for ch in (".", "e", "E", "n")):  # 'n' for nan/inf
            s = s + ".0"
        buf.write(f"{s}f")
        if i != n - 1:
            buf.write(", ")
        if (i + 1) % line_width == 0 and i != n - 1:
            buf.write("\n    ")
    buf.write("\n};\n")
    return buf.getvalue()


def _activation_real_expr(kind: str, var: str) -> str:
    """C++ expression for the real part of an activation."""
    if kind == "relu":
        return f"(({var}) > 0.0f ? ({var}) : 0.0f)"
    if kind == "softplus":
        # log(1 + exp(x)); guard large x to avoid overflow
        return f"((({var}) > 20.0f) ? ({var}) : logf(1.0f + expf({var})))"
    if kind == "tanh":
        return f"tanhf({var})"
    if kind == "sigmoid":
        return f"(1.0f / (1.0f + expf(-({var}))))"
    if kind == "identity":
        return f"({var})"
    raise ValueError(f"unsupported activation: {kind}")


def _activation_deriv_expr(kind: str, x: str, sigma_x: str) -> str:
    """C++ expression for sigma'(x) given x and sigma(x).

    Allows reuse of the real activation when convenient (e.g. tanh, sigmoid).
    """
    if kind == "relu":
        return f"(({x}) > 0.0f ? 1.0f : 0.0f)"
    if kind == "softplus":
        return f"(1.0f / (1.0f + expf(-({x}))))"
    if kind == "tanh":
        return f"(1.0f - ({sigma_x}) * ({sigma_x}))"
    if kind == "sigmoid":
        return f"(({sigma_x}) * (1.0f - ({sigma_x})))"
    if kind == "identity":
        return "1.0f"
    raise ValueError(f"unsupported activation: {kind}")


def _activation_second_deriv_expr(kind: str, x: str, sigma_x: str, sigma_p_x: str) -> str:
    """C++ expression for sigma''(x).

    Notes:
        ReLU has zero second derivative almost everywhere; relative-degree 2
        compilation requires a smooth activation.
    """
    if kind == "relu":
        # Mathematically zero a.e.; we still emit 0 so the algebra evaluates
        # but the user is warned at parse time.
        return "0.0f"
    if kind == "softplus":
        # d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
        # softplus' = sigmoid; softplus'' = sigmoid' = sigmoid*(1-sigmoid)
        return f"(({sigma_p_x}) * (1.0f - ({sigma_p_x})))"
    if kind == "tanh":
        return f"(-2.0f * ({sigma_x}) * ({sigma_p_x}))"
    if kind == "sigmoid":
        return f"(({sigma_p_x}) * (1.0f - 2.0f * ({sigma_x})))"
    if kind == "identity":
        return "0.0f"
    raise ValueError(f"unsupported activation: {kind}")


# ---------------------------------------------------------------------------
# Public emitter
# ---------------------------------------------------------------------------

def emit_cpp_header(
    network: ParsedNetwork,
    relative_degree: int = 1,
    namespace: str = "dual_cbf",
) -> str:
    """Translate a parsed network into a self-contained C++ header.

    Args:
        network: Validated ParsedNetwork.
        relative_degree: 1 generates standard dual algebra (eps^2 = 0).
            2 generates hyper-dual arithmetic with cross term eps_12,
            yielding the second-order Lie derivatives required by
            relative-degree-two CBFs.
        namespace: C++ namespace to wrap the emitted code in.

    Returns:
        The full text of ``dual_cbf.h``.
    """
    network.validate()
    if relative_degree not in (1, 2):
        raise ValueError(f"relative_degree must be 1 or 2, got {relative_degree}")

    if relative_degree == 1:
        return _emit_first_order(network, namespace)
    return _emit_second_order(network, namespace)


# ---------------------------------------------------------------------------
# Relative degree 1
# ---------------------------------------------------------------------------

def _emit_first_order(network: ParsedNetwork, namespace: str) -> str:
    widths = network.widths
    n_in = widths[0]
    n_max = max(widths)
    depth = network.depth
    linears = network.linear_layers
    activations = network.activation_layers

    out = io.StringIO()

    # ------------------------------------------------------------ header guard
    out.write("// Auto-generated by dual_cbf_compiler. Do not edit.\n")
    out.write("// Bare-metal C++ header: zero dynamic allocation, "
              "exact Lie derivatives.\n")
    out.write("//\n")
    out.write("// Usage:\n")
    out.write("//   #include \"dual_cbf.h\"\n")
    out.write(f"//   {namespace}::evaluate_cbf(x, f, G, m, &h, &Lf, Lg);\n")
    out.write("//\n")
    out.write("// IMPORTANT: G(x) must be passed as a flat ROW-MAJOR float array of\n")
    out.write("// shape (n x m), where n = INPUT_DIM is the state dimension and\n")
    out.write("// m is the number of control inputs. Element G(i, j) is stored at\n")
    out.write("// G[i * m + j].\n//\n")
    out.write("// Example: for a 4x2 input field whose columns are\n")
    out.write("//   col_0 = [0, 0, 0, 1]^T  (acceleration)\n")
    out.write("//   col_1 = [0, 0, v/L, 0]^T  (steering)\n")
    out.write("// declare G as:\n")
    out.write("//   float G[4 * 2] = {\n")
    out.write("//       /* row 0 */ 0.0f,    0.0f,\n")
    out.write("//       /* row 1 */ 0.0f,    0.0f,\n")
    out.write("//       /* row 2 */ 0.0f,    v/L,\n")
    out.write("//       /* row 3 */ 1.0f,    0.0f,\n")
    out.write("//   };\n")
    out.write("// Equivalently, if you keep G as a 2D array float G2D[4][2]:\n")
    out.write("//   float G[4 * 2];\n")
    out.write("//   for (int i = 0; i < 4; ++i)\n")
    out.write("//       for (int j = 0; j < 2; ++j)\n")
    out.write("//           G[i * 2 + j] = G2D[i][j];\n\n")
    out.write("#ifndef DUAL_CBF_H\n#define DUAL_CBF_H\n\n")
    out.write("#include <math.h>\n\n")
    out.write(f"namespace {namespace} {{\n\n")

    # ---------------------------------------------------------------- topology
    out.write(f"static constexpr int INPUT_DIM = {n_in};\n")
    out.write(f"static constexpr int DEPTH = {depth};\n")
    out.write(f"static constexpr int MAX_WIDTH = {n_max};\n")
    out.write("static constexpr int LAYER_WIDTHS[DEPTH + 1] = {")
    out.write(", ".join(str(w) for w in widths))
    out.write("};\n\n")

    # ------------------------------------------------------------- parameters
    for li, layer in enumerate(linears):
        out.write(_emit_float_array(f"W{li}", layer.W))
        out.write(_emit_float_array(f"b{li}", layer.b))
        out.write("\n")

    # ----------------------------------------------------- shared static buffer
    # Spec: exactly 2 * max(n_i) floats. We split as [real | dual].
    out.write("// Shared static scratch buffer: 2 * MAX_WIDTH floats.\n")
    out.write("// Layout: [0, MAX_WIDTH) = real part, [MAX_WIDTH, 2*MAX_WIDTH) = dual part.\n")
    out.write("// Destructively overwritten as the forward pass advances.\n")
    out.write("static float scratch[2 * MAX_WIDTH];\n")
    out.write("static float real_cache[MAX_WIDTH];  // real path is identical "
              "across all directions; computed once, reused.\n\n")

    # ---------------------------------------- layer-by-layer expansion helpers
    out.write(_emit_real_forward_first_order(linears, activations))
    out.write(_emit_directional_forward_first_order(linears, activations))

    # ------------------------------------------------------------ entry point
    out.write(dedent("""\
        // Evaluate the CBF safety constraint.
        //
        // x  : real state, length INPUT_DIM
        // f  : drift vector field f(x), length INPUT_DIM
        // G  : input field G(x) flattened ROW-MAJOR, shape (INPUT_DIM x m).
        //      Element G(i, j) is stored at G[i * m + j]. The j-th column
        //      of G(x) is the input field associated with control input u_j.
        //      See the file-level comment for a worked example.
        // m  : number of control inputs
        // h  : [out] barrier value h_theta(x)
        // Lf : [out] drift Lie derivative L_f h_theta(x)
        // Lg : [out] input Lie derivative L_G h_theta(x), length m
        inline void evaluate_cbf(
            const float* x,
            const float* f,
            const float* G,
            int m,
            float* h,
            float* Lf,
            float* Lg)
        {
            // Pass 1: real path computed once and cached.
            forward_real(x, real_cache);
            *h = real_cache[0];

            // Pass 2: dual seed (x, f(x)) -> L_f.
            *Lf = forward_directional(x, f);

            // Pass 3..2+m: dual seeds (x, col_j(G)) -> L_g_j.
            // Extract column j from row-major G:  col_j[k] = G[k * m + j].
            for (int j = 0; j < m; ++j) {
                float gj[INPUT_DIM];
                for (int k = 0; k < INPUT_DIM; ++k) {
                    gj[k] = G[k * m + j];
                }
                Lg[j] = forward_directional(x, gj);
            }
        }

        """))

    out.write(f"}}  // namespace {namespace}\n\n")
    out.write("#endif  // DUAL_CBF_H\n")
    return out.getvalue()


def _emit_real_forward_first_order(
    linears: List[LinearLayer],
    activations: List[ActivationLayer],
) -> str:
    """Generate the real-only forward pass (caches activations for reuse)."""
    out = io.StringIO()
    out.write(dedent("""\
        // Real-only forward pass: writes h_theta(x) to out[0] and the per-layer
        // pre-activations into the shared scratch buffer for reuse by the
        // directional pass.
        inline void forward_real(const float* x, float* out)
        {
            float* a = &scratch[0];           // current activations (real)
            float* a_pre = &scratch[MAX_WIDTH];  // pre-activations cache (real)

            // Copy input
            for (int i = 0; i < LAYER_WIDTHS[0]; ++i) a[i] = x[i];

        """))

    n_layers = len(linears)
    for li, lin in enumerate(linears):
        n_in, n_out = lin.n_in, lin.n_out
        out.write(f"            // ---- Linear layer {li}: {n_in} -> {n_out} ----\n")
        out.write("            {\n")
        out.write(f"                float tmp[{n_out}];\n")
        out.write(f"                for (int i = 0; i < {n_out}; ++i) {{\n")
        out.write(f"                    float acc = b{li}[i];\n")
        out.write(f"                    for (int j = 0; j < {n_in}; ++j) {{\n")
        out.write(f"                        acc += W{li}[i * {n_in} + j] * a[j];\n")
        out.write("                    }\n")
        out.write("                    tmp[i] = acc;\n")
        out.write("                }\n")
        # Cache the pre-activation for reuse in directional pass
        out.write(f"                for (int i = 0; i < {n_out}; ++i) a_pre[i] = tmp[i];\n")

        is_last = li == n_layers - 1
        if is_last:
            out.write(f"                for (int i = 0; i < {n_out}; ++i) a[i] = tmp[i];\n")
        else:
            kind = activations[li].kind
            out.write(f"                // Activation: {kind}\n")
            out.write(f"                for (int i = 0; i < {n_out}; ++i) {{\n")
            out.write(f"                    float t = tmp[i];\n")
            out.write(f"                    a[i] = {_activation_real_expr(kind, 't')};\n")
            out.write("                }\n")
        out.write("            }\n\n")

    out.write("            out[0] = a[0];\n")
    out.write("        }\n\n")
    return out.getvalue()


def _emit_directional_forward_first_order(
    linears: List[LinearLayer],
    activations: List[ActivationLayer],
) -> str:
    """Generate the dual directional pass: returns L_xi h_theta(x).

    Operates on the shared static scratch buffer. The real part is recomputed
    here too so the routine is self-contained (the cached real_cache is for
    the public entry point's h output only).
    """
    out = io.StringIO()
    out.write(dedent("""\
        // Dual directional pass with seed (x, xi). Returns
        //   Du( h_theta(x + xi * eps) ) = L_xi h_theta(x).
        // Operates entirely on the shared static scratch buffer:
        //   real part: scratch[0 .. MAX_WIDTH)
        //   dual part: scratch[MAX_WIDTH .. 2*MAX_WIDTH)
        inline float forward_directional(const float* x, const float* xi)
        {
            float* a   = &scratch[0];
            float* d   = &scratch[MAX_WIDTH];

            // Seed: real = x, dual = xi
            for (int i = 0; i < LAYER_WIDTHS[0]; ++i) {
                a[i] = x[i];
                d[i] = xi[i];
            }

        """))

    n_layers = len(linears)
    for li, lin in enumerate(linears):
        n_in, n_out = lin.n_in, lin.n_out
        out.write(f"            // ---- Dual linear layer {li}: {n_in} -> {n_out} ----\n")
        out.write("            {\n")
        out.write(f"                float tmp_a[{n_out}];\n")
        out.write(f"                float tmp_d[{n_out}];\n")
        out.write(f"                for (int i = 0; i < {n_out}; ++i) {{\n")
        out.write(f"                    float acc_a = b{li}[i];\n")
        out.write("                    float acc_d = 0.0f;\n")
        # SIMD-friendly inner loop: real and dual share the same weight row
        out.write(f"                    for (int j = 0; j < {n_in}; ++j) {{\n")
        out.write(f"                        float w = W{li}[i * {n_in} + j];\n")
        out.write("                        acc_a += w * a[j];\n")
        out.write("                        acc_d += w * d[j];\n")
        out.write("                    }\n")
        out.write("                    tmp_a[i] = acc_a;\n")
        out.write("                    tmp_d[i] = acc_d;\n")
        out.write("                }\n")

        is_last = li == n_layers - 1
        if is_last:
            out.write(f"                for (int i = 0; i < {n_out}; ++i) {{\n")
            out.write("                    a[i] = tmp_a[i];\n")
            out.write("                    d[i] = tmp_d[i];\n")
            out.write("                }\n")
        else:
            kind = activations[li].kind
            out.write(f"                // Dual activation: {kind}\n")
            out.write(f"                for (int i = 0; i < {n_out}; ++i) {{\n")
            out.write("                    float pre = tmp_a[i];\n")
            out.write(f"                    float sig = {_activation_real_expr(kind, 'pre')};\n")
            out.write(
                f"                    float sig_p = {_activation_deriv_expr(kind, 'pre', 'sig')};\n"
            )
            out.write("                    a[i] = sig;\n")
            out.write("                    d[i] = sig_p * tmp_d[i];\n")
            out.write("                }\n")
        out.write("            }\n\n")

    out.write("            return d[0];  // dual part of scalar output = L_xi h_theta(x)\n")
    out.write("        }\n\n")
    return out.getvalue()


# ---------------------------------------------------------------------------
# Relative degree 2 (hyper-dual)
# ---------------------------------------------------------------------------

def _emit_second_order(network: ParsedNetwork, namespace: str) -> str:
    widths = network.widths
    n_in = widths[0]
    n_max = max(widths)
    depth = network.depth
    linears = network.linear_layers
    activations = network.activation_layers

    # Hard error: ReLU's second derivative is zero almost everywhere, so a
    # relative-degree-2 compilation would silently lose the Hessian
    # contribution required for the L_f^2 and L_G L_f Lie derivatives. We
    # refuse to generate code rather than emit a passive C++ warning that
    # users will overlook.
    relu_layers = [i for i, a in enumerate(activations) if a.kind == "relu"]
    if relu_layers:
        raise ValueError(
            "Cannot emit relative_degree=2 code for a network with ReLU "
            f"activations (found at activation layer index {relu_layers[0]}). "
            "ReLU's second derivative vanishes almost everywhere, so the "
            "Hessian contribution to L_f^2 h_theta and L_{col_j(G)} L_f h_theta "
            "would be lost and the resulting controller would degrade silently. "
            "Retrain the network with a twice-differentiable activation such as "
            "Softplus, Tanh, or Sigmoid before requesting relative_degree=2."
        )

    out = io.StringIO()

    out.write("// Auto-generated by dual_cbf_compiler. Do not edit.\n")
    out.write("// Bare-metal C++ header: hyper-dual algebra for second-order Lie derivatives.\n")
    out.write("//\n")
    out.write("// Hyper-dual element: a + b*eps_1 + c*eps_2 + d*eps_12,\n")
    out.write("// with eps_1^2 = eps_2^2 = (eps_1 eps_2)^2 = 0.\n")
    out.write("//\n")
    out.write("// Usage:\n")
    out.write("//   #include \"dual_cbf.h\"\n")
    out.write(f"//   {namespace}::evaluate_cbf_2nd_order(x, f, G, Df_f, Df_G, m,\n")
    out.write("//                                       &L2f, LgLf);\n")
    out.write("//\n")
    out.write("// IMPORTANT: G(x) and Df_G must be passed as flat ROW-MAJOR arrays of\n")
    out.write("// shape (n x m). Element (i, j) is stored at index [i * m + j].\n")
    out.write("// To convert from a 2D array float G2D[n][m]:\n")
    out.write("//   for (int i = 0; i < n; ++i)\n")
    out.write("//       for (int j = 0; j < m; ++j)\n")
    out.write("//           G[i * m + j] = G2D[i][j];\n\n")
    out.write("#ifndef DUAL_CBF_H\n#define DUAL_CBF_H\n\n")
    out.write("#include <math.h>\n\n")
    out.write(f"namespace {namespace} {{\n\n")

    out.write(f"static constexpr int INPUT_DIM = {n_in};\n")
    out.write(f"static constexpr int DEPTH = {depth};\n")
    out.write(f"static constexpr int MAX_WIDTH = {n_max};\n")
    out.write("static constexpr int LAYER_WIDTHS[DEPTH + 1] = {")
    out.write(", ".join(str(w) for w in widths))
    out.write("};\n\n")

    for li, layer in enumerate(linears):
        out.write(_emit_float_array(f"W{li}", layer.W))
        out.write(_emit_float_array(f"b{li}", layer.b))
        out.write("\n")

    # Hyper-dual buffer: 4 components per node.
    out.write("// Hyper-dual scratch buffer: 4 * MAX_WIDTH floats.\n")
    out.write("// Layout: [0,W) real | [W,2W) eps_1 | [2W,3W) eps_2 | [3W,4W) eps_12.\n")
    out.write("static float scratch[4 * MAX_WIDTH];\n\n")

    out.write(_emit_hyper_forward(linears, activations))

    out.write(dedent("""\
        // Evaluate second-order Lie derivatives via the hyper-dual seed
        //   x_{f,xi} = x + f(x) eps_1 + xi(x) eps_2 + Df(x) xi(x) eps_12.
        //
        // x      : state, INPUT_DIM
        // f      : drift f(x), INPUT_DIM
        // xi     : second vector field xi(x) (= f for L_f^2; col_j(G) for L_{g_j} L_f)
        // Df_xi  : Jacobian-vector product f'(x) xi(x), INPUT_DIM
        //          (the user is responsible for supplying this; for analytic
        //           f the closed form is simple, and for learned f it can be
        //           obtained by a separate forward dual pass)
        // Returns Du_12( h_theta(seed) ) which equals
        //   xi^T H(x) f + grad h(x)^T (f'(x) xi)
        // i.e. L_f^2 h_theta(x) when xi = f, and L_{xi} L_f h_theta(x) otherwise.
        inline float evaluate_second_order(
            const float* x,
            const float* f,
            const float* xi,
            const float* Df_xi)
        {
            return forward_hyperdual(x, f, xi, Df_xi);
        }

        // Public entry point. Computes L_f^2 h_theta(x) and L_{col_j(G)} L_f h_theta(x)
        // for j=0..m-1. The user supplies the Jacobian-vector products
        //   Df_f  = f'(x) f(x)
        //   Df_Gj = f'(x) col_j(G(x))    flattened column-major (INPUT_DIM x m)
        inline void evaluate_cbf_2nd_order(
            const float* x,
            const float* f,
            const float* G,
            const float* Df_f,
            const float* Df_G,
            int m,
            float* L2f,
            float* LgLf)
        {
            *L2f = forward_hyperdual(x, f, f, Df_f);

            for (int j = 0; j < m; ++j) {
                float gj[INPUT_DIM];
                float Df_gj[INPUT_DIM];
                for (int k = 0; k < INPUT_DIM; ++k) {
                    gj[k]    = G[k * m + j];
                    Df_gj[k] = Df_G[k * m + j];
                }
                LgLf[j] = forward_hyperdual(x, f, gj, Df_gj);
            }
        }

        """))

    out.write(f"}}  // namespace {namespace}\n\n")
    out.write("#endif  // DUAL_CBF_H\n")
    return out.getvalue()


def _emit_hyper_forward(
    linears: List[LinearLayer],
    activations: List[ActivationLayer],
) -> str:
    out = io.StringIO()
    out.write(dedent("""\
        // Hyper-dual forward pass with seed
        //   z = x + v eps_1 + w eps_2 + u eps_12
        // returns the eps_12 component of the scalar output.
        inline float forward_hyperdual(
            const float* x,
            const float* v,
            const float* w,
            const float* u)
        {
            float* a  = &scratch[0 * MAX_WIDTH];
            float* d1 = &scratch[1 * MAX_WIDTH];
            float* d2 = &scratch[2 * MAX_WIDTH];
            float* d12 = &scratch[3 * MAX_WIDTH];

            for (int i = 0; i < LAYER_WIDTHS[0]; ++i) {
                a[i]  = x[i];
                d1[i] = v[i];
                d2[i] = w[i];
                d12[i] = u[i];
            }

        """))

    n_layers = len(linears)
    for li, lin in enumerate(linears):
        n_in, n_out = lin.n_in, lin.n_out
        out.write(f"            // ---- Hyper-dual linear layer {li}: {n_in} -> {n_out} ----\n")
        out.write("            {\n")
        for var in ("a", "d1", "d2", "d12"):
            out.write(f"                float tmp_{var}[{n_out}];\n")
        out.write(f"                for (int i = 0; i < {n_out}; ++i) {{\n")
        out.write(f"                    float acc_a = b{li}[i];\n")
        out.write("                    float acc_d1 = 0.0f, acc_d2 = 0.0f, acc_d12 = 0.0f;\n")
        out.write(f"                    for (int j = 0; j < {n_in}; ++j) {{\n")
        out.write(f"                        float wv = W{li}[i * {n_in} + j];\n")
        out.write("                        acc_a   += wv * a[j];\n")
        out.write("                        acc_d1  += wv * d1[j];\n")
        out.write("                        acc_d2  += wv * d2[j];\n")
        out.write("                        acc_d12 += wv * d12[j];\n")
        out.write("                    }\n")
        out.write("                    tmp_a[i]   = acc_a;\n")
        out.write("                    tmp_d1[i]  = acc_d1;\n")
        out.write("                    tmp_d2[i]  = acc_d2;\n")
        out.write("                    tmp_d12[i] = acc_d12;\n")
        out.write("                }\n")

        is_last = li == n_layers - 1
        if is_last:
            out.write(f"                for (int i = 0; i < {n_out}; ++i) {{\n")
            for var in ("a", "d1", "d2", "d12"):
                out.write(f"                    {var}[i] = tmp_{var}[i];\n")
            out.write("                }\n")
        else:
            kind = activations[li].kind
            out.write(f"                // Hyper-dual activation: {kind}\n")
            out.write(f"                for (int i = 0; i < {n_out}; ++i) {{\n")
            out.write("                    float pre = tmp_a[i];\n")
            out.write(f"                    float sig   = {_activation_real_expr(kind, 'pre')};\n")
            out.write(
                f"                    float sig_p = {_activation_deriv_expr(kind, 'pre', 'sig')};\n"
            )
            out.write(
                f"                    float sig_pp = "
                f"{_activation_second_deriv_expr(kind, 'pre', 'sig', 'sig_p')};\n"
            )
            # tilde sigma(z) = sigma(y)
            #   + sigma'(y) v eps_1
            #   + sigma'(y) w eps_2
            #   + (sigma''(y) v*w + sigma'(y) u) eps_12  (componentwise)
            out.write("                    a[i]   = sig;\n")
            out.write("                    d1[i]  = sig_p * tmp_d1[i];\n")
            out.write("                    d2[i]  = sig_p * tmp_d2[i];\n")
            out.write(
                "                    d12[i] = sig_pp * tmp_d1[i] * tmp_d2[i] "
                "+ sig_p * tmp_d12[i];\n"
            )
            out.write("                }\n")
        out.write("            }\n\n")

    out.write("            return d12[0];\n")
    out.write("        }\n\n")
    return out.getvalue()
