"""Offline FP64 numerical decomposition, excluded from detection timing."""

import torch

from algorithms.classic.reference import checks
from backends.nvidia.runtime import multiply, pack_operands

from .quantization import hadamard_pair
from .types import PROFILES, ExecutionContext


@torch.no_grad()
def diagnose(a, b, *, profile, phase="forward", context=None):
    context = context or ExecutionContext()
    original_product = a.double() @ b.double()
    a, b = a.float(), b.float()
    if profile == "nvfp4" and phase == "wgrad":
        a, b = hadamard_pair(a, b, context.quant_seed(phase) ^ 0x48A2)
    pa, pb = pack_operands(a, b, profile, phase, context)
    m, k = a.shape
    n, code = b.shape[1], PROFILES[profile].code
    ops = torch.ops.sdc
    aq = ops.decode(*pa, m, k, code).double()
    bq = ops.decode(*pb, n, k, code).T.double()
    ideal = aq @ bq
    actual = multiply(pa, pb, m, n, k, profile)
    predicted = ops.reference(*pa, *pb, m, n, k, code)
    _, residual, _ = ops.verify(actual, predicted, 0., 0., False)
    _, observed, _ = ops.verify(actual, torch.zeros_like(predicted), 0., 0., False)
    ideal_checks, exact_observed = checks(ideal), checks(actual)
    components = {
        "main_compute": exact_observed - ideal_checks,
        "output_reduction": observed.double() - exact_observed,
        "reference_path": predicted.double() - ideal_checks,
        "comparison_rounding": residual.double() - (observed.double()-predicted.double()),
    }
    reconstructed = components["main_compute"] + components["output_reduction"] - components["reference_path"] + components["comparison_rounding"]
    def norm(x):
        return float(x.double().norm())
    return {"components_l2": {key: norm(value) for key, value in components.items()},
            "residual_l2": norm(residual), "decomposition_error_l2": norm(reconstructed-residual.double()),
            "quantization_product_error_l2": norm(ideal-original_product),
            "gemm_rounding_error_l2": norm(actual.double()-ideal),
            "reference": "FP64 product of decoded Q+S; offline diagnostics only"}
