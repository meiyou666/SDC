"""Independent FP64 formulas for tests and numerical diagnostics."""

import torch

from algorithms import Algorithm


def checks(c: torch.Tensor) -> torch.Tensor:
    c = c.double()
    m, n = c.shape
    row_weights = torch.arange(1, m + 1, device=c.device, dtype=c.dtype)
    col_weights = torch.arange(1, n + 1, device=c.device, dtype=c.dtype)
    return torch.cat((c.sum(0), row_weights @ c, c.sum(1), c @ col_weights))


def reference(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a, b = a.double(), b.double()
    m, n = a.shape[0], b.shape[1]
    u = torch.arange(1, m + 1, device=a.device, dtype=a.dtype)
    v = torch.arange(1, n + 1, device=a.device, dtype=a.dtype)
    return torch.cat((a.sum(0) @ b, (u @ a) @ b, a @ b.sum(1), a @ (b @ v)))


def verify(c: torch.Tensor, expected: torch.Tensor, method: Algorithm):
    observed = checks(c)
    residual = observed - expected
    threshold = method.atol + method.rtol * torch.maximum(observed.abs(), expected.abs())
    bad = ~torch.isfinite(residual) | (residual.abs() > threshold)
    return residual, threshold, bad


def locate_correct(c: torch.Tensor, expected: torch.Tensor, method: Algorithm):
    residual, threshold, bad = verify(c, expected, method)
    m, n = c.shape
    status = torch.zeros(6, dtype=torch.int64, device=c.device)
    status[2:4] = -1
    status[0] = bad.any()
    status[5] = status[0]
    cols = (bad[:n] | bad[n:2*n]).nonzero().flatten()
    rows = (bad[2*n:2*n+m] | bad[2*n+m:]).nonzero().flatten()
    if len(cols) != 1 or len(rows) != 1 or not torch.isfinite(residual).all():
        return c, status, residual, threshold
    i, j = int(rows[0]), int(cols[0])
    er, ec = residual[2*n+i], residual[j]
    tol = threshold[2*n+i] + threshold[j]
    if abs(er) <= threshold[2*n+i] or abs(ec) <= threshold[j]:
        return c, status, residual, threshold
    if abs(er-ec) > tol:
        return c, status, residual, threshold
    if abs(residual[n+j] - (i+1)*ec) > threshold[n+j] + (i+1)*threshold[j]:
        return c, status, residual, threshold
    if abs(residual[2*n+m+i] - (j+1)*er) > threshold[2*n+m+i] + (j+1)*threshold[2*n+i]:
        return c, status, residual, threshold
    corrected = c.clone()
    corrected[i, j] -= ((er+ec)/2).to(c.dtype)
    status[1:4] = torch.tensor([1, i, j], device=c.device)
    status[4] = 1
    status[5] = verify(corrected, expected, method)[2].any()
    return corrected, status, residual, threshold
