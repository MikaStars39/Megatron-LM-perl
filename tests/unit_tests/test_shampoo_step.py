# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for the Shampoo step() on a small linear regression.

Run: pytest tests/unit_tests/test_shampoo_step.py -v
"""

import pytest
import torch

from megatron.core.optimizer.shampoo import Shampoo, _inv_pth_root


def _linreg_loss(W, X, y):
    """W: [d_out, d_in], X: [N, d_in], y: [N, d_out]."""
    return ((X @ W.T - y) ** 2).mean()


def test_inv_pth_root_identity():
    """M = I -> M^(-1/p) = I."""
    I = torch.eye(5, dtype=torch.float32)
    out = _inv_pth_root(I, 4, eps=1e-12)
    assert torch.allclose(out, I, atol=1e-5)


def test_inv_pth_root_diagonal():
    """For a diagonal matrix the inverse-pth-root is element-wise on the diagonal."""
    diag = torch.tensor([1.0, 4.0, 9.0, 16.0])
    M = torch.diag(diag)
    out = _inv_pth_root(M, 4, eps=1e-12)
    expected_diag = diag.pow(-0.25)
    assert torch.allclose(torch.diagonal(out), expected_diag, atol=1e-5)
    # Off-diagonal should be (numerically) zero.
    off = out - torch.diag(torch.diagonal(out))
    assert off.abs().max() < 1e-5


def test_shampoo_decreases_loss():
    """Shampoo on linear regression: loss strictly decreases over a 50-step window
    and converges to low loss within 200 steps."""
    torch.manual_seed(0)
    d_in, d_out, N = 8, 4, 64
    X = torch.randn(N, d_in)
    W_star = torch.randn(d_out, d_in)
    y = X @ W_star.T

    W = torch.zeros(d_out, d_in, requires_grad=True)
    opt = Shampoo([W], lr=0.1, momentum=0.9, shampoo_beta=0.95,
                  precondition_frequency=5, correct_factor_bias=True)
    losses = []
    for _ in range(200):
        opt.zero_grad()
        loss = _linreg_loss(W, X, y)
        loss.backward()
        opt.step()
        losses.append(loss.item())

    # Loss-decrease check: every 50-step window's tail must be below its head.
    for start in range(0, 150, 50):
        head = sum(losses[start:start + 5]) / 5
        tail = sum(losses[start + 45:start + 50]) / 5
        assert tail < head, (
            f"Loss did not decrease in window [{start}, {start+50}): head={head:.4g}, tail={tail:.4g}"
        )

    # Convergence check: final loss should be small. We don't claim to beat
    # SGD-with-momentum on this toy 4x8 regression -- low-dim full-batch SGDM is
    # hard to dominate. We only check the optimizer is functional and converges.
    assert losses[-1] < 1e-2, f"Shampoo failed to converge: final loss {losses[-1]:.4g}"


def test_factor_bias_correction_matters():
    """First-step parameter delta should be ~ (1 - shampoo_beta)^(-1/2) ~ 4.47x larger
    when correct_factor_bias is OFF (with default shampoo_beta=0.95).

    Reasoning: with correction OFF, L = (1-beta)*GG^T at step 1, so
    inv_root_L = ((1-beta)*GG^T)^(-1/4) = (1-beta)^(-1/4) * (GG^T)^(-1/4).
    Same factor on R. update = inv_root_L @ m @ inv_root_R picks up
    (1-beta)^(-1/2) overall versus the bias-corrected version.
    """
    torch.manual_seed(0)
    d_in, d_out = 6, 4
    grad_template = torch.randn(d_out, d_in)
    beta = 0.95

    def first_step_norm(correct_factor_bias):
        torch.manual_seed(0)
        W = torch.zeros(d_out, d_in, requires_grad=True)
        # Use SGD-style momentum=0 so the EMA equals the raw gradient at step 1
        # (lerp_(grad, 1-0)=grad). This isolates the L,R bias-correction effect.
        opt = Shampoo([W], lr=1.0, momentum=0.0, shampoo_beta=beta,
                      precondition_frequency=1, correct_bias=False,
                      correct_factor_bias=correct_factor_bias)
        # Manually inject the gradient (bypasses autograd; we want a controlled grad).
        W.grad = grad_template.clone()
        opt.step()
        return W.detach().norm().item()

    delta_corrected = first_step_norm(correct_factor_bias=True)
    delta_uncorrected = first_step_norm(correct_factor_bias=False)
    expected_ratio = (1 - beta) ** (-0.5)  # ~4.47 for beta=0.95
    actual_ratio = delta_uncorrected / delta_corrected

    assert abs(actual_ratio - expected_ratio) / expected_ratio < 0.05, (
        f"Bias-correction ratio off: actual {actual_ratio:.3f}, expected {expected_ratio:.3f}"
    )


def test_shampoo_rejects_non_2d_param():
    W1d = torch.zeros(8, requires_grad=True)
    opt = Shampoo([W1d], lr=0.1)
    W1d.grad = torch.ones_like(W1d)
    with pytest.raises(TypeError, match="2D"):
        opt.step()


def test_shampoo_rejects_split_qkv():
    W = torch.zeros(4, 4, requires_grad=True)
    with pytest.raises(NotImplementedError, match="split_qkv"):
        Shampoo([W], lr=0.1, split_qkv=True)
