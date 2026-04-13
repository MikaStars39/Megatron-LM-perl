# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""GASD optimizer: Geometry-Aware Steepest Descent for RLVR.

Pure geometry-based preconditioner using the weight's spectral structure:

    M_t = beta * M_{t-1} + (1-beta) * G_t          # EMA momentum
    G_nesterov = lerp(G_t, M_t, beta)               # Nesterov lookahead (optional)
    Solve (WW^T + eps*I) Delta = G_nesterov via CG   # GASD preconditioning
    Delta = Delta / RMS(Delta) * scale                # RMS normalization
    W_t = W_{t-1} - lr * (Delta + lambda * W_{t-1})  # Update (decoupled WD)

The GASD preconditioner (WW^T + eps*I)^{-1} encodes the weight's spectral
structure: principal directions (large singular values) are damped, while
off-principal directions (small singular values) are amplified. This matches
the Three-Gate Theory observation that RLVR updates preferentially occur in
off-principal subspaces.

W is the current weight at each step (recomputed every step).
eps is computed adaptively per layer: eps = alpha * ||W||_F^2 / min(n, m).
"""

import logging
from typing import Callable, Dict, List, Optional

import torch
from torch.optim.optimizer import ParamsT

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.module import MegatronModule
from megatron.core.utils import log_single_rank

from . import _get_param_groups, get_megatron_optimizer
from .optimizer import (
    ChainedOptimizer,
    Float16OptimizerWithFloat16Params,
    FP32Optimizer,
    MegatronOptimizer,
)
from .optimizer_config import OptimizerConfig, ParamKey

logger = logging.getLogger(__name__)


class GASD(torch.optim.Optimizer):
    """GASD: Geometry-Aware Steepest Descent.

    Solves (WW^T + eps*I) Delta = G via Conjugate Gradient to precondition
    the update based on the current weight's spectral geometry.

    Only applies the GASD transform to 2D parameters.
    Non-2D parameters (embedding, LayerNorm, bias) are updated directly.

    Args:
        params: Parameters to optimize.
        lr: Learning rate.
        momentum: Momentum coefficient (beta) for EMA.
        weight_decay: Decoupled weight decay coefficient.
        use_nesterov: Whether to use Nesterov-style momentum.
        epsilon_alpha: Coefficient for adaptive epsilon: eps = alpha * ||W||_F^2 / min(n,m).
        cg_iters: Number of Conjugate Gradient iterations (default 10).
        rms_scale: Scale factor after RMS normalization of the CG output (default 1.0).
        split_qkv: Whether to split QKV parameters.
        is_qkv_fn: Function to check if a parameter is QKV.
        qkv_split_shapes: Shapes for QKV splitting.
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 3e-4,
        momentum: float = 0.95,
        weight_decay: float = 0.01,
        use_nesterov: bool = True,
        epsilon_alpha: float = 1.0,
        cg_iters: int = 10,
        rms_scale: float = 1.0,
        split_qkv: bool = False,
        is_qkv_fn: Optional[Callable[[torch.Tensor], bool]] = None,
        qkv_split_shapes: Optional[tuple] = None,
    ) -> None:
        if cg_iters < 1:
            raise ValueError(f"cg_iters must be at least 1, got {cg_iters}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

        self.use_nesterov = use_nesterov
        self.epsilon_alpha = epsilon_alpha
        self.cg_iters = cg_iters
        self.rms_scale = rms_scale
        self.split_qkv = split_qkv
        self.is_qkv_fn = is_qkv_fn
        self.qkv_split_shapes = qkv_split_shapes

    def _apply_gasd(self, update: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        """Solve (WW^T + eps*I) Delta = G via batch Conjugate Gradient.

        Uses the matrix CG method where Delta, residual R, and search direction P
        are all [n, m] matrices. Each CG iteration requires two matmuls:
          v = W^T @ P   ->  AP = W @ v + eps * P
        W is the current weight (recomputed every step).
        """
        orig_dtype = update.dtype
        G = update.float()
        W_f32 = W.detach().float()
        n, m = W_f32.shape

        # Adaptive epsilon: eps = alpha * ||W||_F^2 / min(n, m)
        fnorm_sq = W_f32.norm().square().clamp_min(1e-12)
        eps = self.epsilon_alpha * fnorm_sq / min(n, m)

        # Batch CG: solve (WW^T + eps*I) Delta = G
        Delta = torch.zeros_like(G)
        R = G.clone()
        P = R.clone()
        rr = (R * R).sum()

        for _ in range(self.cg_iters):
            # A @ P = W @ (W^T @ P) + eps * P
            AP = W_f32 @ (W_f32.t() @ P) + eps * P

            pAp = (P * AP).sum()
            alpha_cg = rr / pAp.clamp_min(1e-30)

            Delta = Delta + alpha_cg * P
            R = R - alpha_cg * AP

            rr_new = (R * R).sum()
            beta_cg = rr_new / rr.clamp_min(1e-30)
            P = R + beta_cg * P
            rr = rr_new

        # RMS normalization: Delta = Delta / RMS(Delta) * scale
        rms = (Delta.square().mean()).sqrt().clamp_min(1e-12)
        Delta = Delta / rms * self.rms_scale

        return Delta.to(orig_dtype)

    def _transform_single(
        self,
        update: torch.Tensor,
        W: torch.Tensor,
    ) -> torch.Tensor:
        """Apply GASD preconditioning to a single 2D tensor."""
        return self._apply_gasd(update, W)

    def _transform_qkv(
        self,
        update: torch.Tensor,
        W: torch.Tensor,
    ) -> torch.Tensor:
        """Apply GASD to QKV parameter by splitting into Q, K, V components."""
        grad_shape = update.shape
        num_query_groups = grad_shape[0] // sum(self.qkv_split_shapes)

        # Split update
        update_parts = torch.split(
            update.view(num_query_groups, sum(self.qkv_split_shapes), -1),
            self.qkv_split_shapes,
            dim=1,
        )
        update_parts = [g.reshape(-1, grad_shape[-1]) for g in update_parts]

        # Split W the same way
        W_parts = torch.split(
            W.view(num_query_groups, sum(self.qkv_split_shapes), -1),
            self.qkv_split_shapes,
            dim=1,
        )
        W_parts = [g.reshape(-1, grad_shape[-1]) for g in W_parts]

        # Apply GASD to each Q, K, V component independently
        qkv_transformed = [
            self._transform_single(u, w).view(
                num_query_groups, -1, grad_shape[-1]
            )
            for u, w in zip(update_parts, W_parts)
        ]
        return torch.cat(qkv_transformed, dim=1).view(grad_shape)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single GASD optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            momentum_beta = group['momentum']
            lr = group['lr']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                # Initialize momentum buffer
                state = self.state[p]
                if len(state) == 0:
                    state['momentum_buffer'] = torch.zeros_like(p.data)

                buf = state['momentum_buffer']

                # EMA momentum: buf = beta*buf + (1-beta)*grad
                buf.lerp_(grad, 1 - momentum_beta)

                # Nesterov momentum
                if self.use_nesterov:
                    update = grad.lerp(buf, momentum_beta)
                else:
                    update = buf

                # Non-2D params: direct update (no GASD)
                if grad.ndim != 2:
                    if weight_decay != 0:
                        p.data.mul_(1 - lr * weight_decay)
                    p.data.add_(update, alpha=-lr)
                    continue

                # Apply GASD preconditioning (using current weight)
                if self.split_qkv and self.is_qkv_fn is not None and self.is_qkv_fn(p):
                    delta = self._transform_qkv(update, p.data)
                else:
                    delta = self._transform_single(update, p.data)

                # Decoupled weight decay: W = (1 - lr*lambda) W - lr*Delta
                if weight_decay != 0:
                    p.data.mul_(1 - lr * weight_decay)
                p.data.add_(delta, alpha=-lr)

        return loss


def get_megatron_gasd_optimizer(
    config: OptimizerConfig,
    model_chunks: List[MegatronModule],
    config_overrides: Optional[Dict[ParamKey, OptimizerConfig]] = None,
    use_gloo_process_groups: bool = True,
    pg_collection: Optional[ProcessGroupCollection] = None,
) -> MegatronOptimizer:
    """Create GASD optimizer for Megatron model chunks.

    1. Split params into linear (2D, non-embedding) and nonlinear
    2. Freeze nonlinear -> create GASD for linear -> wrap bf16
    3. Freeze linear -> create Adam for nonlinear
    4. Unfreeze all -> return ChainedOptimizer

    Args:
        config: Optimizer configuration.
        model_chunks: List of model chunks.
        config_overrides: Per-parameter config overrides.
        use_gloo_process_groups: Whether to use Gloo process groups.
        pg_collection: Process group collection for TP.

    Returns:
        ChainedOptimizer containing GASD (for linear) + Adam (for nonlinear).
    """
    config.optimizer = 'adam'

    if config.use_distributed_optimizer:
        raise Exception('GASD with distributed optimizer is not supported.')

    if pg_collection is None:
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()

    log_single_rank(logger, logging.INFO, f'Setting up GASD optimizer with config {config}')

    optimizers = []
    linear_params = []
    nonlinear_params = []

    for model_chunk in model_chunks:
        num_attention_heads = model_chunk.config.num_attention_heads
        num_query_groups = model_chunk.config.num_query_groups
        kv_channels = model_chunk.config.kv_channels
        qkv_split_shapes = [
            num_attention_heads // num_query_groups * kv_channels,
            kv_channels,
            kv_channels,
        ]
        for name, param in model_chunk.named_parameters():
            if not param.requires_grad:
                continue
            if 'experts' in name and 'shared' not in name:
                param.expert_tp = True
            if 'linear_qkv.weight' in name and len(param.shape) == 2:
                param.is_qkv = True
            if (
                not getattr(param, 'is_embedding_or_output_parameter', False)
                and len(param.shape) == 2
            ):
                linear_params.append(param)
            else:
                nonlinear_params.append(param)

    # === GASD optimizer for linear params ===
    for param in nonlinear_params:
        param.requires_grad = False

    linear_param_groups = _get_param_groups(model_chunks, config, config_overrides)

    gasd_optimizer = GASD(
        linear_param_groups,
        lr=config.lr,
        momentum=config.gasd_momentum,
        weight_decay=config.weight_decay,
        use_nesterov=config.gasd_use_nesterov,
        epsilon_alpha=config.gasd_epsilon_alpha,
        cg_iters=config.gasd_cg_iters,
        rms_scale=config.gasd_rms_scale,
        split_qkv=config.gasd_split_qkv,
        is_qkv_fn=lambda p: getattr(p, 'is_qkv', False),
        qkv_split_shapes=qkv_split_shapes,
    )

    def gasd_init_state_fn(opt, config=None):
        for group in opt.param_groups:
            for p in group['params']:
                if len(opt.state[p]) == 0:
                    opt.state[p]['momentum_buffer'] = torch.zeros_like(p.data)

    if config.fp16:
        raise Exception('GASD with fp16 is not supported.')

    if config.bf16:
        gasd_wrapped = Float16OptimizerWithFloat16Params(
            gasd_optimizer, config, None, gasd_init_state_fn
        )
    else:
        gasd_wrapped = FP32Optimizer(
            gasd_optimizer, config, gasd_init_state_fn
        )

    optimizers.append(gasd_wrapped)

    # === Adam optimizer for nonlinear params ===
    for param in nonlinear_params:
        param.requires_grad = True
    for param in linear_params:
        param.requires_grad = False

    chained_adam = get_megatron_optimizer(
        config,
        model_chunks,
        config_overrides=config_overrides,
        use_gloo_process_groups=use_gloo_process_groups,
    )

    for param in linear_params:
        param.requires_grad = True

    optimizers += chained_adam.chained_optimizers
    return ChainedOptimizer(optimizers)
