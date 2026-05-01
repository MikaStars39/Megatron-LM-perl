# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""GASD optimizer: Geometry-Aware Steepest Descent for RLVR.

Applies Muon orthogonalization first, then GASD preconditioning via CG:

    M_t = beta * M_{t-1} + (1-beta) * G_t          # EMA momentum
    G_nesterov = lerp(G_t, M_t, beta)               # Nesterov lookahead (optional)
    Phi = NewtonSchulz(G_nesterov) * scale           # Muon orthogonalization
    Solve (WW^T + eps*I) Delta = Phi via CG          # GASD preconditioning
    Delta = Delta / RMS(Delta) * scale               # RMS normalization
    W_t = W_{t-1} - lr * (Delta + lambda * W_{t-1})  # Update (decoupled WD)

The GASD preconditioner (WW^T + eps*I)^{-1} encodes the weight's spectral
structure: principal directions (large singular values) are damped, while
off-principal directions (small singular values) are amplified. This matches
the Three-Gate Theory observation that RLVR updates preferentially occur in
off-principal subspaces.

W is the current weight at each step (recomputed every step).
eps is computed adaptively per layer: eps = alpha(t) * ||W||_F^2 / min(n, m),
where alpha(t) can be annealed from epsilon_alpha to epsilon_alpha_final over training.
"""

import logging
import math
from typing import Callable, Dict, List, Optional

import torch
from torch.optim.optimizer import ParamsT

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.module import MegatronModule
from megatron.core.utils import get_pg_size, log_single_rank

from . import _get_param_groups, get_megatron_optimizer
from .optimizer import (
    ChainedOptimizer,
    Float16OptimizerWithFloat16Params,
    FP32Optimizer,
    MegatronOptimizer,
)
from .optimizer_config import OptimizerConfig, ParamKey

try:
    from emerging_optimizers.orthogonalized_optimizers import get_muon_scale_factor
    from emerging_optimizers.orthogonalized_optimizers.muon_utils import (
        newton_schulz,
        newton_schulz_tp,
    )

    HAVE_EMERGING_OPTIMIZERS = True
except ImportError:
    HAVE_EMERGING_OPTIMIZERS = False

logger = logging.getLogger(__name__)


class GASD(torch.optim.Optimizer):
    """GASD: Geometry-Aware Steepest Descent.

    Applies Muon orthogonalization (Newton-Schulz) for gradient normalization,
    then solves (WW^T + eps*I) Delta = Phi via Conjugate Gradient to precondition
    the update based on the current weight's spectral geometry.
    Finally applies RMS normalization to stabilize the update magnitude.

    Only applies the Muon+GASD transform to 2D parameters.
    Non-2D parameters (embedding, LayerNorm, bias) are updated directly.

    Args:
        params: Parameters to optimize.
        lr: Learning rate.
        momentum: Momentum coefficient (beta) for EMA.
        weight_decay: Decoupled weight decay coefficient.
        use_nesterov: Whether to use Nesterov-style momentum.
        epsilon_alpha: Minimum coefficient for adaptive epsilon: eps = alpha(t) * ||W||_F^2 / min(n,m).
        epsilon_alpha_max: Maximum coefficient for epsilon annealing.
        epsilon_warmup_steps: Hold epsilon_alpha constant for this many steps.
        epsilon_ramp_end_steps: Step at which alpha reaches epsilon_alpha_max. Linear ramp between warmup and this.
        cg_iters: Maximum number of Conjugate Gradient iterations (default 10).
        cg_rtol: Relative residual tolerance for CG early stopping (default 1e-5).
            CG terminates early when ||residual|| / ||rhs|| < cg_rtol. Set to 0 to disable.
        cg_iters_min: Minimum CG iterations after decay. None = no decay. 0 = decay to pure Muon.
            When set, CG iters decay from cg_iters to cg_iters_min following the epsilon
            warmup/ramp schedule. Output is blended: w*GASD + (1-w)*Muon where w=cg_iters_t/cg_iters.
        cg_decay_style: Decay style for CG iters: 'linear' or 'cosine'.
        rms_scale: Scale factor after RMS normalization of the CG output (default 1.0).
        split_qkv: Whether to split QKV parameters.
        is_qkv_fn: Function to check if a parameter is QKV.
        qkv_split_shapes: Shapes for QKV splitting.
        num_ns_steps: Number of Newton-Schulz iteration steps for Muon.
        coefficient_type: NS coefficient type.
        scale_mode: Muon scale mode.
        extra_scale_factor: Additional scale factor for the Muon update.
        fp32_matmul_prec: FP32 matmul precision for NS iteration.
        tp_mode: TP handling mode ('blockwise', 'duplicated', 'distributed').
        pg_collection: Process group collection for TP.
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 3e-4,
        momentum: float = 0.95,
        weight_decay: float = 0.01,
        use_nesterov: bool = True,
        epsilon_alpha: float = 1.0,
        epsilon_alpha_max: float = 1.0,
        epsilon_warmup_steps: int = 50,
        epsilon_ramp_end_steps: int = 800,
        cg_iters: int = 10,
        cg_rtol: float = 1e-5,
        cg_iters_min: Optional[int] = None,
        cg_decay_style: str = "linear",
        rms_scale: float = 1.0,
        split_qkv: bool = False,
        is_qkv_fn: Optional[Callable[[torch.Tensor], bool]] = None,
        qkv_split_shapes: Optional[tuple] = None,
        num_ns_steps: int = 5,
        coefficient_type: str = "quintic",
        scale_mode: str = "spectral",
        extra_scale_factor: float = 1.0,
        fp32_matmul_prec: str = "medium",
        tp_mode: str = "blockwise",
        pg_collection: Optional[ProcessGroupCollection] = None,
    ) -> None:
        if num_ns_steps < 1:
            raise ValueError(f"num_ns_steps must be at least 1, got {num_ns_steps}")
        if cg_iters < 1:
            raise ValueError(f"cg_iters must be at least 1, got {cg_iters}")
        if cg_rtol < 0:
            raise ValueError(f"cg_rtol must be non-negative, got {cg_rtol}")
        if cg_iters_min is not None and not (0 <= cg_iters_min <= cg_iters):
            raise ValueError(
                f"cg_iters_min must be in [0, cg_iters={cg_iters}], got {cg_iters_min}"
            )

        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

        self.use_nesterov = use_nesterov
        self.epsilon_alpha = epsilon_alpha
        self.epsilon_alpha_max = epsilon_alpha_max
        self.epsilon_warmup_steps = epsilon_warmup_steps
        self.epsilon_ramp_end_steps = epsilon_ramp_end_steps
        self.cg_iters = cg_iters
        self.cg_rtol = cg_rtol
        self.cg_iters_min = cg_iters_min
        self.cg_decay_style = cg_decay_style
        self.rms_scale = rms_scale
        self.split_qkv = split_qkv
        self.is_qkv_fn = is_qkv_fn
        self.qkv_split_shapes = qkv_split_shapes
        self.num_ns_steps = num_ns_steps
        self.coefficient_type = coefficient_type
        self.scale_mode = scale_mode
        self.extra_scale_factor = extra_scale_factor
        self.fp32_matmul_prec = fp32_matmul_prec
        self.tp_mode = tp_mode
        self.pg_collection = pg_collection
        self._step_count = 0

    def state_dict(self):
        d = super().state_dict()
        d['gasd_step_count'] = self._step_count
        return d

    def load_state_dict(self, state_dict):
        self._step_count = state_dict.pop('gasd_step_count', 0)
        super().load_state_dict(state_dict)

    def _get_tp_info(self, p: torch.Tensor):
        """Get TP group and partition dim for a parameter."""
        if self.pg_collection:
            tp_group = (
                self.pg_collection.expt_tp
                if getattr(p, 'expert_tp', False)
                else self.pg_collection.tp
            )
        else:
            tp_group = None

        partition_dim = (
            None if self.tp_mode == "blockwise"
            else getattr(p, "partition_dim", None)
        )
        if partition_dim == -1:
            partition_dim = None

        return tp_group, partition_dim

    def _muon_orthogonalize_single(
        self,
        update: torch.Tensor,
        tp_group: Optional[torch.distributed.ProcessGroup],
        partition_dim: Optional[int],
    ) -> torch.Tensor:
        """Apply Muon orthogonalization (Newton-Schulz + scale) to a single 2D tensor."""
        orig_prec = torch.get_float32_matmul_precision()
        torch.set_float32_matmul_precision(self.fp32_matmul_prec)

        orig_dtype = update.dtype
        update_f32 = update.float()

        if partition_dim is None:
            orth = newton_schulz(
                update_f32,
                steps=self.num_ns_steps,
                coefficient_type=self.coefficient_type,
            )
        else:
            orth = newton_schulz_tp(
                update_f32,
                steps=self.num_ns_steps,
                coefficient_type=self.coefficient_type,
                tp_group=tp_group,
                partition_dim=partition_dim,
                tp_mode="duplicated" if self.tp_mode == "blockwise" else self.tp_mode,
            )

        # Compute scale factor accounting for TP
        size = [update.size(-2), update.size(-1)]
        if partition_dim is not None and tp_group is not None:
            size[partition_dim] *= get_pg_size(tp_group)
        scale_factor = get_muon_scale_factor(size[0], size[1], mode=self.scale_mode)
        result = orth * scale_factor * self.extra_scale_factor

        torch.set_float32_matmul_precision(orig_prec)
        return result.to(orig_dtype)

    def _get_cg_iters(self) -> int:
        """Compute current CG iteration count based on decay schedule.

        Reuses epsilon warmup/ramp window. Returns self.cg_iters when no decay.
        """
        if self.cg_iters_min is None or self.cg_iters_min >= self.cg_iters:
            return self.cg_iters
        step = self._step_count
        if step <= self.epsilon_warmup_steps:
            return self.cg_iters
        if step >= self.epsilon_ramp_end_steps:
            return self.cg_iters_min
        t = (step - self.epsilon_warmup_steps) / (
            self.epsilon_ramp_end_steps - self.epsilon_warmup_steps
        )
        if self.cg_decay_style == "cosine":
            t = (1 - math.cos(math.pi * t)) / 2
        cg_float = self.cg_iters * (1 - t) + self.cg_iters_min * t
        return max(self.cg_iters_min, round(cg_float))

    def _apply_gasd(self, muon_update: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        """Solve (WW^T + eps*I) Delta = Phi via batch Conjugate Gradient.

        Avoids forming the [n, n] matrix WW^T explicitly. Each CG iteration
        only requires two matmuls: v = W^T @ P, then AP = W @ v + eps * P.
        W is the current weight (recomputed every step).

        When CG iter decay is enabled, blends the RMS-normalized CG output with
        the raw Muon output (Phi) using w = cg_iters_t / cg_iters. This smoothly
        transitions from GASD (w=1, RMS-normed) to pure Muon (w=0, natural scale).
        """
        cg_iters_t = self._get_cg_iters()

        # Pure Muon: skip CG entirely, preserve Muon's natural scale (e.g. 0.2)
        if cg_iters_t == 0:
            return muon_update

        orig_dtype = muon_update.dtype
        Phi = muon_update.float()
        W_f32 = W.detach().float()
        n, m = W_f32.shape

        # Three-stage epsilon: [0, warmup] = alpha_min, [warmup, ramp_end] = linear, [ramp_end, ∞] = alpha_max
        step = self._step_count
        if step <= self.epsilon_warmup_steps:
            alpha_t = self.epsilon_alpha
        elif step >= self.epsilon_ramp_end_steps:
            alpha_t = self.epsilon_alpha_max
        else:
            t = (step - self.epsilon_warmup_steps) / (self.epsilon_ramp_end_steps - self.epsilon_warmup_steps)
            alpha_t = self.epsilon_alpha + t * (self.epsilon_alpha_max - self.epsilon_alpha)
        eps = alpha_t

        # Batch CG: solve (WW^T + eps*I) Delta = Phi
        Delta = torch.zeros_like(Phi)
        R = Phi.clone()
        P = R.clone()
        rr = (R * R).sum()
        rr_init = rr
        cg_tol_sq = self.cg_rtol ** 2 * rr_init

        for cg_step in range(cg_iters_t):
            # A @ P = W @ (W^T @ P) + eps * P
            AP = W_f32 @ (W_f32.t() @ P) + eps * P

            pAp = (P * AP).sum()
            alpha_cg = rr / pAp.clamp_min(1e-30)

            Delta = Delta + alpha_cg * P
            R = R - alpha_cg * AP

            rr_new = (R * R).sum()

            # Early stopping: relative residual ||r||/||b|| < cg_rtol
            if rr_new < cg_tol_sq:
                rr = rr_new
                break

            beta_cg = rr_new / rr.clamp_min(1e-30)
            P = R + beta_cg * P
            rr = rr_new
        else:
            # CG did not converge — log warning
            rel_residual = (rr / rr_init.clamp_min(1e-30)).sqrt().item()
            log_single_rank(
                logger,
                logging.WARNING,
                f"GASD CG not converged: {cg_iters_t} iters, "
                f"rel_residual={rel_residual:.2e}, rtol={self.cg_rtol:.1e}, "
                f"shape={tuple(W_f32.shape)}, eps={eps:.2e}, step={self._step_count}",
            )

        # RMS normalization: Delta = Delta / RMS(Delta) * scale
        rms = (Delta.square().mean()).sqrt().clamp_min(1e-12)
        Delta = Delta / rms * self.rms_scale

        # Blend GASD and Muon outputs for smooth scale transition
        # w=1.0: full GASD (RMS-normed), w=0.0: pure Muon (natural scale)
        if self.cg_iters_min is not None and cg_iters_t < self.cg_iters:
            w = cg_iters_t / self.cg_iters
            Delta = w * Delta + (1 - w) * Phi

        return Delta.to(orig_dtype)

    def _transform_single(
        self,
        update: torch.Tensor,
        W: torch.Tensor,
        tp_group: Optional[torch.distributed.ProcessGroup],
        partition_dim: Optional[int],
    ) -> torch.Tensor:
        """Apply Muon orthogonalization + GASD preconditioning to a single 2D tensor."""
        muon_update = self._muon_orthogonalize_single(update, tp_group, partition_dim)
        return self._apply_gasd(muon_update, W)

    def _transform_qkv(
        self,
        update: torch.Tensor,
        W: torch.Tensor,
        tp_group: Optional[torch.distributed.ProcessGroup],
        partition_dim: Optional[int],
    ) -> torch.Tensor:
        """Apply Muon+GASD to QKV parameter by splitting into Q, K, V components."""
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

        # Apply Muon+GASD to each Q, K, V component independently
        qkv_transformed = [
            self._transform_single(u, w, tp_group, partition_dim).view(
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

                # Non-2D params: direct update (no Muon/GASD)
                if grad.ndim != 2:
                    if weight_decay != 0:
                        p.data.mul_(1 - lr * weight_decay)
                    p.data.add_(update, alpha=-lr)
                    continue

                # Get TP info
                tp_group, partition_dim = self._get_tp_info(p)

                # Apply Muon orthogonalization + GASD preconditioning
                if self.split_qkv and self.is_qkv_fn is not None and self.is_qkv_fn(p):
                    delta = self._transform_qkv(update, p.data, tp_group, partition_dim)
                else:
                    delta = self._transform_single(update, p.data, tp_group, partition_dim)

                # Decoupled weight decay: W = (1 - lr*lambda) W - lr*Delta
                if weight_decay != 0:
                    p.data.mul_(1 - lr * weight_decay)
                p.data.add_(delta, alpha=-lr)

        self._step_count += 1
        return loss


def get_megatron_gasd_optimizer(
    config: OptimizerConfig,
    model_chunks: List[MegatronModule],
    config_overrides: Optional[Dict[ParamKey, OptimizerConfig]] = None,
    use_gloo_process_groups: bool = True,
    pg_collection: Optional[ProcessGroupCollection] = None,
) -> MegatronOptimizer:
    """Create GASD optimizer for Megatron model chunks.

    Follows the same pattern as get_megatron_muon_projected_optimizer():
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

    assert HAVE_EMERGING_OPTIMIZERS, (
        "GASD requires 'emerging_optimizers' package for Muon orthogonalization. "
        "Install from: https://github.com/NVIDIA-NeMo/Emerging-Optimizers.git@v0.1.0"
    )

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
        epsilon_alpha_max=config.gasd_epsilon_alpha_max,
        epsilon_warmup_steps=config.gasd_epsilon_warmup_steps,
        epsilon_ramp_end_steps=config.gasd_epsilon_ramp_end_steps,
        cg_iters=config.gasd_cg_iters,
        cg_rtol=config.gasd_cg_rtol,
        cg_iters_min=config.gasd_cg_iters_min,
        cg_decay_style=config.gasd_cg_decay_style,
        rms_scale=config.gasd_rms_scale,
        num_ns_steps=config.gasd_num_ns_steps,
        scale_mode=config.gasd_scale_mode,
        extra_scale_factor=config.gasd_extra_scale_factor,
        fp32_matmul_prec=config.gasd_fp32_matmul_prec,
        tp_mode=config.gasd_tp_mode,
        split_qkv=config.gasd_split_qkv,
        is_qkv_fn=lambda p: getattr(p, 'is_qkv', False),
        qkv_split_shapes=qkv_split_shapes,
        pg_collection=pg_collection,
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
