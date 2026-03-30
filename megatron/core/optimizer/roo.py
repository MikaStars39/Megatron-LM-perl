# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Roo (Matrix Natural Gradient) optimizer for Megatron-LM.

Implements the spectral transform f(σ) = σ/(σ²+ε²) via Gram matrix inversion
using Newton-Schulz iteration (pure GEMM, Tensor Core friendly).

Core algorithm:
    M_t = β * M_{t-1} + G_t                    # Momentum accumulation
    A = M_t^T @ M_t + ε² I                     # Gram matrix + regularization
    A_inv ≈ NS_inverse(A, steps=5)              # Newton-Schulz inverse iteration
    Φ = M_t @ A_inv                             # Matrix natural gradient direction
    Φ = Φ / RMS(Φ) * scale_factor              # RMS normalization
    W_t = W_{t-1} - η * (Φ + λ * W_{t-1})     # Update (decoupled weight decay)
"""

import logging
from typing import Any, Callable, Dict, List, Optional

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

logger = logging.getLogger(__name__)


def newton_schulz_inverse(A: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Approximate inverse of symmetric positive definite matrix A via Newton-Schulz iteration.

    All operations are GEMM (matrix multiply), fully utilizing GPU Tensor Cores.

    X_0 = I / ||A||_F
    X_{k+1} = X_k @ (2I - A_normalized @ X_k)

    Args:
        A: Symmetric positive definite matrix [n, n] (e.g., M^T M + ε²I).
        steps: Number of iteration steps (default 5).

    Returns:
        Approximate inverse of A, shape [n, n].
    """
    n = A.shape[0]

    # Normalize to ensure convergence: ||I - Ã X_0||₂ < 1
    c = A.norm()  # Frobenius norm
    A_tilde = A / c

    # X_0 = I
    X = torch.eye(n, dtype=A.dtype, device=A.device)

    # Iterate: X_{k+1} = X_k @ (2I - A_tilde @ X_k)
    I_2 = 2.0 * torch.eye(n, dtype=A.dtype, device=A.device)
    for _ in range(steps):
        X = X @ (I_2 - A_tilde @ X)

    # Denormalize
    return X / c


def compute_gram_with_tp(
    M: torch.Tensor,
    eps: float,
    tp_group: Optional[torch.distributed.ProcessGroup],
    partition_dim: Optional[int],
    tp_mode: str,
) -> torch.Tensor:
    """Compute Gram matrix M^T M + ε²I with tensor-parallel awareness.

    For row-parallel (partition_dim=0): M^T M = Σ M_i^T M_i, needs allreduce.
    For col-parallel (partition_dim=1): each rank independent, no communication.
    """
    gram = M.t() @ M

    if (
        tp_group is not None
        and tp_mode == "allreduce_gram"
        and partition_dim == 0
    ):
        torch.distributed.all_reduce(gram, group=tp_group)

    gram.diagonal().add_(eps * eps)
    return gram


class Roo(torch.optim.Optimizer):
    """Roo (Matrix Natural Gradient) optimizer.

    Applies spectral transform f(σ) = σ/(σ²+ε²) to momentum via Gram matrix
    inversion using Newton-Schulz iteration. Only applies to 2D parameters.

    Args:
        params: Parameters to optimize.
        lr: Learning rate.
        momentum: Momentum coefficient (β).
        epsilon: Regularization epsilon for Gram matrix.
        scale_factor: Scale factor after RMS normalization.
        weight_decay: Decoupled weight decay coefficient.
        split_qkv: Whether to split QKV parameters.
        is_qkv_fn: Function to check if a parameter is QKV.
        qkv_split_shapes: Shapes for QKV splitting.
        num_ns_steps: Number of Newton-Schulz iteration steps.
        fp32_matmul_prec: FP32 matmul precision for NS iteration.
        tp_mode: TP handling mode ('allreduce_gram' or 'blockwise').
        pg_collection: Process group collection for TP.
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 3e-4,
        momentum: float = 0.95,
        epsilon: float = 0.01,
        scale_factor: float = 1.0,
        weight_decay: float = 0.01,
        split_qkv: bool = False,
        is_qkv_fn: Optional[Callable[[torch.Tensor], bool]] = None,
        qkv_split_shapes: Optional[tuple] = None,
        num_ns_steps: int = 5,
        fp32_matmul_prec: str = "medium",
        tp_mode: str = "allreduce_gram",
        pg_collection: Optional[ProcessGroupCollection] = None,
    ) -> None:
        if num_ns_steps < 1:
            raise ValueError(f"num_ns_steps must be at least 1, got {num_ns_steps}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            epsilon=epsilon,
            scale_factor=scale_factor,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

        self.num_ns_steps = num_ns_steps
        self.fp32_matmul_prec = fp32_matmul_prec
        self.tp_mode = tp_mode
        self.pg_collection = pg_collection
        self.split_qkv = split_qkv
        self.is_qkv_fn = is_qkv_fn
        self.qkv_split_shapes = qkv_split_shapes

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

    def _roo_transform_single(
        self,
        M: torch.Tensor,
        epsilon: float,
        scale_factor: float,
        tp_group: Optional[torch.distributed.ProcessGroup],
        partition_dim: Optional[int],
    ) -> torch.Tensor:
        """Apply Roo spectral transform to a single 2D momentum tensor.

        Steps:
            1. Compute Gram matrix: A = M^T M + ε²I (with TP allreduce if needed)
            2. Newton-Schulz inverse: A_inv ≈ NS(A, steps)
            3. Transform: Φ = M @ A_inv
            4. RMS normalize and scale
        """
        # Save and set matmul precision
        orig_prec = torch.backends.cuda.matmul.allow_tf32
        if self.fp32_matmul_prec == "high":
            torch.backends.cuda.matmul.allow_tf32 = False
        else:
            torch.backends.cuda.matmul.allow_tf32 = True

        # Upcast to float32 for numerical stability
        orig_dtype = M.dtype
        M_f32 = M.float()

        # Gram matrix with TP awareness
        gram = compute_gram_with_tp(M_f32, epsilon, tp_group, partition_dim, self.tp_mode)

        # Newton-Schulz inverse
        A_inv = newton_schulz_inverse(gram, steps=self.num_ns_steps)

        # Apply transform: Φ = M @ A_inv
        phi = M_f32 @ A_inv

        # Cast back to original dtype
        phi = phi.to(orig_dtype)

        # RMS normalization + scale
        rms = phi.norm() / (phi.numel() ** 0.5)
        if rms > 0:
            phi = phi / rms * scale_factor

        # Restore matmul precision
        torch.backends.cuda.matmul.allow_tf32 = orig_prec

        return phi

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single Roo optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            momentum_beta = group['momentum']
            epsilon = group['epsilon']
            scale_factor = group['scale_factor']
            lr = group['lr']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                # Only apply Roo transform to 2D parameters
                if grad.ndim != 2:
                    # For non-2D params, just do SGD with momentum
                    state = self.state[p]
                    if len(state) == 0:
                        state['momentum_buffer'] = torch.zeros_like(p.data)
                    buf = state['momentum_buffer']
                    buf.mul_(momentum_beta).add_(grad)
                    if weight_decay != 0:
                        p.data.mul_(1 - lr * weight_decay)
                    p.data.add_(buf, alpha=-lr)
                    continue

                # Initialize momentum buffer
                state = self.state[p]
                if len(state) == 0:
                    state['momentum_buffer'] = torch.zeros_like(p.data)

                buf = state['momentum_buffer']
                # M_t = β * M_{t-1} + G_t
                buf.mul_(momentum_beta).add_(grad)

                # Get TP info for this parameter
                tp_group, partition_dim = self._get_tp_info(p)

                # Apply Roo transform (with QKV splitting if needed)
                if self.split_qkv and self.is_qkv_fn is not None and self.is_qkv_fn(p):
                    phi = self._roo_transform_qkv(
                        buf, epsilon, scale_factor, tp_group, partition_dim
                    )
                else:
                    phi = self._roo_transform_single(
                        buf, epsilon, scale_factor, tp_group, partition_dim
                    )

                # Decoupled weight decay: W = (1 - η·λ) W - η·Φ
                if weight_decay != 0:
                    p.data.mul_(1 - lr * weight_decay)
                p.data.add_(phi, alpha=-lr)

        return loss

    def _roo_transform_qkv(
        self,
        M: torch.Tensor,
        epsilon: float,
        scale_factor: float,
        tp_group: Optional[torch.distributed.ProcessGroup],
        partition_dim: Optional[int],
    ) -> torch.Tensor:
        """Apply Roo transform to QKV parameter by splitting into Q, K, V components."""
        grad_shape = M.shape
        num_query_groups = grad_shape[0] // sum(self.qkv_split_shapes)
        qkv_parts = torch.split(
            M.view(num_query_groups, sum(self.qkv_split_shapes), -1),
            self.qkv_split_shapes,
            dim=1,
        )
        qkv_parts = [g.reshape(-1, grad_shape[-1]) for g in qkv_parts]

        # Apply Roo transform to each Q, K, V component independently
        qkv_transformed = [
            self._roo_transform_single(
                g, epsilon, scale_factor, tp_group, partition_dim
            ).view(num_query_groups, -1, grad_shape[-1])
            for g in qkv_parts
        ]
        return torch.cat(qkv_transformed, dim=1).view(grad_shape)


def get_megatron_roo_optimizer(
    config: OptimizerConfig,
    model_chunks: List[MegatronModule],
    config_overrides: Optional[Dict[ParamKey, OptimizerConfig]] = None,
    use_gloo_process_groups: bool = True,
    pg_collection: Optional[ProcessGroupCollection] = None,
) -> MegatronOptimizer:
    """Create Roo optimizer for Megatron model chunks.

    Follows the same pattern as get_megatron_muon_optimizer():
    1. Split params into linear (2D, non-embedding) and nonlinear
    2. Freeze nonlinear → create Roo optimizer for linear params → wrap bf16
    3. Freeze linear → create Adam for nonlinear params
    4. Unfreeze all → return ChainedOptimizer([roo_wrapped, adam_wrapped])

    Args:
        config: Optimizer configuration.
        model_chunks: List of model chunks.
        config_overrides: Per-parameter config overrides.
        use_gloo_process_groups: Whether to use Gloo process groups.
        pg_collection: Process group collection for TP.

    Returns:
        ChainedOptimizer containing Roo (for linear) + Adam (for nonlinear).
    """
    # Roo uses adam config for the nonlinear params path
    config.optimizer = 'adam'

    if config.use_distributed_optimizer:
        raise Exception('Roo with distributed optimizer is not supported.')

    if pg_collection is None:
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()

    log_single_rank(logger, logging.INFO, f'Setting up Roo optimizer with config {config}')

    optimizers = []
    linear_params = []
    nonlinear_params = []

    for model_chunk in model_chunks:
        # Get QKV split shapes from model config
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
            # Mark expert TP parameters
            if 'experts' in name and 'shared' not in name:
                param.expert_tp = True
            # Mark QKV parameters
            if 'linear_qkv.weight' in name and len(param.shape) == 2:
                param.is_qkv = True
            # Split: 2D non-embedding → linear (Roo), rest → nonlinear (Adam)
            if (
                not getattr(param, 'is_embedding_or_output_parameter', False)
                and len(param.shape) == 2
            ):
                linear_params.append(param)
            else:
                nonlinear_params.append(param)

    # === Roo optimizer for linear params ===
    # Freeze nonlinear params and create param groups for Roo
    for param in nonlinear_params:
        param.requires_grad = False

    linear_param_groups = _get_param_groups(model_chunks, config, config_overrides)

    roo_optimizer = Roo(
        linear_param_groups,
        lr=config.lr,
        momentum=config.roo_momentum,
        epsilon=config.roo_epsilon,
        scale_factor=config.roo_scale_factor,
        weight_decay=config.weight_decay,
        num_ns_steps=config.roo_num_ns_steps,
        fp32_matmul_prec=config.roo_fp32_matmul_prec,
        tp_mode=config.roo_tp_mode,
        split_qkv=config.roo_split_qkv,
        is_qkv_fn=lambda p: getattr(p, 'is_qkv', False),
        qkv_split_shapes=qkv_split_shapes,
        pg_collection=pg_collection,
    )

    def roo_init_state_fn(opt, config=None):
        for group in opt.param_groups:
            for p in group['params']:
                if len(opt.state[p]) == 0:
                    opt.state[p]['momentum_buffer'] = torch.zeros_like(p.data)

    def adam_init_state_fn(opt, config=None):
        for group in opt.param_groups:
            for p in group['params']:
                if len(opt.state[p]) == 0:
                    if config is None or not config.use_precision_aware_optimizer:
                        opt.state[p]['exp_avg'] = torch.zeros_like(p.data)
                        opt.state[p]['exp_avg_sq'] = torch.zeros_like(p.data)
                    else:
                        opt.initialize_state(p)

    # Wrap in mixed precision optimizer
    if config.fp16:
        raise Exception('Roo with fp16 is not supported.')

    if config.bf16:
        roo_wrapped = Float16OptimizerWithFloat16Params(
            roo_optimizer, config, None, roo_init_state_fn
        )
    else:
        roo_wrapped = FP32Optimizer(roo_optimizer, config, roo_init_state_fn)

    optimizers.append(roo_wrapped)

    # === Adam optimizer for nonlinear params ===
    # Unfreeze nonlinear, freeze linear
    for param in nonlinear_params:
        param.requires_grad = True
    for param in linear_params:
        param.requires_grad = False

    # Create Adam for nonlinear params (linear params are frozen, so skipped)
    chained_adam = get_megatron_optimizer(
        config,
        model_chunks,
        config_overrides=config_overrides,
        use_gloo_process_groups=use_gloo_process_groups,
    )

    # Unfreeze everything
    for param in linear_params:
        param.requires_grad = True

    # Chain Roo + Adam
    optimizers += chained_adam.chained_optimizers
    return ChainedOptimizer(optimizers)
