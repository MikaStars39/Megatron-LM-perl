# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""RooNormal (SVD-based spectral clipping) optimizer for Megatron-LM.

Implements the spectral transform f(σ) = min(1/(σ+ε), clip_value) via explicit SVD.
This is a diagnostic variant of Roo that directly computes singular values for
inspection and applies hard clipping instead of soft regularization.

Uses Muon-style EMA momentum with optional Nesterov acceleration.

Core algorithm:
    M_t = β * M_{t-1} + (1-β) * G_t            # EMA momentum (Muon-style)
    G_nesterov = lerp(G_t, M_t, β)              # Nesterov lookahead (optional)
    U, S, Vh = SVD(G_nesterov)                   # Full SVD decomposition
    S_inv = 1 / (S + ε)                         # Inverse singular values
    S_clipped = clamp(S_inv, max=clip_value)    # Hard clip to prevent explosion
    Φ = U @ diag(S_clipped) @ Vh                # Reconstruct with clipped inverse SVs
    Φ = Φ / RMS(Φ) * scale_factor              # RMS normalization
    W_t = W_{t-1} - η * (Φ + λ * W_{t-1})     # Update (decoupled weight decay)
"""

import json
import logging
import os
from typing import Callable, Dict, List, Optional

import torch
from torch.optim.optimizer import ParamsT

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


class RooNormal(torch.optim.Optimizer):
    """RooNormal (SVD-based spectral clipping) optimizer.

    Applies spectral transform f(σ) = min(1/(σ+ε), clip_value) to momentum via
    explicit SVD decomposition. Uses Muon-style EMA momentum with optional Nesterov.
    Only applies SVD transform to 2D parameters.

    Args:
        params: Parameters to optimize.
        lr: Learning rate.
        momentum: Momentum coefficient (β) for EMA: buf = β*buf + (1-β)*grad.
        clip_value: Maximum value for inverse singular values (hard clip threshold).
        epsilon: Small constant added before taking reciprocal to prevent division by zero.
        scale_factor: Scale factor after RMS normalization.
        weight_decay: Decoupled weight decay coefficient.
        use_nesterov: Whether to use Nesterov-style momentum (default True, same as Muon).
        split_qkv: Whether to split QKV parameters.
        is_qkv_fn: Function to check if a parameter is QKV.
        qkv_split_shapes: Shapes for QKV splitting.
        svd_log_interval: Log SVD singular value statistics every N steps. 0 disables.
        svd_log_dir: Directory to write SVD log files. Empty string disables file logging.
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 3e-4,
        momentum: float = 0.95,
        clip_value: float = 20.0,
        epsilon: float = 1e-7,
        scale_factor: float = 1.0,
        weight_decay: float = 0.01,
        use_nesterov: bool = True,
        split_qkv: bool = False,
        is_qkv_fn: Optional[Callable[[torch.Tensor], bool]] = None,
        qkv_split_shapes: Optional[tuple] = None,
        svd_log_interval: int = 10,
        svd_log_dir: str = "",
    ) -> None:
        defaults = dict(
            lr=lr,
            momentum=momentum,
            clip_value=clip_value,
            epsilon=epsilon,
            scale_factor=scale_factor,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

        self.use_nesterov = use_nesterov
        self.split_qkv = split_qkv
        self.is_qkv_fn = is_qkv_fn
        self.qkv_split_shapes = qkv_split_shapes
        self.svd_log_interval = svd_log_interval
        self.svd_log_dir = svd_log_dir
        self._step_count = 0

        if self.svd_log_dir:
            os.makedirs(self.svd_log_dir, exist_ok=True)

    def _roo_normal_transform_single(
        self,
        M: torch.Tensor,
        clip_value: float,
        epsilon: float,
        scale_factor: float,
        return_sv: bool = False,
    ):
        """Apply SVD-based spectral clipping to a single 2D tensor.

        Steps:
            1. SVD: U, S, Vh = svd(M)
            2. Inverse: S_inv = 1 / (S + ε)
            3. Clip: S_clipped = clamp(S_inv, max=clip_value)
            4. Reconstruct: Φ = (U * S_clipped) @ Vh
            5. RMS normalize and scale

        Returns:
            phi or (phi, S) depending on return_sv.
        """
        orig_dtype = M.dtype
        M_f32 = M.float()

        U, S, Vh = torch.linalg.svd(M_f32, full_matrices=False)

        S_inv = 1.0 / (S + epsilon)
        S_clipped = torch.clamp(S_inv, max=clip_value)

        # Reconstruct: (U * S_clipped[None, :]) @ Vh avoids constructing diagonal matrix
        phi = (U * S_clipped.unsqueeze(0)) @ Vh

        phi = phi.to(orig_dtype)

        # RMS normalization + scale
        rms = phi.norm() / (phi.numel() ** 0.5)
        if rms > 0:
            phi = phi / rms * scale_factor

        if return_sv:
            return phi, S
        return phi

    def _roo_normal_transform_qkv(
        self,
        M: torch.Tensor,
        clip_value: float,
        epsilon: float,
        scale_factor: float,
        return_sv: bool = False,
    ):
        """Apply RooNormal transform to QKV parameter by splitting into Q, K, V components."""
        grad_shape = M.shape
        num_query_groups = grad_shape[0] // sum(self.qkv_split_shapes)
        qkv_parts = torch.split(
            M.view(num_query_groups, sum(self.qkv_split_shapes), -1),
            self.qkv_split_shapes,
            dim=1,
        )
        qkv_parts = [g.reshape(-1, grad_shape[-1]) for g in qkv_parts]

        all_sv = []
        qkv_transformed = []
        for g in qkv_parts:
            result = self._roo_normal_transform_single(
                g, clip_value, epsilon, scale_factor, return_sv=return_sv
            )
            if return_sv:
                phi, sv = result
                all_sv.append(sv)
            else:
                phi = result
            qkv_transformed.append(
                phi.view(num_query_groups, -1, grad_shape[-1])
            )

        phi_out = torch.cat(qkv_transformed, dim=1).view(grad_shape)
        if return_sv:
            return phi_out, all_sv
        return phi_out

    @staticmethod
    def _collect_svd_stats(S: torch.Tensor, param_shape: tuple, clip_value: float) -> dict:
        """Collect summary statistics from singular values."""
        S_inv = 1.0 / (S + 1e-7)
        return {
            'shape': list(param_shape),
            'num_sv': S.numel(),
            'sigma_max': S.max().item(),
            'sigma_min': S.min().item(),
            'sigma_mean': S.mean().item(),
            'sigma_median': S.median().item(),
            'num_clipped': (S_inv >= clip_value).sum().item(),
            'sigma_top5': S[:5].tolist(),
        }

    def _write_svd_log(self, step: int, stats_list: list):
        """Write SVD statistics to logger and optionally to file."""
        for i, stats in enumerate(stats_list):
            log_single_rank(
                logger, logging.INFO,
                f"[SVD step={step}] param {i} shape={stats['shape']} "
                f"sigma_max={stats['sigma_max']:.4f} sigma_min={stats['sigma_min']:.6f} "
                f"sigma_mean={stats['sigma_mean']:.4f} "
                f"num_clipped={stats['num_clipped']}/{stats['num_sv']} "
                f"top5={[f'{v:.4f}' for v in stats['sigma_top5']]}"
            )

        if self.svd_log_dir:
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            if rank == 0:
                log_path = os.path.join(self.svd_log_dir, 'svd_log_rank0.jsonl')
                with open(log_path, 'a') as f:
                    for stats in stats_list:
                        record = {'step': step, **stats}
                        f.write(json.dumps(record) + '\n')

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single RooNormal optimization step.

        Uses Muon-style EMA momentum: buf = β*buf + (1-β)*grad
        With optional Nesterov: grad_for_transform = grad.lerp(buf, β)
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._step_count += 1
        should_log = (
            self.svd_log_interval > 0
            and self._step_count % self.svd_log_interval == 0
        )
        svd_stats = []

        for group in self.param_groups:
            momentum_beta = group['momentum']
            clip_value = group['clip_value']
            epsilon = group['epsilon']
            scale_factor = group['scale_factor']
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

                # EMA update (Muon-style): buf = β*buf + (1-β)*grad
                buf.lerp_(grad, 1 - momentum_beta)

                # Nesterov momentum (ref: OrthogonalizedOptimizer L189-L192)
                if self.use_nesterov:
                    update = grad.lerp(buf, momentum_beta)
                else:
                    update = buf

                # Non-2D params: direct update (no SVD)
                if grad.ndim != 2:
                    if weight_decay != 0:
                        p.data.mul_(1 - lr * weight_decay)
                    p.data.add_(update, alpha=-lr)
                    continue

                # 2D params: apply SVD transform
                if self.split_qkv and self.is_qkv_fn is not None and self.is_qkv_fn(p):
                    result = self._roo_normal_transform_qkv(
                        update, clip_value, epsilon, scale_factor, return_sv=should_log
                    )
                    if should_log:
                        phi, sv_list = result
                        for sv in sv_list:
                            svd_stats.append(
                                self._collect_svd_stats(sv, p.shape, clip_value)
                            )
                    else:
                        phi = result
                else:
                    result = self._roo_normal_transform_single(
                        update, clip_value, epsilon, scale_factor, return_sv=should_log
                    )
                    if should_log:
                        phi, sv = result
                        svd_stats.append(
                            self._collect_svd_stats(sv, p.shape, clip_value)
                        )
                    else:
                        phi = result

                # Decoupled weight decay + update
                if weight_decay != 0:
                    p.data.mul_(1 - lr * weight_decay)
                p.data.add_(phi, alpha=-lr)

        if should_log and svd_stats:
            self._write_svd_log(self._step_count, svd_stats)

        return loss


def get_megatron_roo_normal_optimizer(
    config: OptimizerConfig,
    model_chunks: List[MegatronModule],
    config_overrides: Optional[Dict[ParamKey, OptimizerConfig]] = None,
    use_gloo_process_groups: bool = True,
) -> MegatronOptimizer:
    """Create RooNormal optimizer for Megatron model chunks.

    Follows the same pattern as get_megatron_roo_optimizer():
    1. Split params into linear (2D, non-embedding) and nonlinear
    2. Freeze nonlinear -> create RooNormal for linear -> wrap bf16
    3. Freeze linear -> create Adam for nonlinear
    4. Unfreeze all -> return ChainedOptimizer([roo_normal_wrapped, adam_wrapped])
    """
    # RooNormal uses adam config for the nonlinear params path
    config.optimizer = 'adam'

    if config.use_distributed_optimizer:
        raise Exception('RooNormal with distributed optimizer is not supported.')

    log_single_rank(logger, logging.INFO, f'Setting up RooNormal optimizer with config {config}')

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
            if 'linear_qkv.weight' in name and len(param.shape) == 2:
                param.is_qkv = True
            if (
                not getattr(param, 'is_embedding_or_output_parameter', False)
                and len(param.shape) == 2
            ):
                linear_params.append(param)
            else:
                nonlinear_params.append(param)

    # === RooNormal optimizer for linear params ===
    for param in nonlinear_params:
        param.requires_grad = False

    linear_param_groups = _get_param_groups(model_chunks, config, config_overrides)

    roo_normal_optimizer = RooNormal(
        linear_param_groups,
        lr=config.lr,
        momentum=config.roo_normal_momentum,
        clip_value=config.roo_normal_clip_value,
        epsilon=config.roo_normal_epsilon,
        scale_factor=config.roo_normal_scale_factor,
        weight_decay=config.weight_decay,
        use_nesterov=config.roo_normal_use_nesterov,
        split_qkv=config.roo_normal_split_qkv,
        is_qkv_fn=lambda p: getattr(p, 'is_qkv', False),
        qkv_split_shapes=qkv_split_shapes,
        svd_log_interval=config.roo_normal_svd_log_interval,
        svd_log_dir=config.roo_normal_svd_log_dir,
    )

    def roo_normal_init_state_fn(opt, config=None):
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

    if config.fp16:
        raise Exception('RooNormal with fp16 is not supported.')

    if config.bf16:
        roo_normal_wrapped = Float16OptimizerWithFloat16Params(
            roo_normal_optimizer, config, None, roo_normal_init_state_fn
        )
    else:
        roo_normal_wrapped = FP32Optimizer(
            roo_normal_optimizer, config, roo_normal_init_state_fn
        )

    optimizers.append(roo_normal_wrapped)

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
