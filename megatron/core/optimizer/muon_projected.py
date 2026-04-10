# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""MuonProjected optimizer: Muon with initial-weight projection.

Applies projection (I - alpha * W0 @ W0^T) after Muon orthogonalization:

    M_t = beta * M_{t-1} + (1-beta) * G_t          # EMA momentum (Muon-style)
    G_nesterov = lerp(G_t, M_t, beta)               # Nesterov lookahead (optional)
    Phi = NewtonSchulz(G_nesterov) * scale           # Muon orthogonalization
    Phi_proj = Phi - alpha * W0_hat @ (W0_hat^T @ Phi)       # Projection (W0_hat = W0/||W0||_F, precomputed)
    W_t = W_{t-1} - lr * (Phi_proj + lambda * W_{t-1})  # Update (decoupled WD)

W0 is the initial weight captured after checkpoint loading, fixed throughout training.
"""

import logging
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


class MuonProjected(torch.optim.Optimizer):
    """MuonProjected: Muon with initial-weight projection.

    Applies (I - alpha * W0 @ W0^T) @ Muon(G) as the parameter update.
    Only applies the Muon+projection transform to 2D parameters.

    Args:
        params: Parameters to optimize.
        lr: Learning rate.
        momentum: Momentum coefficient (beta) for EMA.
        weight_decay: Decoupled weight decay coefficient.
        use_nesterov: Whether to use Nesterov-style momentum.
        projection_alpha: Coefficient for the projection (default 0.5).
        split_qkv: Whether to split QKV parameters.
        is_qkv_fn: Function to check if a parameter is QKV.
        qkv_split_shapes: Shapes for QKV splitting.
        num_ns_steps: Number of Newton-Schulz iteration steps.
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
        projection_alpha: float = 0.5,
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

        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

        self.use_nesterov = use_nesterov
        self.projection_alpha = projection_alpha
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
            # Non-TP or blockwise: use non-TP newton_schulz directly
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

    def _apply_projection(self, muon_update: torch.Tensor, W0_hat: torch.Tensor) -> torch.Tensor:
        """Apply (I - alpha * W0_hat @ W0_hat^T) @ muon_update.

        W0_hat = W0 / ||W0||_F is precomputed in capture_initial_weights().
        Efficient form: muon_update - alpha * W0_hat @ (W0_hat^T @ muon_update)
        """
        orig_dtype = muon_update.dtype
        muon_f32 = muon_update.float()
        W0h_f32 = W0_hat.float()

        # W0_hat^T @ muon_update: [n, m] @ [m, n] = [n, n]
        intermediate = W0h_f32.t() @ muon_f32
        # W0_hat @ intermediate: [m, n] @ [n, n] = [m, n]
        projection_term = W0h_f32 @ intermediate

        result = muon_f32 - self.projection_alpha * projection_term
        return result.to(orig_dtype)

    def _transform_single(
        self,
        update: torch.Tensor,
        W0: torch.Tensor,
        tp_group: Optional[torch.distributed.ProcessGroup],
        partition_dim: Optional[int],
    ) -> torch.Tensor:
        """Apply Muon orthogonalization + projection to a single 2D tensor."""
        muon_update = self._muon_orthogonalize_single(update, tp_group, partition_dim)
        return self._apply_projection(muon_update, W0)

    def _transform_qkv(
        self,
        update: torch.Tensor,
        W0: torch.Tensor,
        tp_group: Optional[torch.distributed.ProcessGroup],
        partition_dim: Optional[int],
    ) -> torch.Tensor:
        """Apply Muon+projection to QKV parameter by splitting into Q, K, V components."""
        grad_shape = update.shape
        num_query_groups = grad_shape[0] // sum(self.qkv_split_shapes)

        # Split update
        update_parts = torch.split(
            update.view(num_query_groups, sum(self.qkv_split_shapes), -1),
            self.qkv_split_shapes,
            dim=1,
        )
        update_parts = [g.reshape(-1, grad_shape[-1]) for g in update_parts]

        # Split W0 the same way
        W0_parts = torch.split(
            W0.view(num_query_groups, sum(self.qkv_split_shapes), -1),
            self.qkv_split_shapes,
            dim=1,
        )
        W0_parts = [g.reshape(-1, grad_shape[-1]) for g in W0_parts]

        # Apply transform to each Q, K, V component independently
        qkv_transformed = [
            self._transform_single(u, w, tp_group, partition_dim).view(
                num_query_groups, -1, grad_shape[-1]
            )
            for u, w in zip(update_parts, W0_parts)
        ]
        return torch.cat(qkv_transformed, dim=1).view(grad_shape)

    def capture_initial_weights(self):
        """Capture normalized W0_hat = W0 / ||W0||_F for the projection.

        Must be called AFTER checkpoint loading so W0 reflects the pretrained weights.
        Precomputes the normalized W0 so that each step only needs:
            muon_update - alpha * W0_hat @ (W0_hat^T @ muon_update)
        """
        count = 0
        for group in self.param_groups:
            for p in group['params']:
                if p.data.ndim == 2:
                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = torch.zeros_like(p.data)
                    W0_f32 = p.data.float()
                    fnorm = W0_f32.norm()  # Frobenius norm
                    state['W0_hat'] = (W0_f32 / fnorm.clamp_min(1e-12)).to(torch.bfloat16)
                    count += 1
        log_single_rank(
            logger, logging.INFO,
            f'MuonProjected: captured W0_hat for {count} parameters (projection_alpha={self.projection_alpha})'
        )

    def state_dict(self):
        """Override to exclude W0_hat from checkpoint (re-captured after load)."""
        sd = super().state_dict()
        for state in sd['state'].values():
            state.pop('W0_hat', None)
        return sd

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single MuonProjected optimization step."""
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

                # EMA momentum (Muon-style): buf = beta*buf + (1-beta)*grad
                buf.lerp_(grad, 1 - momentum_beta)

                # Nesterov momentum
                if self.use_nesterov:
                    update = grad.lerp(buf, momentum_beta)
                else:
                    update = buf

                # Non-2D params: direct update (no Muon/projection)
                if grad.ndim != 2:
                    if weight_decay != 0:
                        p.data.mul_(1 - lr * weight_decay)
                    p.data.add_(update, alpha=-lr)
                    continue

                # Lazy W0_hat capture (fallback if capture_initial_weights wasn't called)
                if 'W0_hat' not in state:
                    W0_f32 = p.data.float()
                    fnorm = W0_f32.norm()
                    state['W0_hat'] = (W0_f32 / fnorm.clamp_min(1e-12)).to(torch.bfloat16)

                # Get TP info
                tp_group, partition_dim = self._get_tp_info(p)

                # Apply Muon orthogonalization + projection
                if self.split_qkv and self.is_qkv_fn is not None and self.is_qkv_fn(p):
                    phi = self._transform_qkv(update, state['W0_hat'], tp_group, partition_dim)
                else:
                    phi = self._transform_single(update, state['W0_hat'], tp_group, partition_dim)

                # Decoupled weight decay: W = (1 - lr*lambda) W - lr*Phi
                if weight_decay != 0:
                    p.data.mul_(1 - lr * weight_decay)
                p.data.add_(phi, alpha=-lr)

        return loss


def get_megatron_muon_projected_optimizer(
    config: OptimizerConfig,
    model_chunks: List[MegatronModule],
    config_overrides: Optional[Dict[ParamKey, OptimizerConfig]] = None,
    use_gloo_process_groups: bool = True,
    pg_collection: Optional[ProcessGroupCollection] = None,
) -> MegatronOptimizer:
    """Create MuonProjected optimizer for Megatron model chunks.

    Follows the same pattern as get_megatron_roo_optimizer():
    1. Split params into linear (2D, non-embedding) and nonlinear
    2. Freeze nonlinear -> create MuonProjected for linear -> wrap bf16
    3. Freeze linear -> create Adam for nonlinear
    4. Unfreeze all -> return ChainedOptimizer

    Args:
        config: Optimizer configuration.
        model_chunks: List of model chunks.
        config_overrides: Per-parameter config overrides.
        use_gloo_process_groups: Whether to use Gloo process groups.
        pg_collection: Process group collection for TP.

    Returns:
        ChainedOptimizer containing MuonProjected (for linear) + Adam (for nonlinear).
    """
    config.optimizer = 'adam'

    assert HAVE_EMERGING_OPTIMIZERS, (
        "MuonProjected requires 'emerging_optimizers' package. "
        "Install from: https://github.com/NVIDIA-NeMo/Emerging-Optimizers.git@v0.1.0"
    )

    if config.use_distributed_optimizer:
        raise Exception('MuonProjected with distributed optimizer is not supported.')

    if pg_collection is None:
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()

    log_single_rank(logger, logging.INFO, f'Setting up MuonProjected optimizer with config {config}')

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

    # === MuonProjected optimizer for linear params ===
    for param in nonlinear_params:
        param.requires_grad = False

    linear_param_groups = _get_param_groups(model_chunks, config, config_overrides)

    muon_proj_optimizer = MuonProjected(
        linear_param_groups,
        lr=config.lr,
        momentum=config.muon_projected_momentum,
        weight_decay=config.weight_decay,
        use_nesterov=config.muon_projected_use_nesterov,
        projection_alpha=config.muon_projected_projection_alpha,
        num_ns_steps=config.muon_projected_num_ns_steps,
        scale_mode=config.muon_projected_scale_mode,
        extra_scale_factor=config.muon_projected_extra_scale_factor,
        fp32_matmul_prec=config.muon_projected_fp32_matmul_prec,
        tp_mode=config.muon_projected_tp_mode,
        split_qkv=config.muon_projected_split_qkv,
        is_qkv_fn=lambda p: getattr(p, 'is_qkv', False),
        qkv_split_shapes=qkv_split_shapes,
        pg_collection=pg_collection,
    )

    def muon_projected_init_state_fn(opt, config=None):
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
        raise Exception('MuonProjected with fp16 is not supported.')

    if config.bf16:
        muon_proj_wrapped = Float16OptimizerWithFloat16Params(
            muon_proj_optimizer, config, None, muon_projected_init_state_fn
        )
    else:
        muon_proj_wrapped = FP32Optimizer(
            muon_proj_optimizer, config, muon_projected_init_state_fn
        )

    optimizers.append(muon_proj_wrapped)

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
