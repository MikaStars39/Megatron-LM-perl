# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Shampoo optimizer (Gupta et al. 2018, https://arxiv.org/abs/1802.09568).

True Shampoo: update = L^(-1/4) @ momentum @ R^(-1/4), where L = EMA(G G^T)
and R = EMA(G^T G) are Kronecker factors of the gradient outer products and
inverse 4th roots are computed via eigendecomposition.

Megatron integration wrapper splits parameters into:
  - Linear (2D, non-embedding) -> Shampoo
  - Nonlinear (embedding, LayerNorm, bias) -> Adam

TP correctness: under tensor parallelism, one of L, R is a partial sum across
the TP-split dim and is all-reduced; the other is a local diagonal block of
the global factor (block-diagonal Shampoo approximation).
"""

import logging
from itertools import chain
from typing import Dict, List, Optional

import torch

from megatron.core.dist_checkpointing.dict_utils import nested_values
from megatron.core.dist_checkpointing.mapping import (
    ShardedStateDict,
    ShardedTensor as MappingShardedTensor,
    ShardedTensorFactory,
)
from megatron.core.dist_checkpointing.optimizer import (
    get_param_id_to_sharded_param_map,
    make_sharded_optimizer_tensor,
    optim_state_to_sharding_state,
)
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
from .soap import _make_precond_sharded_tensor

try:
    from emerging_optimizers.utils.eig import eigh_with_fallback

    HAVE_SHAMPOO_DEPS = True
except ImportError:
    HAVE_SHAMPOO_DEPS = False

logger = logging.getLogger(__name__)

# Shampoo state keys whose shapes differ from the model parameter.
_SHAMPOO_PRECOND_KEYS = ('L', 'R', 'inv_root_L', 'inv_root_R')


# =====================================================================
# Core algorithm
# =====================================================================

@torch.no_grad()
def _update_kron_factors_tp(p, L, R, grad, shampoo_beta, pg_collection):
    """EMA-update L,R with TP all-reduce on the partial-sum factor.

    L_local = G_local @ G_local.T;  R_local = G_local.T @ G_local
    partition_dim == 0    -> all-reduce R (sum over the split first dim)
    partition_dim == 1    -> all-reduce L (sum over the split second dim)
    partition_dim is None -> no comm (replicated weight)
    """
    outer_L = grad @ grad.T
    outer_R = grad.T @ grad

    tp_group = None
    if pg_collection is not None:
        tp_group = (
            pg_collection.expt_tp
            if getattr(p, 'expert_tp', False)
            else pg_collection.tp
        )
    pdim = getattr(p, 'partition_dim', None)
    if pdim == -1:
        pdim = None

    if tp_group is not None and torch.distributed.get_world_size(tp_group) > 1:
        if pdim == 0:
            torch.distributed.all_reduce(outer_R, group=tp_group)
        elif pdim == 1:
            torch.distributed.all_reduce(outer_L, group=tp_group)

    L.lerp_(outer_L, 1 - shampoo_beta)
    R.lerp_(outer_R, 1 - shampoo_beta)


@torch.no_grad()
def _inv_pth_root(M, p_root, eps):
    """M^(-1/p) for symmetric PSD M, via eigh with fp64 fallback."""
    eigvals, eigvecs = eigh_with_fallback(M)
    inv = eigvals.clamp_min(eps).pow(-1.0 / p_root)
    # Q * diag(inv) @ Q.T  ==  (Q * inv[None, :]) @ Q.T
    return (eigvecs * inv) @ eigvecs.T


class Shampoo(torch.optim.Optimizer):
    """True Shampoo for 2D parameters.

    Args:
        params: parameters or param groups.
        lr: learning rate.
        momentum: heavy-ball momentum on the gradient EMA.
        shampoo_beta: EMA beta for L, R Kronecker factors.
        eps: eigenvalue floor before inverse 4th root.
        weight_decay: decoupled weight decay coefficient.
        precondition_frequency: steps between inverse-root refreshes (also runs at step 1).
        max_update_rms: clip update RMS to this value (0 = off).
        correct_bias: Adam-style bias correction on the momentum (off by default; classical
            Anil-style Shampoo).
        correct_factor_bias: EMA bias correction on L,R before forming inverse roots
            (distributed-Shampoo standard; on by default).
        split_qkv: v2 placeholder; raises if set.
        pg_collection: process-group collection for TP all-reduce. Pass None to disable
            (single-rank or non-Megatron use).
    """

    def __init__(
        self,
        params,
        lr,
        momentum: float = 0.9,
        shampoo_beta: float = 0.95,
        eps: float = 1e-12,
        weight_decay: float = 0.0,
        precondition_frequency: int = 20,
        max_update_rms: float = 0.0,
        correct_bias: bool = False,
        correct_factor_bias: bool = True,
        split_qkv: bool = False,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):
        if split_qkv:
            raise NotImplementedError(
                "split_qkv (per-Q/K/V Kronecker factors for fused QKV columns) is not "
                "implemented in v1; the block-diagonal approximation couples Q/K/V "
                "correlations into one L. Tracked as TODO."
            )
        if not HAVE_SHAMPOO_DEPS:
            raise ImportError(
                "Shampoo requires 'emerging_optimizers' for eigh_with_fallback. "
                "Install from: https://github.com/NVIDIA-NeMo/Emerging-Optimizers"
            )
        defaults = dict(
            lr=lr,
            momentum=momentum,
            shampoo_beta=shampoo_beta,
            eps=eps,
            weight_decay=weight_decay,
            precondition_frequency=precondition_frequency,
            max_update_rms=max_update_rms,
            correct_bias=correct_bias,
            correct_factor_bias=correct_factor_bias,
        )
        super().__init__(params, defaults)
        self.pg_collection = pg_collection

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.dim() != 2:
                    raise TypeError("Shampoo only supports 2D tensors")

                grad = p.grad.to(torch.float32)
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
                    state["L"] = torch.zeros(
                        p.shape[0], p.shape[0], dtype=torch.float32, device=p.device
                    )
                    state["R"] = torch.zeros(
                        p.shape[1], p.shape[1], dtype=torch.float32, device=p.device
                    )
                    state["inv_root_L"] = torch.eye(
                        p.shape[0], dtype=torch.float32, device=p.device
                    )
                    state["inv_root_R"] = torch.eye(
                        p.shape[1], dtype=torch.float32, device=p.device
                    )

                step1 = state["step"] + 1

                # 1. EMA-update L,R with TP-aware all-reduce.
                _update_kron_factors_tp(
                    p,
                    state["L"],
                    state["R"],
                    grad,
                    group["shampoo_beta"],
                    self.pg_collection,
                )

                # 2. Heavy-ball momentum on the gradient.
                state["exp_avg"].lerp_(grad, 1 - group["momentum"])
                m = state["exp_avg"]
                if group["correct_bias"]:
                    m = m / (1 - group["momentum"] ** step1)

                # 3. Refresh inverse 4th roots periodically (also at step 1).
                if (step1 - 1) % group["precondition_frequency"] == 0:
                    if group["correct_factor_bias"]:
                        denom = 1.0 - group["shampoo_beta"] ** step1
                        L_hat = state["L"] / denom
                        R_hat = state["R"] / denom
                    else:
                        L_hat, R_hat = state["L"], state["R"]
                    state["inv_root_L"] = _inv_pth_root(L_hat, 4, group["eps"])
                    state["inv_root_R"] = _inv_pth_root(R_hat, 4, group["eps"])

                # 4. Compute update -- precondition from step 1.
                update = state["inv_root_L"] @ m @ state["inv_root_R"]

                # 5. RMS clip.
                if group["max_update_rms"] > 0:
                    rms = update.square().mean().sqrt()
                    scale = (group["max_update_rms"] / (rms + 1e-7)).clamp(max=1.0)
                    update = update * scale

                # 6. Decoupled weight decay.
                if group["weight_decay"] > 0:
                    p.mul_(1 - group["lr"] * group["weight_decay"])

                # 7. Apply.
                p.add_(update, alpha=-group["lr"])
                state["step"] = step1

        return loss


# =====================================================================
# Megatron Float16 / FP32 wrappers (sharded checkpoint support)
# =====================================================================

class ShampooFloat16Optimizer(Float16OptimizerWithFloat16Params):
    """Float16 wrapper that handles Shampoo's irregularly-shaped state.

    Mirrors SoapFloat16Optimizer; only the precond-key set differs.
    """

    def sharded_state_dict(
        self,
        model_sharded_state_dict: ShardedStateDict,
        is_loading: bool = False,
        metadata: Optional[dict] = None,
    ):
        if is_loading:
            self.init_state_fn(self.optimizer, self.config)

        state_dict = self.state_dict()

        id_to_sharded_param_map = get_param_id_to_sharded_param_map(
            model_sharded_state_dict, chain.from_iterable(g for g in self.float16_groups)
        )

        assert len(state_dict['fp32_from_fp16_params']) == len(
            state_dict['optimizer']['param_groups']
        )
        state_dict['fp32_from_fp16_params'] = [
            [
                make_sharded_optimizer_tensor(
                    id_to_sharded_param_map[param_id],
                    fp32_param,
                    prefix='optimizer.state.fp32_param',
                )
                for param_id, fp32_param in zip(state_group['params'], fp32_group)
            ]
            for fp32_group, state_group in zip(
                state_dict['fp32_from_fp16_params'], state_dict['optimizer']['param_groups']
            )
        ]

        step = self._extract_common_per_param_step(state_dict['optimizer'])

        # Split precond tensors from Adam-compatible state without mutating live state.
        precond_states = {}
        adam_state = {}
        for param_id, param_state in state_dict['optimizer']['state'].items():
            precond = {}
            adam = {}
            for k, v in param_state.items():
                if k in _SHAMPOO_PRECOND_KEYS:
                    precond[k] = v
                else:
                    adam[k] = v
            precond_states[param_id] = precond
            adam_state[param_id] = adam
        state_dict['optimizer']['state'] = adam_state

        optim_state_to_sharding_state(
            state_dict['optimizer'], id_to_sharded_param_map, exclude_keys="step"
        )

        for param_id, precond in precond_states.items():
            if param_id not in id_to_sharded_param_map:
                for key, tensor in precond.items():
                    state_dict['optimizer']['state'][param_id][key] = tensor
                continue
            model_sh_param = id_to_sharded_param_map[param_id]
            for precond_key, precond_tensor in precond.items():
                state_dict['optimizer']['state'][param_id][precond_key] = (
                    _make_precond_sharded_tensor(
                        precond_key, precond_tensor, model_sh_param
                    )
                )

        if step:
            state_dict['optimizer']['state']['common_step'] = step
        return state_dict


class ShampooFP32Optimizer(FP32Optimizer):
    """FP32 wrapper analog of ShampooFloat16Optimizer."""

    def sharded_state_dict(
        self,
        model_sharded_state_dict: ShardedStateDict,
        is_loading: bool = False,
        metadata: Optional[dict] = None,
    ):
        if is_loading:
            self.init_state_fn(self.optimizer, self.config)

        state_dict = self.state_dict()
        id_to_sharded_param_map = get_param_id_to_sharded_param_map(
            model_sharded_state_dict, self.get_parameters()
        )
        step = self._extract_common_per_param_step(state_dict)

        precond_states = {}
        adam_state = {}
        for param_id, param_state in state_dict['state'].items():
            precond = {}
            adam = {}
            for k, v in param_state.items():
                if k in _SHAMPOO_PRECOND_KEYS:
                    precond[k] = v
                else:
                    adam[k] = v
            precond_states[param_id] = precond
            adam_state[param_id] = adam
        state_dict['state'] = adam_state

        optim_state_to_sharding_state(
            state_dict, id_to_sharded_param_map, exclude_keys="step"
        )

        for param_id, precond in precond_states.items():
            if param_id not in id_to_sharded_param_map:
                for key, tensor in precond.items():
                    state_dict['state'][param_id][key] = tensor
                continue
            model_sh_param = id_to_sharded_param_map[param_id]
            for precond_key, precond_tensor in precond.items():
                state_dict['state'][param_id][precond_key] = (
                    _make_precond_sharded_tensor(
                        precond_key, precond_tensor, model_sh_param
                    )
                )

        if step:
            state_dict['state']['common_step'] = step
        return state_dict


# =====================================================================
# Megatron entrypoint
# =====================================================================

def get_megatron_shampoo_optimizer(
    config: OptimizerConfig,
    model_chunks: List[MegatronModule],
    config_overrides: Optional[Dict[ParamKey, OptimizerConfig]] = None,
    use_gloo_process_groups: bool = True,
    pg_collection: Optional[ProcessGroupCollection] = None,
) -> MegatronOptimizer:
    """Create a Shampoo optimizer for Megatron model chunks.

    1. Split params into linear (2D, non-embedding) and nonlinear.
    2. Freeze nonlinear -> create Shampoo for linear -> wrap bf16/fp32.
    3. Freeze linear -> create Adam for nonlinear (chained).
    4. Unfreeze all -> return ChainedOptimizer.
    """
    assert HAVE_SHAMPOO_DEPS, (
        "Shampoo requires 'emerging_optimizers' (for eigh_with_fallback). "
        "Install from: https://github.com/NVIDIA-NeMo/Emerging-Optimizers"
    )

    if config.use_distributed_optimizer:
        raise Exception('Shampoo with distributed optimizer is not supported.')
    if config.fp16:
        raise Exception('Shampoo with fp16 is not supported (use bf16 or fp32).')
    if config.optimizer_cpu_offload:
        raise Exception('Shampoo with CPU offload is not supported.')
    if getattr(config, 'use_precision_aware_optimizer', False):
        raise Exception('Shampoo with precision-aware optimizer is not supported.')

    if pg_collection is None:
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()

    log_single_rank(logger, logging.INFO, f'Setting up Shampoo optimizer with config {config}')

    optimizers = []
    linear_params = []
    nonlinear_params = []

    for model_chunk in model_chunks:
        for name, param in model_chunk.named_parameters():
            if not param.requires_grad:
                continue
            if 'experts' in name and 'shared' not in name:
                param.expert_tp = True
            if (
                not getattr(param, 'is_embedding_or_output_parameter', False)
                and len(param.shape) == 2
            ):
                linear_params.append(param)
            else:
                nonlinear_params.append(param)

    # === Shampoo optimizer for linear params ===
    for param in nonlinear_params:
        param.requires_grad = False

    linear_param_groups = _get_param_groups(model_chunks, config, config_overrides)

    shampoo_optimizer = Shampoo(
        linear_param_groups,
        lr=config.lr,
        momentum=config.shampoo_momentum,
        shampoo_beta=config.shampoo_beta,
        eps=config.shampoo_eps,
        weight_decay=config.weight_decay,
        precondition_frequency=config.shampoo_precondition_frequency,
        max_update_rms=config.shampoo_max_update_rms,
        correct_bias=config.shampoo_correct_bias,
        correct_factor_bias=config.shampoo_correct_factor_bias,
        split_qkv=config.shampoo_split_qkv,
        pg_collection=pg_collection,
    )

    def shampoo_init_state_fn(opt, config=None):
        # Initialize all Shampoo state keys so checkpoint load can map them back.
        for group in opt.param_groups:
            for p in group['params']:
                if len(opt.state[p]) == 0:
                    opt.state[p]['step'] = 0
                    opt.state[p]['exp_avg'] = torch.zeros_like(p.data, dtype=torch.float32)
                    opt.state[p]['L'] = torch.zeros(
                        p.shape[0], p.shape[0], dtype=torch.float32, device=p.device
                    )
                    opt.state[p]['R'] = torch.zeros(
                        p.shape[1], p.shape[1], dtype=torch.float32, device=p.device
                    )
                    opt.state[p]['inv_root_L'] = torch.eye(
                        p.shape[0], dtype=torch.float32, device=p.device
                    )
                    opt.state[p]['inv_root_R'] = torch.eye(
                        p.shape[1], dtype=torch.float32, device=p.device
                    )

    if config.bf16:
        shampoo_wrapped = ShampooFloat16Optimizer(
            shampoo_optimizer, config, None, shampoo_init_state_fn
        )
    else:
        shampoo_wrapped = ShampooFP32Optimizer(
            shampoo_optimizer, config, shampoo_init_state_fn
        )

    optimizers.append(shampoo_wrapped)

    # === Adam optimizer for nonlinear params ===
    for param in nonlinear_params:
        param.requires_grad = True
    for param in linear_params:
        param.requires_grad = False

    # Temporarily set optimizer to 'adam' so get_megatron_optimizer creates an Adam optimizer,
    # then restore to avoid permanent config mutation.
    orig_optimizer = config.optimizer
    config.optimizer = 'adam'
    chained_adam = get_megatron_optimizer(
        config,
        model_chunks,
        config_overrides=config_overrides,
        use_gloo_process_groups=use_gloo_process_groups,
    )
    config.optimizer = orig_optimizer

    for param in linear_params:
        param.requires_grad = True

    optimizers += chained_adam.chained_optimizers
    return ChainedOptimizer(optimizers)
