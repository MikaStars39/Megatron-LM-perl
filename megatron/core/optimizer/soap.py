# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""SOAP optimizer: ShampoO with Adam in the Preconditioner eigenbasis.

Megatron integration wrapper that splits parameters into:
  - Linear (2D, non-embedding) -> SOAP
  - Nonlinear (embedding, LayerNorm, bias) -> Adam

Uses the SOAP implementation from emerging_optimizers.
"""

import logging
from itertools import chain
from typing import Dict, List, Optional

import torch

from megatron.core import parallel_state
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

try:
    from emerging_optimizers.soap import SOAP

    HAVE_SOAP = True
except ImportError:
    HAVE_SOAP = False

logger = logging.getLogger(__name__)

# SOAP preconditioner state keys whose shapes differ from the model parameter.
_SOAP_PRECOND_KEYS = ('L', 'R', 'Q_L', 'Q_R')


def _resolve_factory_to_sharded_tensor(factory):
    """Return the inner ShardedTensor representing the weight itself.

    Megatron registers some 2D weights as ShardedTensorFactory (TE-fused
    layers, certain GQA/MoE paths).  The factory's .build() returns a
    sub-state-dict that contains the real ShardedTensor(s).  We pick the
    one whose local data matches the factory's original parameter so that
    prepend_axis_num / global_offset / axis_fragmentations describe the
    same layer-level positioning as the weight.

    Returns None if no matching ShardedTensor can be found.
    """
    try:
        built = factory.build()
    except Exception:
        return None
    factory_data = factory.data
    candidates = [v for v in nested_values(built) if isinstance(v, MappingShardedTensor)]
    if not candidates:
        return None
    # Prefer the ShardedTensor that wraps the same underlying storage as the factory.
    for st in candidates:
        if st.data is factory_data:
            return st
    # Fall back to the one with matching shape.
    for st in candidates:
        if tuple(st.data.shape) == tuple(factory_data.shape):
            return st
    return candidates[0]


def _make_precond_sharded_tensor(precond_key, precond_data, model_sharded_param):
    """Build a ShardedTensor for a SOAP preconditioner matrix.

    All preconditioners are saved with 1D stacking across TP ranks along axis 0.
    Each TP rank's [dim, dim] preconditioner is stacked into a virtual
    [dim*tp, dim] global tensor.  This ensures every chunk of the global tensor
    is covered (required by validate_sharding_integrity).

    When the model parameter uses prepend_axis_num (e.g. for layer indices in PP),
    those prepended axes are replicated so that each layer's preconditioner gets
    a unique position in the global checkpoint.

    Each TP rank's preconditioner is computed from local gradients and has
    unique values, so none can be treated as replicated.

    L/R use allow_shape_mismatch=True because zero-init is correct on
    shape mismatch (matches init_kronecker_factors).  Q_L/Q_R use False
    because they require identity init; on TP reshard the user should
    use --no-load-optim to skip optimizer state.

    Args:
        precond_key: one of 'L', 'R', 'Q_L', 'Q_R'
        precond_data: the local preconditioner tensor [dim, dim]
        model_sharded_param: ShardedTensor or ShardedTensorFactory from model
    """
    tp_group = parallel_state.get_tensor_model_parallel_group()
    dp_cp_group = parallel_state.get_data_parallel_group(with_context_parallel=True)
    tp_rank = torch.distributed.get_rank(tp_group)
    tp_size = torch.distributed.get_world_size(tp_group)
    dp_rank = torch.distributed.get_rank(dp_cp_group)
    allow_mismatch = precond_key in ('L', 'R')

    if isinstance(model_sharded_param, ShardedTensorFactory):
        resolved = _resolve_factory_to_sharded_tensor(model_sharded_param)
        if resolved is not None:
            model_sharded_param = resolved
        else:
            # No inner ShardedTensor recoverable.  Rely on the factory's key
            # being unique per-layer (PP stages naturally carry distinct keys)
            # and shard along TP so each TP rank's preconditioner occupies a
            # distinct slice of the virtual global tensor.
            key = f'optimizer.state.{precond_key}.{model_sharded_param.key}'
            if tp_size == 1:
                return MappingShardedTensor.from_rank_offsets(
                    key,
                    precond_data,
                    replica_id=(0, 0, dp_rank),
                    allow_shape_mismatch=allow_mismatch,
                )
            return MappingShardedTensor.from_rank_offsets(
                key,
                precond_data,
                (0, tp_rank, tp_size),
                replica_id=(0, 0, dp_rank),
                allow_shape_mismatch=allow_mismatch,
            )

    key = f'optimizer.state.{precond_key}.{model_sharded_param.key}'
    prepend_axis_num = model_sharded_param.prepend_axis_num

    # Build rank_offsets for prepended axes (e.g. layer index under PP).
    # These axes have local_axis_shape=1 in from_rank_offsets, so
    # (axis, offset, fragm) maps to global_shape[axis]=fragm,
    # global_offset[axis]=offset.
    prepend_offsets = []
    for axis in range(prepend_axis_num):
        offset = model_sharded_param.global_offset[axis]
        fragm = model_sharded_param.axis_fragmentations[axis]
        prepend_offsets.append((axis, offset, fragm))

    # TP axis index is shifted by prepend_axis_num.
    tp_axis = prepend_axis_num  # axis 0 of the actual [dim, dim] data

    if tp_size == 1:
        return MappingShardedTensor.from_rank_offsets(
            key,
            precond_data,
            *prepend_offsets,
            prepend_axis_num=prepend_axis_num,
            replica_id=(0, 0, dp_rank),
            allow_shape_mismatch=allow_mismatch,
        )

    # TP > 1: stack each rank's [dim, dim] along axis 0 (shifted by prepend)
    # into a [dim*tp, dim] global tensor.  Each rank owns one [dim, dim] slice.
    return MappingShardedTensor.from_rank_offsets(
        key,
        precond_data,
        *prepend_offsets,
        (tp_axis, tp_rank, tp_size),
        prepend_axis_num=prepend_axis_num,
        replica_id=(0, 0, dp_rank),
        allow_shape_mismatch=allow_mismatch,
    )


class SoapFloat16Optimizer(Float16OptimizerWithFloat16Params):
    """Float16 wrapper that handles SOAP preconditioner checkpoint correctly.

    SOAP's state contains Kronecker-factor matrices (L, R, Q_L, Q_R) whose
    shapes are square and do NOT match the model parameter shape.  The default
    ``sharded_state_dict`` assumes every optimizer-state tensor has the same
    shape as the parameter, which triggers an assertion error.

    This subclass excludes those keys from the normal sharded path and saves
    them as properly sharded tensors with block-diagonal TP sharding."""

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

        # Convert fp32_from_fp16_params (same as parent)
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

        # --- SOAP-specific: split precond tensors from Adam state ---
        # NOTE: torch.optim.Optimizer.state_dict() returns shallow references to
        # self.state[param], so pop() would mutate the live optimizer state and
        # break the next optimizer.step(). Build fresh dicts instead.
        precond_states = {}
        adam_state = {}
        for param_id, param_state in state_dict['optimizer']['state'].items():
            precond = {}
            adam = {}
            for k, v in param_state.items():
                if k in _SOAP_PRECOND_KEYS:
                    precond[k] = v
                else:
                    adam[k] = v
            precond_states[param_id] = precond
            adam_state[param_id] = adam
        state_dict['optimizer']['state'] = adam_state

        # Shard the remaining Adam-compatible state (exp_avg, exp_avg_sq)
        optim_state_to_sharding_state(
            state_dict['optimizer'], id_to_sharded_param_map, exclude_keys="step"
        )

        # Put preconditioner tensors back as ShardedTensors with per-rank
        # block-diagonal sharding so each TP/PP rank saves its own copy.
        for param_id, precond in precond_states.items():
            if param_id not in id_to_sharded_param_map:
                # Fallback: save as plain tensor (shouldn't happen in normal flow)
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


class SoapFP32Optimizer(FP32Optimizer):
    """FP32 wrapper that handles SOAP preconditioner checkpoint correctly.

    Same as SoapFloat16Optimizer but adapted for FP32Optimizer's state dict
    layout where state lives at state_dict['state'] instead of
    state_dict['optimizer']['state']."""

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

        # --- SOAP-specific: split precond tensors from Adam state ---
        # Avoid mutating live optimizer state (see SoapFloat16Optimizer).
        precond_states = {}
        adam_state = {}
        for param_id, param_state in state_dict['state'].items():
            precond = {}
            adam = {}
            for k, v in param_state.items():
                if k in _SOAP_PRECOND_KEYS:
                    precond[k] = v
                else:
                    adam[k] = v
            precond_states[param_id] = precond
            adam_state[param_id] = adam
        state_dict['state'] = adam_state

        # Shard the remaining Adam-compatible state (exp_avg, exp_avg_sq)
        optim_state_to_sharding_state(
            state_dict, id_to_sharded_param_map, exclude_keys="step"
        )

        # Put preconditioner tensors back as ShardedTensors with per-rank
        # block-diagonal sharding so each TP/PP rank saves its own copy.
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


def get_megatron_soap_optimizer(
    config: OptimizerConfig,
    model_chunks: List[MegatronModule],
    config_overrides: Optional[Dict[ParamKey, OptimizerConfig]] = None,
    use_gloo_process_groups: bool = True,
    pg_collection: Optional[ProcessGroupCollection] = None,
) -> MegatronOptimizer:
    """Create SOAP optimizer for Megatron model chunks.

    1. Split params into linear (2D, non-embedding) and nonlinear
    2. Freeze nonlinear -> create SOAP for linear -> wrap bf16
    3. Freeze linear -> create Adam for nonlinear
    4. Unfreeze all -> return ChainedOptimizer
    """
    assert HAVE_SOAP, (
        "SOAP optimizer requires 'emerging_optimizers' package. "
        "Install from: https://github.com/NVIDIA-NeMo/Emerging-Optimizers.git@v0.1.0"
    )

    if config.use_distributed_optimizer:
        raise Exception('SOAP with distributed optimizer is not supported.')

    if pg_collection is None:
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()

    log_single_rank(logger, logging.INFO, f'Setting up SOAP optimizer with config {config}')

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

    # === SOAP optimizer for linear params ===
    for param in nonlinear_params:
        param.requires_grad = False

    linear_param_groups = _get_param_groups(model_chunks, config, config_overrides)

    soap_optimizer = SOAP(
        linear_param_groups,
        lr=config.lr,
        betas=(config.soap_beta1, config.soap_beta2),
        shampoo_beta=config.soap_shampoo_beta,
        eps=config.soap_eps,
        weight_decay=config.weight_decay,
        precondition_frequency=config.soap_precondition_frequency,
        adam_warmup_steps=config.soap_adam_warmup_steps,
        correct_bias=config.soap_correct_bias,
        fp32_matmul_prec=config.soap_fp32_matmul_prec,
        use_eigh=config.soap_use_eigh,
        power_iter_steps=config.soap_power_iter_steps,
        max_update_rms=config.soap_max_update_rms,
        nesterov=config.soap_nesterov,
    )

    def soap_init_state_fn(opt, config=None):
        # Initialize all SOAP state keys so checkpoint load can map them back.
        for group in opt.param_groups:
            for p in group['params']:
                if len(opt.state[p]) == 0:
                    opt.state[p]['step'] = 0
                    opt.state[p]['exp_avg'] = torch.zeros_like(p.data, dtype=torch.float32)
                    opt.state[p]['exp_avg_sq'] = torch.zeros_like(p.data, dtype=torch.float32)
                    # Preconditioner matrices (square, shape != param shape)
                    opt.state[p]['L'] = torch.zeros(p.shape[0], p.shape[0], device=p.device)
                    opt.state[p]['R'] = torch.zeros(p.shape[1], p.shape[1], device=p.device)
                    opt.state[p]['Q_L'] = torch.eye(p.shape[0], device=p.device)
                    opt.state[p]['Q_R'] = torch.eye(p.shape[1], device=p.device)

    if config.fp16:
        raise Exception('SOAP with fp16 is not supported.')

    if config.bf16:
        soap_wrapped = SoapFloat16Optimizer(
            soap_optimizer, config, None, soap_init_state_fn
        )
    else:
        soap_wrapped = SoapFP32Optimizer(
            soap_optimizer, config, soap_init_state_fn
        )

    optimizers.append(soap_wrapped)

    # === Adam optimizer for nonlinear params ===
    for param in nonlinear_params:
        param.requires_grad = True
    for param in linear_params:
        param.requires_grad = False

    # Temporarily set optimizer to 'adam' so get_megatron_optimizer creates
    # an Adam optimizer, then restore to avoid permanent config mutation.
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
