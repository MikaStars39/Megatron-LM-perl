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

from megatron.core.dist_checkpointing.mapping import ShardedStateDict
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


class SoapFloat16Optimizer(Float16OptimizerWithFloat16Params):
    """Float16 wrapper that handles SOAP preconditioner checkpoint correctly.

    SOAP's state contains Kronecker-factor matrices (L, R, Q_L, Q_R) whose
    shapes are square and do NOT match the model parameter shape.  The default
    ``sharded_state_dict`` assumes every optimizer-state tensor has the same
    shape as the parameter, which triggers an assertion error.

    This subclass excludes those keys from the normal sharded path and saves
    them as plain (non-sharded, per-rank replicated) tensors instead.
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

        # --- SOAP-specific: pull preconditioner tensors out before sharding ---
        precond_states = {}
        for param_id, param_state in state_dict['optimizer']['state'].items():
            precond_states[param_id] = {}
            for key in _SOAP_PRECOND_KEYS:
                if key in param_state:
                    precond_states[param_id][key] = param_state.pop(key)

        # Shard the remaining Adam-compatible state (exp_avg, exp_avg_sq)
        optim_state_to_sharding_state(
            state_dict['optimizer'], id_to_sharded_param_map, exclude_keys="step"
        )

        # Put preconditioner tensors back as plain (non-sharded) tensors.
        # They will be saved/loaded per-rank without sharding metadata.
        for param_id, precond in precond_states.items():
            for key, tensor in precond.items():
                state_dict['optimizer']['state'][param_id][key] = tensor

        if step:
            state_dict['optimizer']['state']['common_step'] = step
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
    config.optimizer = 'adam'

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
                    opt.state[p]['step'] = torch.tensor(0.0)
                    opt.state[p]['exp_avg'] = torch.zeros_like(p.data)
                    opt.state[p]['exp_avg_sq'] = torch.zeros_like(p.data)
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
        soap_wrapped = FP32Optimizer(
            soap_optimizer, config, soap_init_state_fn
        )

    optimizers.append(soap_wrapped)

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
