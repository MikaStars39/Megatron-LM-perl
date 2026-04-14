# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""SOAP optimizer: ShampoO with Adam in the Preconditioner eigenbasis.

Megatron integration wrapper that splits parameters into:
  - Linear (2D, non-embedding) -> SOAP
  - Nonlinear (embedding, LayerNorm, bias) -> Adam

Uses the SOAP implementation from emerging_optimizers.
"""

import logging
from typing import Dict, List, Optional

import torch

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
        # SOAP manages its own state lazily; just trigger initialization
        for group in opt.param_groups:
            for p in group['params']:
                if len(opt.state[p]) == 0:
                    opt.state[p]['step'] = torch.tensor(0.0)
                    opt.state[p]['exp_avg'] = torch.zeros_like(p.data)
                    opt.state[p]['exp_avg_sq'] = torch.zeros_like(p.data)

    if config.fp16:
        raise Exception('SOAP with fp16 is not supported.')

    if config.bf16:
        soap_wrapped = Float16OptimizerWithFloat16Params(
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
