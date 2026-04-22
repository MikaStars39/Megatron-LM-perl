# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple, Union

import torch

from ..utils import is_te_min_version


@dataclass(frozen=True, slots=True)
class ParamKey:
    """Key to group parameters by. All such grouped parameters can share an
    optimizer config specification."""

    # TODO: Can add layer_id here later.

    name: Union[str, Tuple[str]] = field(default_factory=tuple)
    """Parameter name(s)."""

    attr: Union[str, Tuple[str]] = field(default_factory=tuple)
    """Parameter attribute(s)."""


@dataclass
class OptimizerConfig:
    """Base optimizer configuration object."""

    ##############
    # General
    ##############
    lr: Optional[float] = None
    """Initial learning rate. Depending on decay style and initial warmup, the learning rate at each
       iteration would be different.
    """

    min_lr: Optional[float] = None
    """Minumum value for learning rate. The scheduler clip values below this threshold."""

    weight_decay: float = 0.01
    """Weight decay coefficient for L2 regularization."""

    ##############
    # Precision
    ##############
    fp8_recipe: Optional[str] = None
    """The type of fp8 recipe will affect the processing logic inside distributed optimizer."""

    fp16: bool = False
    """If true, train with fp16 mixed precision training. Defaults to False."""

    bf16: bool = False
    """If true, train with bf16 mixed precision training. Defaults to False."""

    reuse_grad_buf_for_mxfp8_param_ag: bool = False
    """If true, reuse the grad buffer for param AG when using mxfp8 recipe. Should be 
       set to True only when fp8_recipe is mxfp8 and fp8_param_gather is True."""

    params_dtype: torch.dtype = torch.float32
    """dtype used when intializing the weights. Defaults to torch.float32."""

    use_precision_aware_optimizer: bool = False
    """If true, allows optimizer-related tensors (master_param, gradients and optimizer states)
    to be set to lower precision. Defaults to False.
    """

    store_param_remainders: bool = True
    """If true, store the 16-bit FP32 parameter remainders in the optimizer state, excluding the
        16 bits shared with the BF16 parameters. This lowers GPU memory usage. Defaults to True.
    """

    main_grads_dtype: torch.dtype = torch.float32
    """dtype of main grads when enabling precision-aware-optimizer"""

    main_params_dtype: torch.dtype = torch.float32
    """dtype of main params when enabling precision-aware-optimizer"""

    exp_avg_dtype: torch.dtype = torch.float32
    """dtype of exp_avg when enabling precision-aware-optimizer"""

    exp_avg_sq_dtype: torch.dtype = torch.float32
    """dtype of exp_avg_sq when enabling precision-aware-optimizer"""

    optimizer: str = 'adam'
    """Optimizer name. NOTE: Deprecated, use individual optimizer classes instead."""

    ###############
    # Loss scaling
    ###############
    loss_scale: Optional[float] = None
    """Static loss scaling, positive power of 2 values can improve fp16 convergence. If None,
       dynamic loss scaling is used.
    """

    initial_loss_scale: float = 2**32
    """Initial loss-scale for dynamic loss scaling."""

    min_loss_scale: float = 1.0
    """Minimum loss scale for dynamic loss scaling."""

    loss_scale_window: float = 1000
    """Window over which to raise/lower dynamic scale."""

    hysteresis: int = 2
    """Hysteresis for dynamic loss scaling."""

    ###################################################################################
    # Optimizer (NOTE: Deprecated, use individual optimizer classes instead.).
    ###################################################################################
    # Adam.
    adam_beta1: float = 0.9
    """First coefficient for computing running averages of gradient and its square in Adam
    optimizer.
    """

    adam_beta2: float = 0.999
    """Second coefficient for computing running averages of gradient and its square in Adam
    optimizer.
    """

    adam_eps: float = 1e-08
    """Term added to the denominator to improve numerical stability in Adam optimizer."""

    decoupled_weight_decay: bool = True
    """If true, decouples weight decay from the gradient update, equivalent to AdamW. If false,
    original Adam update rule will be used. Defaults to True.
    """

    # SGD.
    sgd_momentum: float = 0.9
    """Momentum factor for SGD optimizer."""

    # RMSprop.
    rmsprop_alpha: float = 0.99
    """Smoothing constant (decay rate) for RMSprop optimizer."""

    rmsprop_eps: float = 1e-8
    """Term added to the denominator to improve numerical stability in RMSprop optimizer."""

    rmsprop_momentum: float = 0.0
    """Momentum factor for RMSprop optimizer."""

    rmsprop_centered: bool = False
    """If true, compute the centered RMSprop (gradient normalized by its variance)."""

    # Muon
    muon_momentum: float = 0.95
    """The momentum used by the internal SGD."""

    muon_split_qkv: bool = True
    """Whether to split QKV parameters for Muon optimizer."""

    muon_use_nesterov: bool = False
    """Whether to use Nesterov-style momentum in the internal SGD."""

    muon_scale_mode: str = "spectral"
    """The mode to use for the scale factor. Defaults to "spectral"."""

    muon_fp32_matmul_prec: str = "medium"
    """The precision to use for the fp32 matmul. Defaults to "medium"."""

    muon_num_ns_steps: int = 5
    """The number of iteration steps to use in the Newton-Schulz iteration."""

    muon_tp_mode: str = "blockwise"
    """How to perform NS calculation for tensor parallel weights. Defaults to "blockwise"."""

    muon_extra_scale_factor: float = 1.0
    """Additional scale factor for the muon update."""

    # Roo (Matrix Natural Gradient)
    roo_momentum: float = 0.95
    """Momentum coefficient for Roo optimizer."""

    roo_epsilon: float = 0.01
    """Regularization epsilon for Roo inverse-spectral transform. Controls the crossover
    point: singular values >> eps are damped (1/sigma), those << eps are amplified (sigma/eps^2)."""

    roo_scale_factor: float = 1.0
    """Scale factor applied after RMS normalization of the Roo update."""

    roo_split_qkv: bool = True
    """Whether to split QKV parameters for Roo optimizer."""

    roo_tp_mode: str = "allreduce_gram"
    """TP handling mode for Roo gram matrix computation.
    'allreduce_gram': allreduce M^T M across TP ranks for row-parallel params.
    'blockwise': treat each TP shard independently (no communication)."""

    roo_num_ns_steps: int = 5
    """Number of Newton-Schulz iteration steps for gram matrix inversion."""

    roo_fp32_matmul_prec: str = "medium"
    """FP32 matmul precision for Newton-Schulz iteration in Roo ('low', 'medium', 'high')."""

    # RooNormal (SVD-based spectral clipping with Muon-style EMA)
    roo_normal_momentum: float = 0.95
    """Momentum coefficient (β) for RooNormal EMA: buf = β*buf + (1-β)*grad."""

    roo_normal_clip_value: float = 20.0
    """Singular value clipping threshold: f(σ) = min(1/(σ+ε), clip_value)."""

    roo_normal_epsilon: float = 1e-7
    """Small constant added before reciprocal to prevent division by zero."""

    roo_normal_scale_factor: float = 1.0
    """Scale factor applied after RMS normalization of the RooNormal update."""

    roo_normal_use_nesterov: bool = True
    """Whether to use Nesterov-style momentum (same as Muon default)."""

    roo_normal_split_qkv: bool = True
    """Whether to split QKV parameters for RooNormal optimizer."""

    roo_normal_svd_log_interval: int = 10
    """Log SVD singular value statistics every N steps. 0 disables logging."""

    roo_normal_svd_log_dir: str = ""
    """Directory to write SVD singular value logs. Empty string disables file logging."""

    # MuonProjected (Muon with initial-weight projection)
    muon_projected_momentum: float = 0.95
    """Momentum coefficient (beta) for MuonProjected EMA."""

    muon_projected_use_nesterov: bool = True
    """Whether to use Nesterov-style momentum for MuonProjected."""

    muon_projected_projection_alpha: float = 0.5
    """Coefficient for the projection: (I - alpha * W0 @ W0^T)."""

    muon_projected_split_qkv: bool = True
    """Whether to split QKV parameters for MuonProjected optimizer."""

    muon_projected_num_ns_steps: int = 5
    """Number of Newton-Schulz iteration steps for Muon orthogonalization."""

    muon_projected_scale_mode: str = "spectral"
    """Scale mode for Muon update ('spectral', 'unit_rms_norm', 'shape_scaling')."""

    muon_projected_extra_scale_factor: float = 1.0
    """Additional scale factor for the Muon update."""

    muon_projected_fp32_matmul_prec: str = "medium"
    """FP32 matmul precision for NS iteration ('low', 'medium', 'high')."""

    muon_projected_tp_mode: str = "blockwise"
    """TP mode for Newton-Schulz and projection ('blockwise', 'duplicated', 'distributed')."""

    # GASD (Geometry-Aware Steepest Descent)
    gasd_momentum: float = 0.95
    """Momentum coefficient (beta) for GASD EMA."""

    gasd_use_nesterov: bool = True
    """Whether to use Nesterov-style momentum for GASD."""

    gasd_epsilon_alpha: float = 1.0
    """Minimum coefficient for adaptive epsilon: eps = alpha(t) * ||W||_F^2 / min(n, m)."""

    gasd_epsilon_alpha_max: float = 20.0
    """Maximum coefficient for epsilon annealing."""

    gasd_epsilon_warmup_steps: int = 50
    """Hold epsilon at epsilon_alpha for this many initial steps."""

    gasd_epsilon_ramp_end_steps: int = 800
    """Step at which epsilon reaches epsilon_alpha_max (linear ramp from warmup to here)."""

    gasd_cg_iters: int = 10
    """Number of Conjugate Gradient iterations for solving (WW^T + eps*I) Delta = G."""

    gasd_rms_scale: float = 1.0
    """Scale factor applied after RMS normalization of the CG output."""

    gasd_split_qkv: bool = True
    """Whether to split QKV parameters for GASD optimizer."""

    gasd_num_ns_steps: int = 5
    """Number of Newton-Schulz iteration steps for Muon orthogonalization in GASD."""

    gasd_scale_mode: str = "spectral"
    """Scale mode for Muon update in GASD ('spectral', 'unit_rms_norm', 'shape_scaling')."""

    gasd_extra_scale_factor: float = 1.0
    """Additional scale factor for the Muon update in GASD."""

    gasd_fp32_matmul_prec: str = "medium"
    """FP32 matmul precision for NS iteration in GASD ('low', 'medium', 'high')."""

    gasd_tp_mode: str = "blockwise"
    """TP mode for Newton-Schulz and GASD CG ('blockwise', 'duplicated', 'distributed')."""

    # SOAP (ShampoO with Adam in the Preconditioner eigenbasis).
    soap_beta1: float = 0.9
    """First coefficient for inner Adam in SOAP optimizer."""

    soap_beta2: float = 0.95
    """Second coefficient for inner Adam in SOAP optimizer."""

    soap_shampoo_beta: float = 0.95
    """Beta for the Kronecker factor matrices moving average in SOAP optimizer."""

    soap_eps: float = 1e-8
    """Inner Adam epsilon for numerical stability in SOAP optimizer."""

    soap_precondition_frequency: int = 1
    """How often to update the preconditioner eigenbasis in SOAP optimizer."""

    soap_adam_warmup_steps: int = 0
    """Number of steps using plain Adam before enabling preconditioning in SOAP optimizer."""

    soap_correct_bias: bool = True
    """Whether to use bias correction in inner Adam and Kronecker factor EMA."""

    soap_fp32_matmul_prec: str = "high"
    """Precision of matmul operations in SOAP optimizer ('low', 'medium', 'high')."""

    soap_use_eigh: bool = False
    """Whether to use full symmetric eigendecomposition (eigh) to compute eigenbasis."""

    soap_power_iter_steps: int = 1
    """Number of power iteration steps before QR decomposition."""

    soap_max_update_rms: float = 0.0
    """Clip the update RMS to this value (0 means no clipping)."""

    soap_nesterov: bool = False
    """Whether to use Nesterov momentum in inner Adam."""

    soap_split_qkv: bool = True
    """Whether to split QKV parameters for SOAP optimizer."""

    #######################
    # Distributed optimizer
    #######################
    use_distributed_optimizer: bool = False
    """Distribute optimizer state over data-parallel replicas."""

    overlap_param_gather: bool = False
    """If true, overlap param all-gather with forward compute. 
        This argument is intended to have the same value as the "overlap_param_gather" argument 
        in the "distributed_data_parallel_config.py" file. In the optimizer, this argument is 
        only used when "reuse_grad_buf_for_mxfp8_param_ag=True & fp8_param_gather=True".
    """

    overlap_param_gather_with_optimizer_step: bool = False
    """If true, overlap param all-gather of first bucket with optimizer step."""

    #######################
    # Optimizer Offload
    #######################

    optimizer_cpu_offload: bool = False
    """If True, offload optimizer states tensor and compute to CPU."""

    optimizer_offload_fraction: float = 0.0
    """Specifies the fraction of optimizer states to offload from GPU memory to CPU."""

    use_torch_optimizer_for_cpu_offload: bool = False
    """If True, use torch.optim.Optimizer for CPU offload."""

    overlap_cpu_optimizer_d2h_h2d: bool = False
    """
    When set to `True`, this flag enables overlapping of the CPU optimizer
    update process with the data transfer operations. This can help improve
    overall training efficiency by reducing idle time during data movement,
    allowing the optimizer to perform updates while gradients and parameters
    are being transferred between devices.
    """

    pin_cpu_grads: bool = True
    """If True, pin the optimizer gradients to CPU memory."""

    pin_cpu_params: bool = True
    """If True, pin the optimizer parameters to CPU memory."""

    ################
    # Miscellaneous
    ################
    clip_grad: float = 1.0
    """Gradient clipping based on global L2 norm."""

    log_num_zeros_in_grad: bool = False
    """If true, calculate and log the number of zeros in gradient."""

    barrier_with_L1_time: bool = False
    """If true, use barrier with level 1 time measurements."""

    timers: Optional[Callable] = None
    """Function to get timers."""

    config_logger_dir: str = ""
    """When non-empty, dumps entry-point configs to config_logger_dir"""

    def __post_init__(self):
        """Check the validity of the config."""

        # The following condition is used to avoid repetition in distrib_optimizer.py.
        # This is because in distrib_optimizer.py, the process to handle parameters are
        # different for different training precision settings. FP8 cases require different
        # handling while FP8 delayed scaling is an exception because the Adam optimizer in
        # TransformerEngine supports it in the kernel computation.
        # This is also the flag to determine the usage of param.grad or param.decoupled_grad
        self.use_precision_aware_optimizer_no_fp8_or_ds_fp8 = (
            self.use_precision_aware_optimizer
            and (
                self.main_params_dtype != torch.float32
                or (self.fp8_recipe is None or self.fp8_recipe == "delayed")
                or self.optimizer_cpu_offload
            )
        )

        if self.fp8_recipe == "mxfp8":
            if not self.reuse_grad_buf_for_mxfp8_param_ag:
                import warnings

                warnings.warn(
                    "mxfp8 without using reuse_grad_buf_for_mxfp8_param_ag and fp8_param_gather"
                    "will use significant amount additional GPU memory."
                    "Setting --reuse-grad-buf-for-mxfp8-param-ag and --fp8-param-gather is "
                    "recommended for mxfp8 training."
                )

        if self.use_precision_aware_optimizer:
            assert (
                self.optimizer == 'adam'
            ), '--use-precision-aware-optimizer only supported with adam'
            assert (
                self.use_distributed_optimizer
            ), '--use-precision-aware-optimizer only supported with distributed optimizer'

            if not is_te_min_version("2.1.0"):
                self.store_param_remainders = False

            # Only the FusedAdam in TE and HybridDeviceOptimizer supports
            # --use-precision-aware-optimizer.
            # TODO: Remove this check when apex's FusedAdam is no longer used.
            if self.optimizer_cpu_offload:
                return
            try:
                import inspect

                # TODO: Move this below?
                from transformer_engine.pytorch.optimizers import FusedAdam as Adam

                adam_args = inspect.signature(Adam).parameters
                arg_names = [
                    'master_weight_dtype',
                    'exp_avg_dtype',
                    'exp_avg_sq_dtype',
                    'use_decoupled_grad',
                ]
                for name in arg_names:
                    assert name in adam_args, (
                        "Current FusedAdam of TE doesn't support --use-precision-aware-optimizer, "
                        "please update TE version."
                    )
            except ImportError:
                raise RuntimeError(
                    '--use-precision-aware-optimizer requires FusedAdam from TransformerEngine, '
                    'but not found.'
                )
        else:
            assert (
                self.main_grads_dtype == torch.float32
            ), "main_grads_dtype can only be fp32 when not using precision-aware optimizer"
            assert (
                self.main_params_dtype == torch.float32
            ), "main_params_dtype can only be fp32 when not using precision-aware optimizer"
            assert (
                self.exp_avg_dtype == torch.float32
            ), "exp_avg_dtype can only be fp32 when not using precision-aware optimizer"
            assert (
                self.exp_avg_sq_dtype == torch.float32
            ), "exp_avg_sq_dtype can only be fp32 when not using precision-aware optimizer"


@dataclass
class AdamOptimizerConfig(OptimizerConfig):
    """Adam optimizer configuration object."""

    optimizer: str = 'adam'
    """Optimizer name."""

    adam_beta1: float = 0.9
    """First coefficient for computing running averages of gradient and its square in Adam
    optimizer.
    """

    adam_beta2: float = 0.999
    """Second coefficient for computing running averages of gradient and its square in Adam
    optimizer.
    """

    adam_eps: float = 1e-08
    """Term added to the denominator to improve numerical stability in Adam optimizer."""


@dataclass
class SGDOptimizerConfig(OptimizerConfig):
    """SGD optimizer configuration object."""

    optimizer: str = 'sgd'
    """Optimizer name."""

    sgd_momentum: float = 0.9
    """Momentum factor for SGD optimizer."""
