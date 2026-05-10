# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for Shampoo TP-aware Kronecker factor update.

Spawns 2 gloo ranks on localhost and verifies _update_kron_factors_tp matches
the global reference for both partition_dim=0 and partition_dim=1.

Run: pytest tests/unit_tests/test_shampoo_tp.py -v
"""

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# Import path: this test sits in modules/Megatron-LM/tests/unit_tests/
# and the optimizer lives at megatron/core/optimizer/shampoo.py
from megatron.core.optimizer.shampoo import _update_kron_factors_tp


class _PgCollectionStub:
    """Minimal pg_collection stand-in: only the .tp / .expt_tp attrs are touched."""

    def __init__(self, tp_group):
        self.tp = tp_group
        self.expt_tp = tp_group


def _run_partition_dim_1(rank, world_size, d_out, d_in_local, beta, atol_local, atol_reduced, full_g):
    """Worker: split full_g along dim 1, expect L all-reduced, R = local block."""
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29501'
    dist.init_process_group(backend='gloo', rank=rank, world_size=world_size)

    tp_group = dist.new_group(list(range(world_size)))

    # Local slice: rank-i slice of full_g along dim 1.
    g_local = full_g[:, rank * d_in_local:(rank + 1) * d_in_local].contiguous()

    # Fake parameter with partition_dim=1.
    p = torch.zeros(d_out, d_in_local)
    p.partition_dim = 1
    p.tensor_model_parallel = True

    L = torch.zeros(d_out, d_out)
    R = torch.zeros(d_in_local, d_in_local)

    pg = _PgCollectionStub(tp_group)
    _update_kron_factors_tp(p, L, R, g_local, beta, pg)

    # Expected:
    L_full_ref = full_g @ full_g.T
    R_full_ref = full_g.T @ full_g
    expected_L = (1 - beta) * L_full_ref
    expected_R = (1 - beta) * R_full_ref[
        rank * d_in_local:(rank + 1) * d_in_local,
        rank * d_in_local:(rank + 1) * d_in_local,
    ]

    assert torch.allclose(L, expected_L, atol=atol_reduced), \
        f"[rank {rank}] L mismatch (max abs diff {(L - expected_L).abs().max()})"
    assert torch.allclose(R, expected_R, atol=atol_local), \
        f"[rank {rank}] R diag-block mismatch (max abs diff {(R - expected_R).abs().max()})"

    dist.destroy_process_group()


def _run_partition_dim_0(rank, world_size, d_out_local, d_in, beta, atol_local, atol_reduced, full_g):
    """Worker: split full_g along dim 0, expect R all-reduced, L = local block."""
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29502'
    dist.init_process_group(backend='gloo', rank=rank, world_size=world_size)

    tp_group = dist.new_group(list(range(world_size)))

    g_local = full_g[rank * d_out_local:(rank + 1) * d_out_local, :].contiguous()

    p = torch.zeros(d_out_local, d_in)
    p.partition_dim = 0
    p.tensor_model_parallel = True

    L = torch.zeros(d_out_local, d_out_local)
    R = torch.zeros(d_in, d_in)

    pg = _PgCollectionStub(tp_group)
    _update_kron_factors_tp(p, L, R, g_local, beta, pg)

    L_full_ref = full_g @ full_g.T
    R_full_ref = full_g.T @ full_g
    expected_L = (1 - beta) * L_full_ref[
        rank * d_out_local:(rank + 1) * d_out_local,
        rank * d_out_local:(rank + 1) * d_out_local,
    ]
    expected_R = (1 - beta) * R_full_ref

    assert torch.allclose(L, expected_L, atol=atol_local), \
        f"[rank {rank}] L diag-block mismatch (max abs diff {(L - expected_L).abs().max()})"
    assert torch.allclose(R, expected_R, atol=atol_reduced), \
        f"[rank {rank}] R mismatch (max abs diff {(R - expected_R).abs().max()})"

    dist.destroy_process_group()


@pytest.mark.skipif(
    not hasattr(torch.distributed, 'is_gloo_available') or not torch.distributed.is_gloo_available(),
    reason='gloo backend not available',
)
def test_kron_partition_dim_1():
    """partition_dim=1: split G along input dim. L is the partial sum (all-reduced),
    R is a local diagonal block."""
    torch.manual_seed(0)
    world_size = 2
    d_out, d_in = 4, 8
    d_in_local = d_in // world_size
    beta = 0.95
    full_g = torch.randn(d_out, d_in, dtype=torch.float32)

    mp.spawn(
        _run_partition_dim_1,
        args=(world_size, d_out, d_in_local, beta, 1e-5, 1e-4, full_g),
        nprocs=world_size,
        join=True,
    )


@pytest.mark.skipif(
    not hasattr(torch.distributed, 'is_gloo_available') or not torch.distributed.is_gloo_available(),
    reason='gloo backend not available',
)
def test_kron_partition_dim_0():
    """partition_dim=0: split G along output dim. R is the partial sum (all-reduced),
    L is a local diagonal block."""
    torch.manual_seed(0)
    world_size = 2
    d_out, d_in = 8, 4
    d_out_local = d_out // world_size
    beta = 0.95
    full_g = torch.randn(d_out, d_in, dtype=torch.float32)

    mp.spawn(
        _run_partition_dim_0,
        args=(world_size, d_out_local, d_in, beta, 1e-5, 1e-4, full_g),
        nprocs=world_size,
        join=True,
    )


def test_kron_no_pg_collection():
    """pg_collection=None: no all-reduce, factors are pure local outer products."""
    torch.manual_seed(1)
    d_out, d_in = 4, 6
    grad = torch.randn(d_out, d_in, dtype=torch.float32)
    beta = 0.9

    p = torch.zeros(d_out, d_in)
    L = torch.zeros(d_out, d_out)
    R = torch.zeros(d_in, d_in)

    _update_kron_factors_tp(p, L, R, grad, beta, pg_collection=None)

    assert torch.allclose(L, (1 - beta) * (grad @ grad.T), atol=1e-6)
    assert torch.allclose(R, (1 - beta) * (grad.T @ grad), atol=1e-6)


def test_kron_partition_dim_minus_one_normalized_to_none():
    """partition_dim=-1 means 'not partitioned'; treated like None: no all-reduce."""
    torch.manual_seed(2)
    d_out, d_in = 4, 6
    grad = torch.randn(d_out, d_in, dtype=torch.float32)
    beta = 0.9

    p = torch.zeros(d_out, d_in)
    p.partition_dim = -1  # roo.py's "not partitioned" sentinel
    L = torch.zeros(d_out, d_out)
    R = torch.zeros(d_in, d_in)

    # Even with pg_collection set, partition_dim=-1 must skip the all-reduce.
    # We pass None here because building a real PG isn't necessary for this branch.
    _update_kron_factors_tp(p, L, R, grad, beta, pg_collection=None)

    assert torch.allclose(L, (1 - beta) * (grad @ grad.T), atol=1e-6)
    assert torch.allclose(R, (1 - beta) * (grad.T @ grad), atol=1e-6)
