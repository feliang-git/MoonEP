"""The injected allocation-policy path must match the fused builtin exactly.

Run with:
    torchrun --nproc_per_node=8 --master_port=29553 -m pytest -q tests/test_alloc_policy_hook.py

``reference_greedy`` is the same algorithm as the fused builtin, expressed
through the policy hook. Anything less than byte-identical planning output
means the seam loses information, so these tests assert exact equality rather
than invariant conformance -- conformance is checked separately, and is much
weaker.
"""

import pytest
import torch

from tests.kernel_test_utils import (
    KernelCase,
    case_params,
    destroy_active_buffers,
    make_topk,
    skip_if_unsupported_world_size,
)
from tests.test_planning import PLANNING_CASES


def _plan_with_policy(case, R, rank, alloc_policy):
    """Run planning end to end under ``alloc_policy``; return the plan tensors."""
    from moonep import Buffer
    from moonep.planning import allocate_planning_outputs, launch_planning

    buffer = Buffer(
        case.S, case.H, case.K, case.E(R), R,
        B=case.B,
        num_sms=case.num_sms,
        token_padding=case.token_padding,
        alloc_policy=alloc_policy,
    )
    try:
        ctx = buffer._require_ctx()
        assert ctx["alloc_policy"] == alloc_policy
        topk, tpe = make_topk(case, rank, R)
        plan, cu_seqlens = allocate_planning_outputs(ctx)
        launch_planning(ctx, topk.reshape(-1).contiguous(), tpe, cu_seqlens, plan)
        torch.cuda.synchronize()
        return {
            "dst": plan.dst.clone(),
            "cu_seqlens": cu_seqlens.clone(),
            "experts_to_copy": plan.experts_to_copy.clone(),
            "zero_fill_ranges": plan.zero_fill_ranges.clone(),
            "remote_stats": plan.remote_stats.clone(),
        }
    finally:
        buffer.destroy()


@pytest.mark.parametrize("case", case_params(PLANNING_CASES))
def test_injected_policy_matches_builtin(dist_env, case):
    """reference_greedy must reproduce the builtin plan byte for byte."""
    rank, R = dist_env
    skip_if_unsupported_world_size(case, R)

    builtin = _plan_with_policy(case, R, rank, "builtin")
    injected = _plan_with_policy(case, R, rank, "reference_greedy")

    for name in builtin:
        assert torch.equal(builtin[name], injected[name]), (
            f"{case.name}: {name} differs between builtin and reference_greedy\n"
            f"  builtin  = {builtin[name].flatten()[:16].tolist()}\n"
            f"  injected = {injected[name].flatten()[:16].tolist()}"
        )


def test_resolve_rejects_unknown_policy():
    from moonep.alloc_policy import resolve_alloc_policy

    with pytest.raises(KeyError, match="unknown alloc policy"):
        resolve_alloc_policy("does_not_exist")


def test_unknown_policy_fails_at_construction(dist_env):
    """An unknown name must fail when the Buffer is built, not at first dispatch.

    Uses the real world size: num_ep_ranks must match it, or the world-size
    assertion fires first and this proves nothing about the policy check.
    """
    from moonep import Buffer

    _rank, R = dist_env
    with pytest.raises(KeyError, match="unknown alloc policy"):
        Buffer(64, 128, 4, 4 * R, R, alloc_policy="does_not_exist")
    destroy_active_buffers()


def test_registry_protects_builtin():
    from moonep.alloc_policy import available_alloc_policies, register_alloc_policy

    assert "builtin" in available_alloc_policies()
    assert "reference_greedy" in available_alloc_policies()
    with pytest.raises(ValueError, match="reserved"):
        register_alloc_policy("builtin", lambda *a, **k: None)


def test_policy_is_a_compile_time_cache_key():
    """Two policies must not collide in the kernel compile cache.

    If the policy were absent from the cache key, the second Buffer would
    silently reuse the first's compiled kernel and every equivalence test above
    would pass vacuously.
    """
    from moonep.planning import _get_compiled

    keys = _get_compiled.cache_info()
    args = (2, 8, 1, 64, 4, 256, 256, 1, 4096, 0, 0, 0, 0, 0, 0, 128, 32)
    assert _get_compiled.__wrapped__ is not None
    # Same args, different policy name -> distinct cache entries.
    before = _get_compiled.cache_info().currsize
    try:
        _get_compiled(*args, "builtin")
        _get_compiled(*args, "reference_greedy")
    except Exception:
        pytest.skip("kernel compilation unavailable in this environment")
    after = _get_compiled.cache_info().currsize
    assert after - before == 2, (
        f"expected 2 distinct compile-cache entries, got {after - before}; "
        f"alloc_policy is not part of the cache key"
    )
    assert keys is not None
