"""MoonEP's greedy allocation policy, expressed as an injected policy.

This is deliberately a near-copy of the fused builtin code in
``planning.py``. It exists for three reasons:

1. It exercises the injected code path, so the hook is not dead code.
2. It is the bit-equivalence oracle for out-of-tree ports: an MLB policy
   claiming to reproduce MoonEP's algorithm must match this byte for byte,
   which is a far sharper test than "satisfies the invariants".
3. It documents, by being a working example, exactly what a policy may touch.

The one difference from the fused version is what it does *not* do: it writes
``z`` and ``alloc`` only, leaving the ``alloc_cumsum`` prefix to the backend.
That separation is the whole point of the seam.
"""

import cutlass
import cutlass.cute as cute
from cutlass import Int32

from moonep._common import grid_sync
from moonep.planning import ceil_div, reg_scan_argmax_min_idx, reg_scan_argmin_min_idx


@cute.jit
def reference_greedy(
    group_tokens, z, alloc, tpe_cumsum, s_alloc,
    *, R, E, epn, CAP, bar, num_sms, num_threads, pid, tid,
):
    """Fill the roomiest rank from the most overloaded home group, repeatedly.

    Writes ``z[R, R]`` (migration quotas) and ``alloc[R, E]``.

    The receiver is refilled to exactly CAP in one shot, which is what keeps
    invariant 4 true: once ``balance[u] == 0``, ``u`` is never selected again,
    so every destination rank receives from at most one home group.
    """
    # ---- part A: rank-level balance -> z[R, R] ----
    if pid == 0:
        if tid < 32:
            lane = tid
            CHUNK = cutlass.const_expr(ceil_div(R, 32))
            balance = cute.make_rmem_tensor(CHUNK, Int32)
            for j in cutlass.range_constexpr(CHUNK):
                k = lane + j * 32
                balance[j] = 0
                if k < R: balance[j] = group_tokens[k] - CAP
            keep_balancing = True
            while keep_balancing:
                surplus, surplus_rank = reg_scan_argmax_min_idx(balance, R, lane)
                deficit, deficit_rank = reg_scan_argmin_min_idx(balance, R, lane)
                if surplus <= 0 or deficit >= 0:
                    keep_balancing = False
                else:
                    move_tokens = -deficit
                    for j in cutlass.range_constexpr(CHUNK):
                        k = lane + j * 32
                        if k == surplus_rank: balance[j] -= move_tokens
                        elif k == deficit_rank: balance[j] = 0
                    if lane == 0:
                        z[surplus_rank, deficit_rank] = move_tokens
                    cute.arch.sync_warp()
    grid_sync(bar, num_sms, tid)

    # ---- part B: quotas -> per-expert allocation ----
    for owner_rank in cutlass.range(pid, R, num_sms):
        expert_base = owner_rank * epn

        for idx in cutlass.range(tid, epn * R, num_threads):
            local_expert_id = idx // R
            rank_idx = idx - local_expert_id * R
            global_expert = expert_base + local_expert_id
            s_alloc[rank_idx, local_expert_id] = (
                tpe_cumsum[R - 1, global_expert] if rank_idx == owner_rank else 0
            )

        cute.arch.barrier()

        if tid < 32:
            lane = tid
            R_CHUNK = cutlass.const_expr(ceil_div(R, 32))
            EPN_CHUNK = cutlass.const_expr(ceil_div(epn, 32))
            quotas = cute.make_rmem_tensor(R_CHUNK, Int32)
            owner_remaining = cute.make_rmem_tensor(EPN_CHUNK, Int32)
            for j in cutlass.range_constexpr(R_CHUNK):
                rank_idx = lane + j * 32
                quotas[j] = 0
                if rank_idx < R: quotas[j] = z[owner_rank, rank_idx]
            for j in cutlass.range_constexpr(EPN_CHUNK):
                local_expert_id = lane + j * 32
                owner_remaining[j] = 0
                if local_expert_id < epn:
                    owner_remaining[j] = s_alloc[owner_rank, local_expert_id]

            keep_balancing = cutlass.Boolean(True)
            while keep_balancing:
                max_quota, target_rank = reg_scan_argmax_min_idx(quotas, R, lane)
                if max_quota <= 0:
                    keep_balancing = cutlass.Boolean(False)
                else:
                    max_remaining, selected_expert_id = reg_scan_argmax_min_idx(
                        owner_remaining, epn, lane)
                    if max_remaining <= 0:
                        keep_balancing = cutlass.Boolean(False)
                    else:
                        take = cutlass.min(max_remaining, max_quota)
                        for j in cutlass.range_constexpr(R_CHUNK):
                            rank_idx = lane + j * 32
                            if rank_idx == target_rank: quotas[j] = max_quota - take
                        for j in cutlass.range_constexpr(EPN_CHUNK):
                            local_expert_id = lane + j * 32
                            if local_expert_id == selected_expert_id:
                                owner_remaining[j] = max_remaining - take
                        if tid == 0:
                            s_alloc[target_rank, selected_expert_id] += take
                            s_alloc[owner_rank, selected_expert_id] = max_remaining - take
                        cute.arch.sync_warp()
        cute.arch.barrier()

        for idx in cutlass.range(tid, epn * R, num_threads):
            rank_idx = idx // epn
            local_expert_id = idx - rank_idx * epn
            global_expert = expert_base + local_expert_id
            alloc[rank_idx, global_expert] = s_alloc[rank_idx, local_expert_id]

        cute.arch.barrier()
