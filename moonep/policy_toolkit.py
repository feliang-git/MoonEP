"""Device-side building blocks an allocation policy is allowed to use.

An out-of-tree policy is CuTe DSL code inlined into MoonEP's planning kernel,
so it needs a few primitives that MoonEP already implements: a grid-wide
barrier and the warp-level argmax/argmin scans the greedy uses. Without a
sanctioned surface, every policy would reach into ``moonep.planning`` and
``moonep._common``, making MoonEP's internals its de-facto public API.

This module is that surface. Everything re-exported here is considered stable
for policy authors; anything else in ``moonep.planning`` / ``moonep._common``
is not, and may be renamed or restructured without notice.

Typical use:

    import cutlass
    import cutlass.cute as cute
    from cutlass import Int32
    from moonep.alloc_policy import register_alloc_policy
    from moonep.policy_toolkit import (
        ceil_div, grid_sync, reg_scan_argmax_min_idx, reg_scan_argmin_min_idx,
    )

    @cute.jit
    def my_policy(group_tokens, z, alloc, tpe_cumsum, s_alloc,
                  *, R, E, epn, CAP, bar, num_sms, num_threads, pid, tid):
        ...

    register_alloc_policy("mine", my_policy)

Contract for the policy body -- see ``docs/alloc_policy_contract.md``:

- Write ``z[R, R]`` (migration quotas) and ``alloc[R, E]`` (the seam) only.
  ``dst``, ``cu_seqlens``, ``experts_to_copy``, ``zero_fill_ranges`` and
  ``remote_stats`` belong to the backend and must not be touched.
- ``s_alloc`` is an ``[R, epn]`` shared-memory scratch view, reusable per
  owner rank; its contents on entry are undefined.
- ``tpe_cumsum[R-1, e]`` is the global token count of expert ``e``.
- The whole body runs between two grid-wide barriers. Use ``grid_sync`` for
  any additional cross-CTA ordering the policy needs, and
  ``cute.arch.barrier()`` for intra-CTA ordering.
"""

from moonep._common import cross_rank_barrier, grid_sync
from moonep.planning import (
    ceil_div,
    reg_scan_argmax_max_idx,
    reg_scan_argmax_min_idx,
    reg_scan_argmin_min_idx,
    warp_scan_argmax_max_idx,
    warp_scan_argmax_min_idx,
    warp_scan_argmin_min_idx,
)

__all__ = [
    "ceil_div",
    "cross_rank_barrier",
    "grid_sync",
    "reg_scan_argmax_max_idx",
    "reg_scan_argmax_min_idx",
    "reg_scan_argmin_min_idx",
    "warp_scan_argmax_max_idx",
    "warp_scan_argmax_min_idx",
    "warp_scan_argmin_min_idx",
]
