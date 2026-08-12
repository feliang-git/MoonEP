# The allocation-policy contract

MoonEP's planning kernel is one cooperative CuTe-DSL kernel running four
barrier-separated steps. Exactly one of them is a *policy* decision; the other
three are mechanism. This document pins the boundary so the policy can be
replaced without touching the communication backend.

## Where the boundary is

| Step | Where | Role |
|---|---|---|
| 1. local histogram | `planning.py:399` (`run_c1`) | mechanism |
| 2. cross-rank `tpe` exchange | `planning.py:609` | mechanism |
| **3. allocation policy** | **`planning.py:673–798`** | **policy** |
| 4. table build | `planning.py:833+` | mechanism |

Step 3 is bounded by `grid_sync` at `planning.py:670` and `planning.py:798`. It
reads `group_tokens[R]` (and the exchanged `tpe` in the multicast `meta`
buffer) and writes `z[R,R]` and `alloc[R,E]`.

Everything in step 4 — `expert_off`, `cu_seqlens`, `experts_to_copy`,
`zero_fill_ranges`, `remote_stats`, `dst` — is a **pure function of `alloc`**.
See `tests/planning_reference.py` part 2 for the derivation in readable form.

## The seam: `alloc[R, E]`

`int32`, kernel layout (flat as `alloc[d * E + e]`).

> `alloc[d, e]` = the number of **global** tokens routed to expert `e` that are
> computed on destination rank `d`.

`tests/planning_reference.py` already returns exactly this matrix via
`launch_planning_torch_reference(..., return_alloc=True)`, and the existing
kernel tests compare against it. The contract is therefore not new API surface —
it is an existing, already-tested intermediate being promoted to a boundary.

## Invariants

A conforming policy must satisfy all five. `tests/alloc_contract.py` implements
the checker; `tests/test_alloc_contract.py` runs it over adversarial inputs and
includes a negative control proving each check can actually fire.

1. **Non-negative** — `alloc >= 0`.
2. **Conservation** — `alloc.sum(dim=0) == tpe.sum(dim=0)`. Every routed token
   is computed exactly once.
3. **Capacity** — `alloc.sum(dim=1) <= S*K`. No destination rank exceeds its
   real token capacity.
4. **Segment bound** — `(alloc[d] > 0).sum() <= 2 * epn` for every `d`.
5. **Padded fit** — the step-4 padded layout fits in `NvS`. Implied by 3 + 4;
   checked directly as a backstop.

### Invariant 4 is the one that will bite you

It is stated nowhere upstream, but it is load-bearing. `api.py:278` sizes the
dispatch buffer as:

```python
token_padding_extra = (token_padding - 1) * 2 * epn
NvS = NvS_capacity + token_padding_extra
```

The padding headroom budgets for at most `2 * epn` **non-empty VM groups per
destination rank**, each able to waste up to `token_padding - 1` slots rounding
up to a segment boundary. A policy that spreads one destination rank's inbound
tokens across more than `2 * epn` distinct experts overflows `NvS` — tripping
the "padded layout exceeds NvS" assertion in the torch reference, or corrupting
memory past the buffer in the fused kernel.

The builtin greedy policy satisfies this **by construction, not by tuning**:

```python
h = balance.argmax(); u = balance.argmin()
if balance[h] <= 0: break
z[h, u] = -balance[u]; balance[h] += balance[u]; balance[u] = 0
```

Selecting receiver `u` sets `balance[u] = 0`, so `u` is never `argmin` again
while any deficit remains. Every destination rank therefore receives from **at
most one home group**, giving it at most `epn` own experts plus at most `epn`
received experts.

**The one-donor property is sufficient, not necessary.** What the buffer
actually requires is the *count*: at most `2*epn` non-empty groups per
destination rank, because that is how many `token_padding - 1` roundings the
headroom pays for. A policy may take from several home groups provided the
number of distinct *received* experts on any rank stays within `epn`. That
freedom is real and is where a better policy has room to work.

`test_segment_bound_is_tight_not_incidental` asserts the one-donor mechanism,
but it is a *characterization test of the builtin*, not part of the contract:
`check_policy` deliberately runs only `alloc_invariant_errors`, i.e. the count
bound. Do not extend the donor assertion to third-party policies.

This is still the largest constraint on the design space: a naive waterfill or
LP formulation that optimizes only max-rank-load will violate invariant 4
immediately, because spreading load finely across ranks is exactly what it wants
to do. Any such formulation needs an explicit cardinality constraint on
non-zero entries per row. Raising the bound itself is possible but is an **ABI
change** — it changes `NvS` and therefore every buffer size in
`_create_context`.

## Quality metrics (not invariants)

- **B-pressure** — per destination rank, the count of non-empty *remote* expert
  groups. Step 4 grants weight-prefetch slots to only the top-`B` of these
  (`planning_reference.py:158`). Beyond `B`, remote experts still compute
  correctly but are not prefetched: a silent throughput cliff, not an error.
  `alloc` alone cannot express this budget, so policies must be *measured* on
  it. `tests/alloc_contract.py:b_pressure` reports it.
- **Max rank load** — `alloc.sum(dim=1).max()`, the objective being minimized.
- **Migration volume** — total tokens placed off their home rank. Each migrated
  expert costs a weight prefetch, so at equal max load, less migration is
  strictly better. `test_no_gratuitous_migration` asserts the policy does not
  move anything when the home placement already fits.

## Policy interface

Selection is a **compile-time** parameter (it participates in the
`_get_compiled` cache key), not a runtime branch:

```python
Buffer(S, H, K, E, R, alloc_policy="builtin")   # default; upstream fast path
Buffer(..., alloc_policy="mlb:greedy")          # injected from MLB
Buffer(..., alloc_policy="torch")               # host fallback, for development
```

- `"builtin"` — the current code, unchanged, always the default. The upstream
  fast path is never displaced by this work.
- `"mlb:<name>"` — a `@cute.jit` device function supplied out-of-tree and
  inlined between the existing `grid_sync` barriers. One kernel launch, no added
  overhead.
- `"torch"` — runs steps 1–2, copies the exchanged `tpe` to host, calls a plain
  PyTorch policy, copies `alloc` back, runs step 4. Slow by design; it makes
  policy development a pure-Python loop and gives the differential test its
  oracle.

### Device-function signature

```python
@cute.jit
def alloc_policy(group_tokens, z, alloc, meta, R, E, epn, CAP, bar, num_sms, tid) -> None:
    """Read group_tokens (and tpe via meta); write z[R,R] and alloc[R,E].

    Must grid_sync internally if it needs more than one barrier phase.
    Must not write dst, cu_seqlens, experts_to_copy, zero_fill_ranges or
    remote_stats -- those belong to step 4.
    """
```

### Host-function signature

```python
def alloc_policy_torch(tpe: Tensor[R, E], geom: AllocGeometry) -> Tensor[R, E]:
    """Pure function. No CUDA, no process group."""
```

## Conformance

An out-of-tree policy is expected to import the shared suite:

```python
from tests.alloc_contract import check_policy
failures = check_policy(my_policy)   # {(geometry, tpe_kind): [errors]}
assert not failures
```

`check_policy` runs 6 geometries x 8 routing pathologies. Passing it is
necessary but not sufficient: a policy must additionally be compared on the
quality metrics above, and — if it targets the fused path — verified to produce
byte-identical downstream tables when it is intended to be equivalent to
`builtin`.
