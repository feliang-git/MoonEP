"""Registry for pluggable allocation policies.

The allocation policy is step 3 of the planning kernel: it turns per-rank
expert loads into ``alloc[R, E]``, the number of global tokens of each expert
computed on each destination rank. Everything downstream is a pure function of
``alloc``. See ``docs/alloc_policy_contract.md`` for the invariants a policy
must satisfy -- in particular invariant 4, which is load-bearing for buffer
sizing and is easy to violate.

Policies are selected by name and resolved at *compile* time: the name is part
of the ``_get_compiled`` cache key, so the choice costs nothing at runtime and
the builtin path emits the same instructions it always did.

Registering an out-of-tree policy (e.g. from MLB):

    from moonep.alloc_policy import register_alloc_policy

    @cute.jit
    def my_policy(group_tokens, z, alloc, tpe_cumsum, *, R, E, epn, CAP,
                  bar, num_sms, num_threads, pid, tid, scratch):
        ...

    register_alloc_policy("mlb:waterfill", my_policy)
"""

BUILTIN = "builtin"

_REGISTRY: dict[str, object] = {}


def register_alloc_policy(name: str, fn) -> None:
    """Register a ``@cute.jit`` device function under ``name``.

    The function is inlined into the planning kernel between the existing
    grid-wide barriers. It must write ``z[R, R]`` and ``alloc[R, E]`` and must
    not touch ``dst``, ``cu_seqlens``, ``experts_to_copy``,
    ``zero_fill_ranges`` or ``remote_stats`` -- those belong to step 4.
    """
    if name == BUILTIN:
        raise ValueError(f"{BUILTIN!r} is reserved for MoonEP's fused policy")
    if not isinstance(name, str) or not name:
        raise ValueError(f"policy name must be a non-empty str, got {name!r}")
    if name in _REGISTRY and _REGISTRY[name] is not fn:
        raise ValueError(
            f"alloc policy {name!r} is already registered to a different function"
        )
    _REGISTRY[name] = fn


def resolve_alloc_policy(name: str):
    """Return the device function for ``name``, or None for the builtin path.

    None is the sentinel the kernel uses to select its original fused code, so
    the default path stays a compile-time no-op rather than a dispatch.
    """
    if name is None or name == BUILTIN:
        return None
    try:
        return _REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted([BUILTIN, *_REGISTRY])) or BUILTIN
        raise KeyError(
            f"unknown alloc policy {name!r}; registered: {known}"
        ) from None


def available_alloc_policies() -> list[str]:
    return sorted([BUILTIN, *_REGISTRY])
