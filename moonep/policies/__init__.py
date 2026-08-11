"""Built-in example allocation policies.

Importing this package registers them, so ``Buffer(..., alloc_policy=...)``
accepts their names. Out-of-tree policies register themselves the same way via
``moonep.alloc_policy.register_alloc_policy``.
"""

from moonep.alloc_policy import register_alloc_policy

from .reference_greedy import reference_greedy

register_alloc_policy("reference_greedy", reference_greedy)

__all__ = ["reference_greedy"]
