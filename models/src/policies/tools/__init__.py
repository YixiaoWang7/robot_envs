"""
Importable tooling utilities.

Keep heavy logic here (library-style), and keep `scripts/policies/*` as thin CLI wrappers.
"""

from policies.tools.convert_hdf5_to_small_files import split_hdf5_to_shards, split_hdf5s_to_mixed_shards

__all__ = [
    "split_hdf5_to_shards",
    "split_hdf5s_to_mixed_shards",
]

