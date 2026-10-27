"""conda_env.py

Locating conda environments for subprocess-isolated tool wrappers.

Most Python annotation tools in the roster pin mutually incompatible versions
of torch, tensorflow, scanpy or numpy, so each gets its own conda environment
and the wrapper launches it as a subprocess rather than importing it. That
makes two paths machine-specific: the conda executable, and the environment
prefix. Both are resolved here instead of being hardcoded in each wrapper.

Override either with an environment variable:

    export SC_BENCH_CONDA_EXE=/opt/miniconda3/condabin/conda
    export SC_BENCH_CONDA_PREFIX=/opt/miniconda3/envs

Environment names are documented per tool in harness/docs/environments/install_notes_DL.txt.
"""

import os
import shutil
from typing import Optional

# Common install locations, tried in order when nothing is configured.
_DEFAULT_PREFIXES = [
    os.path.expanduser("~/miniconda3"),
    os.path.expanduser("~/anaconda3"),
    os.path.expanduser("~/miniforge3"),
    "/opt/miniconda3",
    "/opt/anaconda3",
]


def conda_executable() -> str:
    """Path to the conda binary.

    Honours SC_BENCH_CONDA_EXE, then $PATH, then the usual install locations.
    Falls back to the bare name "conda" so the failure surfaces as a clear
    subprocess error rather than an import-time crash.
    """
    configured = os.environ.get("SC_BENCH_CONDA_EXE")
    if configured:
        return configured

    on_path = shutil.which("conda")
    if on_path:
        return on_path

    for prefix in _DEFAULT_PREFIXES:
        candidate = os.path.join(prefix, "condabin", "conda")
        if os.path.isfile(candidate):
            return candidate

    return "conda"


def conda_env_path(env_name: str) -> Optional[str]:
    """Directory of the named conda environment, or None if it is not present.

    Honours SC_BENCH_CONDA_PREFIX, then CONDA_PREFIX's parent (i.e. the
    environments directory of whichever conda is active), then the usual
    install locations.
    """
    configured = os.environ.get("SC_BENCH_CONDA_PREFIX")
    if configured:
        candidate = os.path.join(configured, env_name)
        return candidate if os.path.isdir(candidate) else None

    search_roots = list(_DEFAULT_PREFIXES)
    active = os.environ.get("CONDA_PREFIX")
    if active:
        # CONDA_PREFIX points at the active env; its parent is envs/
        search_roots.insert(0, os.path.dirname(os.path.dirname(active)))

    for root in search_roots:
        candidate = os.path.join(root, "envs", env_name)
        if os.path.isdir(candidate):
            return candidate

    return None
