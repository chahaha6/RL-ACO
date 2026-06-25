"""MODBO baseline entry.

The project already has a strategy-fusion multiobjective DBO implementation in
``sfmodbo.py``.  This module exposes it under the standard MODBO name so main.py
can keep experiment switches and output tags aligned with the paper.
"""

from __future__ import annotations

from .sfmodbo import DEFAULT_PARAMS as SFMODBO_DEFAULT_PARAMS
from .sfmodbo import SFMODBO


DEFAULT_PARAMS = {
    **SFMODBO_DEFAULT_PARAMS,
}


class MODBO(SFMODBO):
    """Multiobjective dung beetle optimizer baseline."""

    LOG_NAME = "MODBO"
