from __future__ import annotations

from .sfmodbo import DEFAULT_PARAMS as SFMODBO_DEFAULT_PARAMS
from .sfmodbo import SFMODBO


DEFAULT_PARAMS = {
    **SFMODBO_DEFAULT_PARAMS,
}


class MODBO(SFMODBO):
    """Multiobjective dung beetle optimizer baseline."""

    LOG_NAME = "MODBO"
