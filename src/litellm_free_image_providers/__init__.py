"""Custom LiteLLM providers for two free image-generation APIs:
Pollinations.ai (fast, synchronous) and AI Horde (crowdsourced, async
submit-then-poll) -- commonly wired as a primary/fallback pair.
"""
from importlib.metadata import PackageNotFoundError, version

from .ai_horde import AiHordeImageCustom
from .pollinations import PollinationsImageCustom
from .registration import (
    AI_HORDE_PROVIDER,
    POLLINATIONS_PROVIDER,
    register_ai_horde,
    register_all,
    register_pollinations,
)

try:
    __version__ = version("litellm-free-image-providers")
except PackageNotFoundError:  # not installed (e.g. running from a source checkout)
    __version__ = "0.0.0+unknown"

__all__ = [
    "AiHordeImageCustom",
    "PollinationsImageCustom",
    "AI_HORDE_PROVIDER",
    "POLLINATIONS_PROVIDER",
    "register_ai_horde",
    "register_all",
    "register_pollinations",
]
