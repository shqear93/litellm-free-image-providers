"""Registers these custom providers into LiteLLM's `custom_provider_map`.

LiteLLM only picks up entries in `litellm.custom_provider_map` if they're
registered before a request for that provider is dispatched -- typically
via code that runs at process startup. A common way to do that without
touching LiteLLM's own source is a `sitecustomize.py` on the Python path:
the interpreter's `site` module imports it automatically at startup, before
your main program runs. Example `sitecustomize.py`:

    from litellm_free_image_providers import register_all
    register_all()

Call `register_pollinations()` / `register_ai_horde()` individually if you
only want one of the two providers.
"""
from __future__ import annotations

import litellm
from litellm.utils import custom_llm_setup

from .ai_horde import AiHordeImageCustom
from .pollinations import PollinationsImageCustom


POLLINATIONS_PROVIDER = "pollinations_image_custom"
AI_HORDE_PROVIDER = "ai_horde_image_custom"

# The model string each provider expects after its "<provider>/" prefix is
# stripped by LiteLLM's router. Used only to register $0 cost-map entries
# (see _register_cost_map_entries) so cost tracking/Langfuse spend reporting
# is accurate for these genuinely-free APIs, and to suppress a purely
# cosmetic (non-fatal) "This model isn't mapped yet" log LiteLLM's Router
# otherwise emits on every call when `router_settings.enable_pre_call_checks:
# true` is set.
POLLINATIONS_DEFAULT_MODEL = "flux"
AI_HORDE_DEFAULT_MODEL = "stable_diffusion"


def _register_provider(provider_name: str, handler) -> None:
    if not any(
        item.get("provider") == provider_name
        for item in getattr(litellm, "custom_provider_map", [])
    ):
        litellm.custom_provider_map.append(  # type: ignore[attr-defined]
            {
                "provider": provider_name,
                "custom_handler": handler,
            }
        )


def _register_cost_map_entries() -> None:
    """In short: setting
    `input_cost_per_token`/`output_cost_per_token` on a router deployment's
    `litellm_params` does NOT reliably register a cost-map entry under the
    key LiteLLM's pre-call checks actually look up at call time for a
    "<custom_llm_provider>/<model>"-shaped model string. Registering
    directly here, under the exact keys these providers use, sidesteps that
    entirely. Both APIs are genuinely free, so a $0 entry is also accurate
    for spend tracking, not just a log-suppression hack.
    """
    litellm.register_model(
        {
            f"{POLLINATIONS_PROVIDER}/{POLLINATIONS_DEFAULT_MODEL}": {
                "input_cost_per_token": 0,
                "output_cost_per_token": 0,
                "litellm_provider": POLLINATIONS_PROVIDER,
                "mode": "image_generation",
            },
            f"{AI_HORDE_PROVIDER}/{AI_HORDE_DEFAULT_MODEL}": {
                "input_cost_per_token": 0,
                "output_cost_per_token": 0,
                "litellm_provider": AI_HORDE_PROVIDER,
                "mode": "image_generation",
            },
        }
    )


def register_pollinations() -> None:
    """Registers the Pollinations.ai custom provider into LiteLLM."""
    _register_provider(POLLINATIONS_PROVIDER, PollinationsImageCustom())
    custom_llm_setup()
    _register_cost_map_entries()


def register_ai_horde() -> None:
    """Registers the AI Horde custom provider into LiteLLM."""
    _register_provider(AI_HORDE_PROVIDER, AiHordeImageCustom())
    custom_llm_setup()
    _register_cost_map_entries()


def register_all() -> None:
    """Registers both Pollinations.ai and AI Horde custom providers."""
    _register_provider(POLLINATIONS_PROVIDER, PollinationsImageCustom())
    _register_provider(AI_HORDE_PROVIDER, AiHordeImageCustom())
    custom_llm_setup()
    _register_cost_map_entries()
