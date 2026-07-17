import litellm

from litellm_free_image_providers import register_all
from litellm_free_image_providers.registration import (
    AI_HORDE_PROVIDER,
    POLLINATIONS_PROVIDER,
)


def _clear_registered(provider_name: str) -> None:
    litellm.custom_provider_map = [
        item for item in getattr(litellm, "custom_provider_map", []) if item.get("provider") != provider_name
    ]


def test_register_all_registers_both_providers_exactly_once():
    _clear_registered(POLLINATIONS_PROVIDER)
    _clear_registered(AI_HORDE_PROVIDER)

    register_all()
    register_all()  # idempotency check

    providers = [item["provider"] for item in litellm.custom_provider_map]
    assert providers.count(POLLINATIONS_PROVIDER) == 1
    assert providers.count(AI_HORDE_PROVIDER) == 1
