import httpx
import pytest
import respx
from litellm.llms.custom_llm import CustomLLMError
from litellm.types.utils import ImageResponse

from litellm_free_image_providers.ai_horde import (
    AI_HORDE_API_BASE,
    AiHordeImageCustom,
)


class FakeLoggingObj:
    def pre_call(self, **kwargs):
        pass

    def post_call(self, **kwargs):
        pass


@pytest.fixture
def provider():
    return AiHordeImageCustom()


@pytest.fixture
def logging_obj():
    return FakeLoggingObj()


def _mock_happy_path(job_id="job-1", image_b64="ZmFrZS13ZWJw"):
    respx.post(f"{AI_HORDE_API_BASE}/generate/async").mock(
        return_value=httpx.Response(202, json={"id": job_id, "kudos": 6.0})
    )
    respx.get(f"{AI_HORDE_API_BASE}/generate/check/{job_id}").mock(
        return_value=httpx.Response(200, json={"done": True, "faulted": False, "is_possible": True})
    )
    respx.get(f"{AI_HORDE_API_BASE}/generate/status/{job_id}").mock(
        return_value=httpx.Response(
            200,
            json={
                "done": True,
                "generations": [{"img": image_b64, "censored": False, "id": "gen-1"}],
            },
        )
    )


@respx.mock
def test_image_generation_happy_path(provider, logging_obj):
    _mock_happy_path()

    result = provider.image_generation(
        model="stable_diffusion",
        prompt="a cat",
        api_key=None,
        api_base=None,
        model_response=ImageResponse(),
        optional_params={},
        logging_obj=logging_obj,
    )

    assert result.data[0].b64_json == "ZmFrZS13ZWJw"


@respx.mock
def test_image_generation_uses_anonymous_key_by_default(provider, logging_obj):
    _mock_happy_path()
    provider.image_generation(
        model="stable_diffusion",
        prompt="a cat",
        api_key=None,
        api_base=None,
        model_response=ImageResponse(),
        optional_params={},
        logging_obj=logging_obj,
    )
    submit_request = respx.calls[0].request
    assert submit_request.headers["apikey"] == "0000000000"


@respx.mock
def test_image_generation_raises_on_submit_failure(provider, logging_obj):
    respx.post(f"{AI_HORDE_API_BASE}/generate/async").mock(
        return_value=httpx.Response(400, text="validation error")
    )

    with pytest.raises(CustomLLMError, match="submit failed"):
        provider.image_generation(
            model="stable_diffusion",
            prompt="a cat",
            api_key=None,
            api_base=None,
            model_response=ImageResponse(),
            optional_params={},
            logging_obj=logging_obj,
        )


@respx.mock
def test_image_generation_raises_on_faulted_job(provider, logging_obj):
    job_id = "job-faulted"
    respx.post(f"{AI_HORDE_API_BASE}/generate/async").mock(
        return_value=httpx.Response(202, json={"id": job_id, "kudos": 1.0})
    )
    respx.get(f"{AI_HORDE_API_BASE}/generate/check/{job_id}").mock(
        return_value=httpx.Response(200, json={"done": False, "faulted": True})
    )

    with pytest.raises(CustomLLMError, match="faulted"):
        provider.image_generation(
            model="stable_diffusion",
            prompt="a cat",
            api_key=None,
            api_base=None,
            model_response=ImageResponse(),
            optional_params={},
            logging_obj=logging_obj,
        )


@respx.mock
def test_image_generation_raises_when_not_possible(provider, logging_obj):
    job_id = "job-impossible"
    respx.post(f"{AI_HORDE_API_BASE}/generate/async").mock(
        return_value=httpx.Response(202, json={"id": job_id, "kudos": 1.0})
    )
    respx.get(f"{AI_HORDE_API_BASE}/generate/check/{job_id}").mock(
        return_value=httpx.Response(200, json={"done": False, "faulted": False, "is_possible": False})
    )

    with pytest.raises(CustomLLMError, match="not fulfillable"):
        provider.image_generation(
            model="stable_diffusion",
            prompt="a cat",
            api_key=None,
            api_base=None,
            model_response=ImageResponse(),
            optional_params={},
            logging_obj=logging_obj,
        )


@respx.mock
def test_image_generation_raises_when_all_censored(provider, logging_obj):
    job_id = "job-censored"
    respx.post(f"{AI_HORDE_API_BASE}/generate/async").mock(
        return_value=httpx.Response(202, json={"id": job_id, "kudos": 1.0})
    )
    respx.get(f"{AI_HORDE_API_BASE}/generate/check/{job_id}").mock(
        return_value=httpx.Response(200, json={"done": True, "faulted": False, "is_possible": True})
    )
    respx.get(f"{AI_HORDE_API_BASE}/generate/status/{job_id}").mock(
        return_value=httpx.Response(
            200,
            json={"done": True, "generations": [{"img": "x", "censored": True, "id": "gen-1"}]},
        )
    )

    with pytest.raises(CustomLLMError, match="censored"):
        provider.image_generation(
            model="stable_diffusion",
            prompt="a cat",
            api_key=None,
            api_base=None,
            model_response=ImageResponse(),
            optional_params={},
            logging_obj=logging_obj,
        )


@respx.mock
@pytest.mark.asyncio
async def test_aimage_generation_happy_path(provider, logging_obj):
    _mock_happy_path()

    result = await provider.aimage_generation(
        model="stable_diffusion",
        prompt="a cat",
        model_response=ImageResponse(),
        api_key=None,
        api_base=None,
        optional_params={},
        logging_obj=logging_obj,
    )

    assert result.data[0].b64_json == "ZmFrZS13ZWJw"


def test_build_submit_body_defaults_models_from_model_arg(provider):
    body = provider._build_submit_body("stable_diffusion", "a cat", {})
    assert body["models"] == ["stable_diffusion"]
    assert body["r2"] is False
    assert body["nsfw"] is False


def test_build_submit_body_only_forwards_allow_listed_param_fields(provider):
    body = provider._build_submit_body("stable_diffusion", "a cat", {"width": 512, "unknown_field": "x"})
    assert body["params"] == {"width": 512}
