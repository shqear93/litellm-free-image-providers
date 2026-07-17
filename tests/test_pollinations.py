import base64

import httpx
import pytest
import respx
from litellm.llms.custom_llm import CustomLLMError
from litellm.types.utils import ImageResponse

from litellm_free_image_providers.pollinations import (
    POLLINATIONS_API_BASE,
    PollinationsImageCustom,
)


class FakeLoggingObj:
    def pre_call(self, **kwargs):
        pass

    def post_call(self, **kwargs):
        pass


@pytest.fixture
def provider():
    return PollinationsImageCustom()


@pytest.fixture
def logging_obj():
    return FakeLoggingObj()


@respx.mock
def test_image_generation_returns_b64_json(provider, logging_obj):
    image_bytes = b"\xff\xd8\xff\xe0fakejpegbytes"
    respx.get(url__startswith=POLLINATIONS_API_BASE).mock(
        return_value=httpx.Response(200, content=image_bytes, headers={"content-type": "image/jpeg"})
    )

    result = provider.image_generation(
        model="flux",
        prompt="a cat",
        api_key=None,
        api_base=None,
        model_response=ImageResponse(),
        optional_params={"model": "flux", "width": 512},
        logging_obj=logging_obj,
    )

    assert result.data[0].b64_json == base64.b64encode(image_bytes).decode("ascii")


@respx.mock
def test_image_generation_rejects_non_image_content_type(provider, logging_obj):
    respx.get(url__startswith=POLLINATIONS_API_BASE).mock(
        return_value=httpx.Response(200, text="not an image", headers={"content-type": "text/plain"})
    )

    with pytest.raises(CustomLLMError, match="non-image content-type"):
        provider.image_generation(
            model="flux",
            prompt="a cat",
            api_key=None,
            api_base=None,
            model_response=ImageResponse(),
            optional_params={},
            logging_obj=logging_obj,
        )


@respx.mock
def test_image_generation_rejects_empty_body(provider, logging_obj):
    respx.get(url__startswith=POLLINATIONS_API_BASE).mock(
        return_value=httpx.Response(200, content=b"", headers={"content-type": "image/jpeg"})
    )

    with pytest.raises(CustomLLMError, match="empty response body"):
        provider.image_generation(
            model="flux",
            prompt="a cat",
            api_key=None,
            api_base=None,
            model_response=ImageResponse(),
            optional_params={},
            logging_obj=logging_obj,
        )


@respx.mock
def test_image_generation_rejects_oversized_response(provider, logging_obj):
    respx.get(url__startswith=POLLINATIONS_API_BASE).mock(
        return_value=httpx.Response(
            200,
            content=b"x",
            headers={"content-type": "image/jpeg", "content-length": str(21 * 1024 * 1024)},
        )
    )

    with pytest.raises(CustomLLMError, match="exceeds maximum allowed size"):
        provider.image_generation(
            model="flux",
            prompt="a cat",
            api_key=None,
            api_base=None,
            model_response=ImageResponse(),
            optional_params={},
            logging_obj=logging_obj,
        )


@respx.mock
def test_image_generation_raises_on_http_error_status(provider, logging_obj):
    respx.get(url__startswith=POLLINATIONS_API_BASE).mock(
        return_value=httpx.Response(500, text="upstream error")
    )

    with pytest.raises(CustomLLMError, match="status=500"):
        provider.image_generation(
            model="flux",
            prompt="a cat",
            api_key=None,
            api_base=None,
            model_response=ImageResponse(),
            optional_params={},
            logging_obj=logging_obj,
        )


@respx.mock
def test_image_generation_raises_on_transport_error(provider, logging_obj):
    respx.get(url__startswith=POLLINATIONS_API_BASE).mock(side_effect=httpx.ConnectError("boom"))

    with pytest.raises(CustomLLMError, match="request failed"):
        provider.image_generation(
            model="flux",
            prompt="a cat",
            api_key=None,
            api_base=None,
            model_response=ImageResponse(),
            optional_params={},
            logging_obj=logging_obj,
        )


@respx.mock
@pytest.mark.asyncio
async def test_aimage_generation_returns_b64_json(provider, logging_obj):
    image_bytes = b"\xff\xd8\xff\xe0fakejpegbytes"
    respx.get(url__startswith=POLLINATIONS_API_BASE).mock(
        return_value=httpx.Response(200, content=image_bytes, headers={"content-type": "image/jpeg"})
    )

    result = await provider.aimage_generation(
        model="flux",
        prompt="a cat",
        model_response=ImageResponse(),
        api_key=None,
        api_base=None,
        optional_params={},
        logging_obj=logging_obj,
    )

    assert result.data[0].b64_json == base64.b64encode(image_bytes).decode("ascii")


def test_build_request_only_forwards_allow_listed_fields(provider):
    url, params = provider._build_request("a cat", {"model": "flux", "width": 512, "unknown_field": "x"})
    assert "unknown_field" not in params
    assert params == {"model": "flux", "width": 512}
    assert url == f"{POLLINATIONS_API_BASE}/a%20cat"
