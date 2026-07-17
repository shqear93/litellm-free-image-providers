#!/usr/bin/env python3
"""Custom LiteLLM provider for Pollinations.ai's free image-generation API.

Pollinations.ai exposes a single, unauthenticated, synchronous endpoint
(confirmed live against the real API during implementation of this file --
`curl https://image.pollinations.ai/prompt/...` returned a real
`image/jpeg` body with HTTP 200 and no credentials of any kind):

    GET https://image.pollinations.ai/prompt/{url-encoded-prompt}
        ?model=flux&width=1024&height=1024&seed=...&nologo=...&enhance=...&private=...

This is a GET request, not a POST, and the response body is raw image bytes
(a real `image/jpeg` etc.) rather than a JSON envelope -- there is no
`{"artifacts": [...]}` wrapper to parse. This handler reads the raw bytes,
base64-encodes them itself, and wraps that into the OpenAI-compatible
`litellm.types.utils.ImageResponse` (`data: [{"b64_json": "..."}]`) shape
LiteLLM's `/images/generations` proxy endpoint returns to callers.

Reference: https://github.com/pollinations/pollinations/blob/master/APIDOCS.md
"""
from __future__ import annotations

import base64
from typing import Optional
from urllib.parse import quote

import httpx

from litellm.llms.custom_llm import CustomLLM, CustomLLMError
from litellm.types.utils import ImageObject, ImageResponse


POLLINATIONS_API_BASE = "https://image.pollinations.ai/prompt"

# Query params Pollinations' documented image API understands. Anything the
# caller passes via optional_params outside this set is dropped rather than
# forwarded blindly.
_POLLINATIONS_QUERY_FIELDS = (
    "model",
    "width",
    "height",
    "seed",
    "nologo",
    "enhance",
    "private",
)

# Pollinations documents ~1 request/15s for anonymous usage, but that's a
# rate limit, not a per-request latency figure -- actual generation for a
# single request (confirmed live: a 512x512 "flux"-model request completed
# in a few seconds) is fast. 60s gives ample headroom for slower prompts/
# larger sizes while still failing well before a caller's own outer
# deployment-level timeout, so a stuck request here can't by itself block a
# fallback chain from ever giving up.
DEFAULT_TIMEOUT_SECONDS = 60.0

# Pollinations is an external, unauthenticated third-party service, so an
# oversized/misbehaving response is a real (if low-probability) memory-
# exhaustion vector: content-length is checked first where present, and the
# actual downloaded byte length is checked too since content-length can be
# absent or wrong.
MAX_RESPONSE_BYTES = 20 * 1024 * 1024  # 20MB, generous for a generated image


class PollinationsImageCustom(CustomLLM):
    def _build_request(self, prompt: str, optional_params: dict) -> tuple[str, dict]:
        url = f"{POLLINATIONS_API_BASE}/{quote(prompt, safe='')}"
        params: dict = {}
        for field in _POLLINATIONS_QUERY_FIELDS:
            if field in optional_params and optional_params[field] is not None:
                params[field] = optional_params[field]
        return url, params

    def _parse_response(self, response: httpx.Response, model_response: ImageResponse) -> ImageResponse:
        content_type = response.headers.get("content-type", "")
        if not content_type.startswith("image/"):
            raise CustomLLMError(
                status_code=502,
                message=(
                    "Pollinations image generation returned non-image content-type "
                    f"{content_type!r}: {response.text[:500]!r}"
                ),
            )

        content_length = response.headers.get("content-length")
        if content_length is not None:
            try:
                if int(content_length) > MAX_RESPONSE_BYTES:
                    raise CustomLLMError(
                        status_code=502,
                        message=(
                            "Pollinations image generation response exceeds maximum allowed "
                            f"size: {content_length} > {MAX_RESPONSE_BYTES} bytes"
                        ),
                    )
            except ValueError:
                pass

        image_bytes = response.content
        if not image_bytes:
            raise CustomLLMError(
                status_code=502,
                message="Pollinations image generation returned an empty response body",
            )
        if len(image_bytes) > MAX_RESPONSE_BYTES:
            raise CustomLLMError(
                status_code=502,
                message=(
                    "Pollinations image generation response exceeds maximum allowed size: "
                    f"{len(image_bytes)} > {MAX_RESPONSE_BYTES} bytes"
                ),
            )

        b64_json = base64.b64encode(image_bytes).decode("ascii")
        model_response.data = [ImageObject(b64_json=b64_json)]
        return model_response

    def _raise_for_transport_error(self, error: Exception) -> None:
        raise CustomLLMError(
            status_code=500,
            message=f"Pollinations image generation request failed: {error}",
        ) from error

    def _raise_for_status_error(self, response: httpx.Response) -> None:
        if response.status_code >= 400:
            raise CustomLLMError(
                status_code=response.status_code,
                message=(
                    f"Pollinations image generation failed: status={response.status_code} "
                    f"body={response.text[:2000]}"
                ),
            )

    def image_generation(
        self,
        model: str,
        prompt: str,
        api_key: Optional[str],
        api_base: Optional[str],
        model_response: ImageResponse,
        optional_params: dict,
        logging_obj,
        timeout: Optional[float] = None,
        client=None,
    ) -> ImageResponse:
        url, params = self._build_request(prompt, optional_params)

        logging_obj.pre_call(
            input=prompt,
            api_key=api_key,
            additional_args={"complete_input_dict": params, "api_base": url},
        )

        try:
            with httpx.Client(timeout=timeout or DEFAULT_TIMEOUT_SECONDS) as http_client:
                response = http_client.get(url, params=params)
        except httpx.HTTPError as error:
            self._raise_for_transport_error(error)

        self._raise_for_status_error(response)
        result = self._parse_response(response, model_response)

        logging_obj.post_call(
            input=prompt,
            api_key=api_key,
            additional_args={"complete_input_dict": params},
            original_response=result.model_dump() if hasattr(result, "model_dump") else result,
        )
        return result

    async def aimage_generation(
        self,
        model: str,
        prompt: str,
        model_response: ImageResponse,
        api_key: Optional[str],
        api_base: Optional[str],
        optional_params: dict,
        logging_obj,
        timeout: Optional[float] = None,
        client=None,
    ) -> ImageResponse:
        url, params = self._build_request(prompt, optional_params)

        logging_obj.pre_call(
            input=prompt,
            api_key=api_key,
            additional_args={"complete_input_dict": params, "api_base": url},
        )

        try:
            async with httpx.AsyncClient(timeout=timeout or DEFAULT_TIMEOUT_SECONDS) as http_client:
                response = await http_client.get(url, params=params)
        except httpx.HTTPError as error:
            self._raise_for_transport_error(error)

        self._raise_for_status_error(response)
        result = self._parse_response(response, model_response)

        logging_obj.post_call(
            input=prompt,
            api_key=api_key,
            additional_args={"complete_input_dict": params},
            original_response=result.model_dump() if hasattr(result, "model_dump") else result,
        )
        return result
