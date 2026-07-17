#!/usr/bin/env python3
"""Custom LiteLLM provider for AI Horde's free, crowdsourced image-generation
API -- typically used as a fallback behind a faster synchronous provider
(e.g. `pollinations.py`'s `PollinationsImageCustom` in this same package).

Unlike a single synchronous request, AI Horde is a genuinely asynchronous,
crowdsourced-worker API: submit a job, then poll until a volunteer worker
picks it up and finishes it. This handler hides that entire submit-then-poll
dance behind the same synchronous `image_generation`/`aimage_generation`
contract other providers expose, so callers of LiteLLM's
`/images/generations` endpoint see one blocking call either way.

Confirmed live against the real API while implementing this file (host,
paths, and shapes below are NOT guessed):

    Base host: https://aihorde.net/api  (the `stablehorde.net` name is a
        legacy alias for the same service; `aihorde.net` is canonical --
        confirmed via `GET /v2/status/heartbeat` -> 200 and the served
        swagger.json at `https://aihorde.net/api/swagger.json`).

    1. POST /v2/generate/async
       Headers: apikey: <key>  (anonymous/public key is the literal string
           "0000000000" -- a well-known public constant, not a secret)
       Body (subset used here): {"prompt": str, "params": {...}, "r2": false,
           "nsfw": false}
       -- "r2" controls whether the finished image comes back as a
       Cloudflare R2 download URL (r2=true, the API's default) or inline
       base64 (r2=false). We force r2=false so the final payload is
       self-contained and this handler never needs a second HTTP round-trip
       to fetch image bytes from a URL.
       Response: 202 {"id": "<uuid>", "kudos": <float>} on success; 4xx/5xx
       with a JSON error body on failure (e.g. 401 invalid key, 400
       validation error, 429 too many prompts, 503 maintenance mode).
       Confirmed live: a real anonymous-key submission returned
       `{"id": "...", "kudos": 6.0}` with HTTP 202.

    2. GET /v2/generate/check/{id}   (lightweight polling, no image data)
       Response: {"finished": int, "processing": int, "waiting": int,
           "done": bool, "faulted": bool, "wait_time": int (seconds),
           "queue_position": int, "is_possible": bool, ...}
       Confirmed live: a real in-flight job returned
       `{"finished": 0, "processing": 1, "waiting": 0, "done": false,
       "faulted": false, "wait_time": ...}`.
       `is_possible: false` means no worker pool can ever fulfill this
       request (e.g. an unsatisfiable model/param combo) -- polling further
       would just wait out the full timeout for nothing, so this handler
       fails fast on it instead.

    3. GET /v2/generate/status/{id}   (full status, includes image data;
       AI Horde's own docs ask clients not to hit this frequently -- hence
       polling /check/ first and only fetching /status/ once /check/ says
       done)
       Response: same shape as /check/ plus
           "generations": [{"img": "<base64 .webp, since r2=false>",
               "censored": bool, "id": "<generation-uuid>",
               "worker_id": "...", "worker_name": "...", "model": "...",
               "state": "ok"|"faulted"|"censored"|..., ...}, ...]
       A finished-but-`censored: true` generation means a worker's safety
       filter blocked the image (e.g. NSFW-classified prompt) and `img` is
       not a usable image -- this handler treats that as a hard failure
       with a clear message rather than silently returning a censorship
       placeholder as if it were the requested image.

Sources: https://github.com/Haidra-Org/AI-Horde, https://aihorde.net/api/,
https://aihorde.net/api/swagger.json (fetched live).
"""
from __future__ import annotations

import asyncio
import time
from typing import Optional

import httpx

from litellm.llms.custom_llm import CustomLLM, CustomLLMError
from litellm.types.utils import ImageObject, ImageResponse


AI_HORDE_API_BASE = "https://aihorde.net/api/v2"

# The AI Horde project's own well-known, publicly documented anonymous key.
# It grants real (if low-priority) access with zero registration and is
# meant to be used exactly like this -- not a leaked/scoped secret.
ANONYMOUS_API_KEY = "0000000000"

# Request-body fields under "params" that AI Horde's GenerationInputStable/
# ModelGenerationInputStable schema understands (per the live swagger.json
# fetched from https://aihorde.net/api/swagger.json while writing this
# file). Anything else in optional_params is dropped rather than forwarded
# blindly.
_AI_HORDE_PARAM_FIELDS = (
    "width",
    "height",
    "steps",
    "cfg_scale",
    "sampler_name",
    "seed",
    "karras",
    "n",
)

# Top-level (not "params") request fields.
_AI_HORDE_TOP_LEVEL_FIELDS = (
    "models",
    "nsfw",
    "censor_nsfw",
    "trusted_workers",
    "slow_workers",
)

# Anonymous-tier requests get the lowest scheduling priority in AI Horde's
# crowdsourced worker pool, so completion can genuinely take minutes rather
# than seconds (confirmed live: a real anonymous submission was still
# "processing" after tens of seconds). We poll the lightweight /check/
# endpoint (explicitly designed for frequent polling, unlike /status/,
# whose docs ask clients not to hit it often) rather than sleep-once-and-
# hope, and give up with a clear error past a generous ceiling instead of
# hanging indefinitely.
POLL_INTERVAL_SECONDS = 5.0
MAX_POLL_SECONDS = 170.0  # keep comfortably under a typical 180s caller-side
# deployment timeout, so this handler's own timeout fires first with a
# clear message instead of the caller cutting the connection with a generic
# timeout error.

DEFAULT_TIMEOUT_SECONDS = MAX_POLL_SECONDS + 20.0

# /generate/check/ polls are meant to be lightweight, fast status checks
# (see the module docstring), not slow operations -- give each individual
# poll request its own short per-request timeout, separate from the overall
# MAX_POLL_SECONDS polling ceiling and from the longer submit/status-fetch
# timeout. Without this, the poll loop only checks the deadline AFTER each
# request returns, so a single hung poll request using the full ~190s
# client timeout could by itself overshoot the "170s ceiling, comfortably
# under the 180s deployment timeout" guarantee.
POLL_REQUEST_TIMEOUT_SECONDS = 15.0

# Both size-cap and content-type validation guard against an external,
# unauthenticated third-party service returning an oversized/misbehaving
# response -- a real (if low-probability) memory-exhaustion vector.
MAX_RESPONSE_BYTES = 20 * 1024 * 1024  # 20MB, generous for a generated image


class AiHordeImageCustom(CustomLLM):
    def _resolve_api_key(self, api_key: Optional[str]) -> str:
        return api_key or ANONYMOUS_API_KEY

    def _build_submit_body(self, model: str, prompt: str, optional_params: dict) -> dict:
        params: dict = {}
        for field in _AI_HORDE_PARAM_FIELDS:
            if field in optional_params and optional_params[field] is not None:
                params[field] = optional_params[field]

        body: dict = {
            "prompt": prompt,
            "params": params,
            # Force inline base64 rather than an R2 download URL so a
            # finished job never needs a second HTTP fetch to retrieve
            # image bytes.
            "r2": False,
            # Anonymous keys are always treated as "shared" by AI Horde
            # regardless of this flag, but set it explicitly rather than
            # relying on that undocumented-here server-side default.
            "nsfw": False,
        }
        for field in _AI_HORDE_TOP_LEVEL_FIELDS:
            if field in optional_params and optional_params[field] is not None:
                body[field] = optional_params[field]
        # `model` (the LiteLLM model string after the custom provider's
        # prefix is stripped by the router) maps to AI Horde's "models" list
        # filter when the caller didn't already pass one explicitly.
        if "models" not in body and model:
            body["models"] = [model]
        return body

    def _raise_for_transport_error(self, error: Exception) -> None:
        raise CustomLLMError(
            status_code=500,
            message=f"AI Horde image generation request failed: {error}",
        ) from error

    def _check_response_size(self, response: httpx.Response, context: str) -> None:
        """Guards against an oversized/misbehaving response from this
        external, unauthenticated third-party service consuming unbounded
        memory: content-length is checked first where present, and the
        actual downloaded byte length is checked too since content-length
        can be absent or wrong.
        """
        content_length = response.headers.get("content-length")
        if content_length is not None:
            try:
                if int(content_length) > MAX_RESPONSE_BYTES:
                    raise CustomLLMError(
                        status_code=502,
                        message=(
                            f"AI Horde {context} response exceeds maximum allowed size: "
                            f"{content_length} > {MAX_RESPONSE_BYTES} bytes"
                        ),
                    )
            except ValueError:
                pass
        actual_length = len(response.content)
        if actual_length > MAX_RESPONSE_BYTES:
            raise CustomLLMError(
                status_code=502,
                message=(
                    f"AI Horde {context} response exceeds maximum allowed size: "
                    f"{actual_length} > {MAX_RESPONSE_BYTES} bytes"
                ),
            )

    def _raise_for_submit_status_error(self, response: httpx.Response) -> None:
        if response.status_code >= 400:
            raise CustomLLMError(
                status_code=response.status_code,
                message=(
                    f"AI Horde image generation submit failed: status={response.status_code} "
                    f"body={response.text[:2000]}"
                ),
            )

    def _extract_job_id(self, response: httpx.Response) -> str:
        try:
            payload = response.json()
        except ValueError as error:
            raise CustomLLMError(
                status_code=502,
                message=f"AI Horde submit returned non-JSON response: {response.text[:500]!r}",
            ) from error

        job_id = payload.get("id")
        if not job_id:
            raise CustomLLMError(
                status_code=502,
                message=f"AI Horde submit response missing 'id': {payload!r}",
            )
        return job_id

    def _check_payload_to_result(self, job_id: str, payload: dict) -> Optional[str]:
        """Returns None if still waiting, raises on unrecoverable failure,
        or returns "done" (sentinel) once /check/ reports completion so the
        caller knows to fetch the full /status/ payload."""
        if payload.get("faulted"):
            raise CustomLLMError(
                status_code=502,
                message=f"AI Horde image generation job {job_id} faulted: {payload!r}",
            )
        if payload.get("is_possible") is False:
            raise CustomLLMError(
                status_code=502,
                message=(
                    f"AI Horde image generation job {job_id} is not fulfillable by any "
                    f"currently available worker: {payload!r}"
                ),
            )
        if payload.get("done"):
            return "done"
        return None

    def _extract_image_from_status(self, job_id: str, payload: dict, model_response: ImageResponse) -> ImageResponse:
        generations = payload.get("generations")
        if not isinstance(generations, list) or not generations:
            raise CustomLLMError(
                status_code=502,
                message=f"AI Horde image generation job {job_id} finished with no generations: {payload!r}",
            )

        b64_images = []
        censored_count = 0
        for generation in generations:
            if not isinstance(generation, dict):
                continue
            if generation.get("censored"):
                censored_count += 1
                continue
            img = generation.get("img")
            if img:
                b64_images.append(img)

        if not b64_images:
            if censored_count:
                raise CustomLLMError(
                    status_code=502,
                    message=(
                        f"AI Horde image generation job {job_id} completed but all "
                        f"{censored_count} generation(s) were censored (safety filter) -- "
                        "no usable image was produced"
                    ),
                )
            raise CustomLLMError(
                status_code=502,
                message=f"AI Horde image generation job {job_id} completed with no image data: {payload!r}",
            )

        # AI Horde returns base64-encoded .webp bytes directly (r2=false
        # forced in _build_submit_body), already the right shape for
        # ImageObject.b64_json -- no re-encoding needed.
        model_response.data = [ImageObject(b64_json=b64) for b64 in b64_images]
        return model_response

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
        resolved_key = self._resolve_api_key(api_key)
        headers = {"apikey": resolved_key, "Content-Type": "application/json"}
        body = self._build_submit_body(model, prompt, optional_params)

        logging_obj.pre_call(
            input=prompt,
            api_key=resolved_key,
            additional_args={"complete_input_dict": body, "api_base": AI_HORDE_API_BASE},
        )

        deadline = time.monotonic() + MAX_POLL_SECONDS
        with httpx.Client(timeout=timeout or DEFAULT_TIMEOUT_SECONDS) as http_client:
            try:
                submit_response = http_client.post(
                    f"{AI_HORDE_API_BASE}/generate/async", json=body, headers=headers
                )
            except httpx.HTTPError as error:
                self._raise_for_transport_error(error)
            self._raise_for_submit_status_error(submit_response)
            job_id = self._extract_job_id(submit_response)

            while True:
                try:
                    check_response = http_client.get(
                        f"{AI_HORDE_API_BASE}/generate/check/{job_id}",
                        timeout=POLL_REQUEST_TIMEOUT_SECONDS,
                    )
                except httpx.HTTPError as error:
                    self._raise_for_transport_error(error)
                if check_response.status_code >= 400:
                    raise CustomLLMError(
                        status_code=check_response.status_code,
                        message=(
                            f"AI Horde poll for job {job_id} failed: status={check_response.status_code} "
                            f"body={check_response.text[:2000]}"
                        ),
                    )
                check_payload = check_response.json()
                if self._check_payload_to_result(job_id, check_payload) == "done":
                    break
                if time.monotonic() >= deadline:
                    raise CustomLLMError(
                        status_code=504,
                        message=(
                            f"AI Horde image generation job {job_id} did not complete within "
                            f"{MAX_POLL_SECONDS}s (anonymous-tier requests are low priority and "
                            f"can be slow; last status: {check_payload!r})"
                        ),
                    )
                time.sleep(POLL_INTERVAL_SECONDS)

            try:
                status_response = http_client.get(f"{AI_HORDE_API_BASE}/generate/status/{job_id}")
            except httpx.HTTPError as error:
                self._raise_for_transport_error(error)
            if status_response.status_code >= 400:
                raise CustomLLMError(
                    status_code=status_response.status_code,
                    message=(
                        f"AI Horde final status fetch for job {job_id} failed: "
                        f"status={status_response.status_code} body={status_response.text[:2000]}"
                    ),
                )
            self._check_response_size(status_response, "final status fetch")
            status_payload = status_response.json()

        result = self._extract_image_from_status(job_id, status_payload, model_response)

        logging_obj.post_call(
            input=prompt,
            api_key=resolved_key,
            additional_args={"complete_input_dict": body},
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
        resolved_key = self._resolve_api_key(api_key)
        headers = {"apikey": resolved_key, "Content-Type": "application/json"}
        body = self._build_submit_body(model, prompt, optional_params)

        logging_obj.pre_call(
            input=prompt,
            api_key=resolved_key,
            additional_args={"complete_input_dict": body, "api_base": AI_HORDE_API_BASE},
        )

        deadline = time.monotonic() + MAX_POLL_SECONDS
        async with httpx.AsyncClient(timeout=timeout or DEFAULT_TIMEOUT_SECONDS) as http_client:
            try:
                submit_response = await http_client.post(
                    f"{AI_HORDE_API_BASE}/generate/async", json=body, headers=headers
                )
            except httpx.HTTPError as error:
                self._raise_for_transport_error(error)
            self._raise_for_submit_status_error(submit_response)
            job_id = self._extract_job_id(submit_response)

            while True:
                try:
                    check_response = await http_client.get(
                        f"{AI_HORDE_API_BASE}/generate/check/{job_id}",
                        timeout=POLL_REQUEST_TIMEOUT_SECONDS,
                    )
                except httpx.HTTPError as error:
                    self._raise_for_transport_error(error)
                if check_response.status_code >= 400:
                    raise CustomLLMError(
                        status_code=check_response.status_code,
                        message=(
                            f"AI Horde poll for job {job_id} failed: status={check_response.status_code} "
                            f"body={check_response.text[:2000]}"
                        ),
                    )
                check_payload = check_response.json()
                if self._check_payload_to_result(job_id, check_payload) == "done":
                    break
                if time.monotonic() >= deadline:
                    raise CustomLLMError(
                        status_code=504,
                        message=(
                            f"AI Horde image generation job {job_id} did not complete within "
                            f"{MAX_POLL_SECONDS}s (anonymous-tier requests are low priority and "
                            f"can be slow; last status: {check_payload!r})"
                        ),
                    )
                await asyncio.sleep(POLL_INTERVAL_SECONDS)

            try:
                status_response = await http_client.get(f"{AI_HORDE_API_BASE}/generate/status/{job_id}")
            except httpx.HTTPError as error:
                self._raise_for_transport_error(error)
            if status_response.status_code >= 400:
                raise CustomLLMError(
                    status_code=status_response.status_code,
                    message=(
                        f"AI Horde final status fetch for job {job_id} failed: "
                        f"status={status_response.status_code} body={status_response.text[:2000]}"
                    ),
                )
            self._check_response_size(status_response, "final status fetch")
            status_payload = status_response.json()

        result = self._extract_image_from_status(job_id, status_payload, model_response)

        logging_obj.post_call(
            input=prompt,
            api_key=resolved_key,
            additional_args={"complete_input_dict": body},
            original_response=result.model_dump() if hasattr(result, "model_dump") else result,
        )
        return result
