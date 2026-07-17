# litellm-free-image-providers

[![CI](https://github.com/shqear93/litellm-free-image-providers/actions/workflows/ci.yml/badge.svg)](https://github.com/shqear93/litellm-free-image-providers/actions/workflows/ci.yml)
[![Release Please](https://github.com/shqear93/litellm-free-image-providers/actions/workflows/release-please.yml/badge.svg)](https://github.com/shqear93/litellm-free-image-providers/actions/workflows/release-please.yml)
[![PyPI version](https://img.shields.io/pypi/v/litellm-free-image-providers.svg)](https://pypi.org/project/litellm-free-image-providers/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/python-%3E%3D3.10-brightgreen)](https://www.python.org)

Custom [LiteLLM](https://github.com/BerriAI/litellm) providers for two genuinely free image-generation APIs that LiteLLM has no built-in support for:

- **[Pollinations.ai](https://pollinations.ai)** — a fast, unauthenticated, synchronous API. Good as a primary provider.
- **[AI Horde](https://aihorde.net)** — a free, crowdsourced-worker API (submit a job, poll until a volunteer worker finishes it). Good as a fallback — it hides the submit-then-poll dance behind the same blocking `image_generation`/`aimage_generation` contract LiteLLM expects, so callers of `/images/generations` see one call either way.

Both are commonly wired as a primary/fallback pair (Pollinations first, AI Horde behind it) in LiteLLM's router `fallbacks` config.

## Why this exists

LiteLLM's `/images/generations` endpoint has no native provider for either of these APIs. LiteLLM does support registering custom providers via `litellm.custom_provider_map` + the `litellm.llms.custom_llm.CustomLLM` base class — that's a first-class, public LiteLLM extension mechanism, not a fork. This package packages that up as an installable dependency instead of code you'd otherwise have to copy-paste into your own deployment.

## Installation

```bash
pip install litellm-free-image-providers
```

## Usage

LiteLLM only picks up custom providers registered *before* a request for that provider is dispatched — typically via code that runs at process startup. The standard way to do that without touching LiteLLM's own source is a `sitecustomize.py` on the Python path: the interpreter's `site` module imports it automatically at startup, before your main program runs.

`sitecustomize.py`:

```python
from litellm_free_image_providers import register_all

register_all()
```

Or register just one:

```python
from litellm_free_image_providers import register_pollinations

register_pollinations()
```

Then reference them in your LiteLLM `config.yaml` like any other custom provider — the model string must be `<provider>/<anything>` (LiteLLM's async image-generation path needs the `custom_llm_provider` prefix baked into the model string itself, not just set via `litellm_params.custom_llm_provider`, since it doesn't forward that field through on the image-generation path):

```yaml
model_list:
  - model_name: pollinations-image
    litellm_params:
      model: pollinations_image_custom/flux
      custom_llm_provider: pollinations_image_custom
      timeout: 45
      num_retries: 0

  - model_name: ai-horde-image
    litellm_params:
      model: ai_horde_image_custom/stable_diffusion
      custom_llm_provider: ai_horde_image_custom
      timeout: 180
      num_retries: 0

router_settings:
  fallbacks:
    - pollinations-image: ["pollinations-image", "ai-horde-image"]
```

`num_retries: 0` on both is deliberate: LiteLLM's router retries the *same* deployment before ever falling back to the next one in the chain, so without this a failing `pollinations-image` call would burn a second attempt before AI Horde is even tried — worse latency for no benefit once a request has already failed once.

## Development

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## License

[MIT](./LICENSE)
