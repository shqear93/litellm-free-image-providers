# Contributing

Thanks for considering a contribution to `litellm-free-image-providers`.

## Getting set up

This project uses [mise](https://mise.jdx.dev) to pin the Python version. Install it, then:

```bash
git clone https://github.com/shqear93/litellm-free-image-providers.git
cd litellm-free-image-providers
mise install
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

Without mise, any Python 3.10+ works the same.

## Making changes

- Keep changes focused — one logical change per PR.
- Add or update tests for any behavior change. Both providers talk to real external APIs with real quirks (async submit-then-poll, censorship handling, size limits) — untested changes here are risky.
- Run `pytest` before opening a PR.
- If you change the request/response handling for either provider, verify against the real API if you can — both modules' docstrings document exact confirmed-live request/response shapes; keep that accurate.

## Commit messages

This project uses [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `docs:`, `test:`, `chore:`, etc.) — releases and the changelog are generated automatically from these via [release-please](https://github.com/googleapis/release-please).

## Reporting bugs

Please include:

- What you expected vs. what happened.
- Relevant log output (redact API keys — though note the AI Horde anonymous key `0000000000` is a public constant, not a secret).
- Your `litellm-free-image-providers` and `litellm` versions.

## Security

Please don't open a public issue for security vulnerabilities. Email the maintainer instead — see the repository owner's profile for contact info.
