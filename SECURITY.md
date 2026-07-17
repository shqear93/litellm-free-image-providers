# Security

## Reporting a vulnerability

Please **do not** open a public GitHub issue for security vulnerabilities.

Instead, use [GitHub's private vulnerability reporting](https://github.com/shqear93/litellm-free-image-providers/security/advisories/new) for this repository, or email the maintainer directly (see the repository owner's GitHub profile).

## Supported versions

Only the latest published version is supported. Please upgrade before reporting an issue.

## Notes on the two providers

- Both Pollinations.ai and AI Horde are external, unauthenticated third-party services. Both providers cap response size at 20MB and validate content-type/shape before trusting a response, since an oversized or misbehaving response from either would otherwise be a memory-exhaustion vector.
- AI Horde's anonymous API key (`0000000000`) is a publicly documented constant, not a secret — it's fine for it to appear in logs, config, or code.
