# Project 04 — Customer Support Intent Router

**Status:** Scaffold; implementation and tests not yet added.

## Objective
Classify incoming support tickets by category and priority, and return a concise draft response in a validated structured JSON schema.

## Planned components
- Pydantic output schema and rule-based offline demo.
- Optional LLM provider adapter configured through environment variables.
- Retry/backoff, parsing validation and prompt benchmark.
- Tests for malformed model responses and ticket examples.

Never commit API keys, customer emails, or confidential support data. Keep the deterministic offline demo usable without credentials.
