# Project 04 — Customer Support Intent Router

**Status:** Implemented and locally verified. Thirteen automated tests pass; the offline CLI smoke test passes and all 18 curated synthetic benchmark cases match the expected category and priority. See VALIDATION_REPORT.json.

## Objective

Classify support tickets into an intent category and priority, return a safe draft response, and emit a validated JSON object that can be passed to a support queue or reviewed by an agent.

## Features

- Nine intent categories: account access, billing/payment, cancellation, delivery status, refund/return, security/privacy, technical issue, product question and general inquiry.
- Four priority levels: low, medium, high and urgent, with escalation for high-risk cases.
- Offline-first deterministic rule router; no API key, network, training data or GPU needed for the default path.
- Pydantic request and response schemas reject malformed fields and unexpected keys.
- Optional OpenAI-compatible chat-completions adapter, response parsing/validation and exponential retry/backoff. It only activates when explicitly requested and an API key is present.
- Curated 18-ticket benchmark covering category and priority behavior.
- Unit tests for categories, security precedence, schema validation, provider parsing, retry behavior and missing credentials.

## Quick start

Python 3.10+ is recommended.

~~~bash
python -m venv .venv
# Windows: .venv/Scripts/activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m src.cli --ticket-id T-1001 --subject "Charged twice" --message "My card was charged twice for the same order."
python -m src.benchmark
~~~

The CLI prints validated JSON with fields: ticket_id, category, priority, confidence, escalation_required, draft_response, and rationale.

### Optional LLM provider

Use .env.example as a reference for the required variables, then export OPENAI_API_KEY in your shell (do not commit an actual secret file). The adapter reads environment variables directly; it does not automatically load a .env file. Then run:

~~~bash
python -m src.cli --provider openai --ticket-id T-1002 --subject "Login issue" --message "I cannot log in to my account."
~~~

The adapter uses OPENAI_MODEL and OPENAI_BASE_URL when configured. The offline router remains the default. Provider use requires network access and may incur API charges; no network calls are made by tests.

## Google Colab

~~~python
!git clone https://github.com/chinthavishnupriya/ml-genai-enterprise-lab.git
%cd /content/ml-genai-enterprise-lab/04-customer-support-intent-router
!python -m pip install -r requirements.txt
!python -m unittest discover -s tests -v
!python -m src.benchmark
!python -m src.cli --ticket-id T-1001 --subject "Charged twice" --message "My card was charged twice for the same order."
~~~

## Verification summary

- Python compilation: PASS
- Unit tests: 13 passed, 0 failed
- Offline CLI smoke test: PASS
- Mocked optional-provider parsing/retry tests: PASS; no real API call was made
- Curated benchmark: 18 cases, 100% category and priority match on this handcrafted set only

The benchmark is a rule-coverage check, not an estimate of production performance. See VALIDATION_REPORT.json for environment details and limitations.

## Safety and limitations

- The offline classifier is rule-based. Its confidence is a heuristic signal, not a calibrated probability; benchmark scores only reflect this curated synthetic set, not real production accuracy.
- Drafts are suggestions for a human support agent. They do not claim a refund, cancellation or other external action has been completed.
- Ticket text is untrusted input. The optional LLM prompt instructs the model to treat it as data, and the result is validated against a strict schema, but prompt injection risk is not eliminated.
- Minimize or redact personal data; never send passwords, one-time passcodes, full card numbers, secrets or confidential support data to a model.
- Before production use, evaluate on a representative, privacy-reviewed labeled set, calibrate priorities, monitor drift and require human review for sensitive or high-impact cases.
