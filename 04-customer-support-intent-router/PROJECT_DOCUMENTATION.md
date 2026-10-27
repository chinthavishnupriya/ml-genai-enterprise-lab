# Project 04 — Customer Support Intent Router

## 1. Project overview

Classify support tickets into an intent, assign a priority, determine whether escalation is required, and generate a draft response. The default path is a deterministic, offline rules engine. An optional OpenAI-compatible chat-completions provider can be selected explicitly; its output is parsed and validated before use.

## 2. Technology stack

| Technology | Purpose |
|---|---|
| Python | Router, provider adapter, benchmark and CLI |
| Pydantic v2 | Strict request/response data contracts and field validation |
| Python regular expressions | Keyword-based intent and severity rules |
| Python standard library `urllib` | Optional HTTP provider requests without a separate client SDK |
| `unittest` / mocked transport | Offline unit and provider-retry tests |

## 3. Routing categories and priority

The router supports nine categories: account access, billing/payment, cancellation, delivery status, refund/return, security/privacy, technical issue, product question and general inquiry. Priority levels are low, medium, high and urgent.

The deterministic rules look for phrase matches in the subject and message. Security/privacy keywords receive category precedence. High/urgent cases are marked for escalation. Confidence is a heuristic based on rule matches; it is not a calibrated probability.

## 4. Request and response schema

`SupportTicket` validates fields such as ticket ID, subject, message, channel and optional customer tier, rejects unexpected fields and rejects blank required text. `IntentResult` validates category, priority, confidence range, escalation flag, draft response and rationale.

Draft text intentionally avoids claiming an external action (such as a refund or cancellation) has completed. It also warns users not to include passwords, one-time codes or full payment-card numbers.

## 5. Optional provider path and safety boundary

The optional provider uses `OPENAI_API_KEY`, and may use `OPENAI_MODEL` and `OPENAI_BASE_URL`. It requests a JSON object, parses common JSON formats, validates the result and checks that the response ticket ID matches the request. Transient or malformed results are retried with exponential backoff.

**Deterministic post-processing is authoritative for risk:** a detected security incident is forced to security/privacy, urgent priority, escalation, and a fixed security-safe draft. The model cannot downgrade a deterministic high/urgent priority. This does not eliminate prompt injection, provider errors or incorrect classification; sensitive cases still require human review.

## 6. Source map

| File | Responsibility |
|---|---|
| `src/schema.py` | Enums and Pydantic request/response models |
| `src/router.py` | Rule dictionaries, category/priority classification and safe drafts |
| `src/llm_provider.py` | Optional provider transport, JSON parsing, retries and risk enforcement |
| `src/cli.py` | Command-line interface |
| `src/benchmark.py` | Curated 18-ticket synthetic benchmark |
| `tests/test_router.py` | Deterministic routing behavior |
| `tests/test_provider.py` | JSON parsing, retries, schema/ID checks and priority overrides |
| `VALIDATION_REPORT.json` | Test and benchmark evidence |

## 7. Install and run

```bash
python -m venv .venv
# Activate .venv first
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m src.benchmark
python -m src.cli --ticket-id T-1001 --subject "Charged twice" --message "My card was charged twice for the same order."
```

Optional provider example (requires a valid key and can incur API charges):

```bash
python -m src.cli --provider openai --ticket-id T-1002 --subject "Login issue" --message "I cannot log in to my account."
```

The adapter reads environment variables directly; a `.env` file is not automatically loaded by the Python code.

## 8. Verification

GitHub Actions passed all 15 tests after the security/priority regression tests were added. The earlier handcrafted benchmark passed 18 of 18 expected category/priority cases. This is a rule-coverage test on a small synthetic set, not evidence of real support performance. The tests use mock provider transport; no real API request is part of normal validation.

## 9. Limitations and next steps

- Expand to a representative, privacy-reviewed labelled ticket set and report class-wise metrics.
- Calibrate confidence scores rather than treating phrase-match scores as probabilities.
- Review policy-specific escalation thresholds with support operations.
- Apply PII redaction and retention rules before optional external provider calls.
- Keep a human agent in charge of refunds, account recovery, and security incidents.
