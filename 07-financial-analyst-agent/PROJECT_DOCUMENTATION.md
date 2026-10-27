# Project 07 — Autonomous Financial Analyst Agent

## 1. Project overview

This is an offline, deterministic tool-using agent for analysing invented quarterly metrics and searching fictional research notes. It routes supported questions to a read-only SQLite query, restricted calculator or local document search, then assembles an answer with citations and a tool trace.

It is not a general LLM planner and does not fetch live market data.

## 2. Technology stack

| Technology | Purpose |
|---|---|
| Python | Routing, tool orchestration, answer formatting and CLI |
| SQLite / `sqlite3` | Local fictional quarterly financial metrics |
| Python AST | Safe arithmetic expression parsing without Python `eval` |
| Python standard library | Read-only SQLite URI, authorizer/progress handler, regex and Markdown search |
| `unittest` | Agent, SQL restrictions, calculator and search tests |

No API key, external embedding library, GPU or network call is required.

## 3. Agent tools and access controls

**Read-only SQL tool**
- Initializes a tiny synthetic table if no demo database exists.
- Opens the database in URI read-only mode.
- Uses a SQLite authorizer to permit reading only the approved `quarterly_financials` table and a small allowlist of aggregate functions.
- Rejects non-SELECT statements, comments and semicolons, uses bound parameters, limits returned rows and limits SQLite virtual-machine steps.

**Safe calculator**
- Parses arithmetic into an AST and evaluates only numeric literals and allowed arithmetic operators.
- Rejects calls, unsupported nodes, division/modulo by zero and excessive exponentiation/results.

**Local document search**
- Reads local Markdown/TXT research notes, breaks them into heading sections, normalizes simple plurals and ranks matching passages with a heading-match bonus.

## 4. Question routing

The deterministic planner handles a finite set of patterns: company operating margin, Q3-to-Q4 revenue growth comparison, cash/debt balances, general quarterly summaries and research-note topics such as liquidity/leverage risks. It declines personalized stock recommendations and price targets. Each tool call creates a trace event with input summary, observation and source.

## 5. Source map

| File | Responsibility |
|---|---|
| `src/tools.py` | SQLite access control, calculator and local section search |
| `src/agent.py` | Question routing, tool calls, answer assembly and traces |
| `src/cli.py` | Plain-text or JSON answer, citations and trace |
| `src/evaluate.py` | Four deterministic smoke questions |
| `data/research/` | Fictional market outlook, risk register and methodology notes |
| `tests/test_agent.py` | Routing, calculations, citations and refusal tests |
| `tests/test_tools.py` | SQL access restrictions, calculator security and local search |
| `VALIDATION_REPORT.json` | Previously recorded Colab verification |

## 6. Run locally

```bash
python -m unittest discover -s tests -v
python -m src.evaluate
python -m src.cli "What was ACME Systems operating margin in Q4?" --json
python -m src.cli "What are the main liquidity and leverage risks?"
python -m src.cli "Compare Q3 to Q4 revenue growth across the companies." --json
```

The first run creates `data/financials_demo.sqlite` using invented values. The SQLite file is ignored by Git.

## 7. Example calculation

The supplied synthetic Q4 figures for ACME Systems include revenue of $1,400,000 and operating income of $210,000. The calculator computes operating margin as operating income / revenue × 100 = 15%. This example demonstrates tool routing and arithmetic, not a real-company financial result.

## 8. Validation and limitations

Twenty unit tests and a four-case synthetic smoke evaluation were recorded as passing. The evaluation checks expected question routing, citations and trace presence; it is not a financial correctness benchmark.

The planner is intentionally narrow, the numbers are fictional, and the agent has no current market data. Extend it only with reviewed, authorized sources, explicit permissions, audit logs, robust error handling and domain-expert oversight.
