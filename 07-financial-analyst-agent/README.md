# Project 07 — Autonomous Financial Analyst Agent

**Status:** Verified in Google Colab. All 20 automated tests pass, the four-case agent evaluation reports 100% answer coverage on its synthetic smoke set, the operating-margin example calculates 15.00%, and research-note search returns the relevant Liquidity and leverage section. See VALIDATION_REPORT.json.

## Objective

Demonstrate a tool-using financial analyst workflow: read synthetic financial metrics through a read-only SQL tool, retrieve relevant local research passages, run safe arithmetic, and assemble an evidence-backed answer.

## Features

- Automatically creates a small SQLite database with fictional quarterly metrics for three fictional companies.
- Typed internal tool observations and an action/observation trace for every tool call.
- SQL read-only mode using SQLite URI read-only access, a database authorizer, one allowlisted table, parameterized queries, comment/multi-statement rejection, row caps, and a query-step budget.
- Safe calculator evaluates a restricted arithmetic AST; it does not call Python eval.
- Local Markdown research-note search with file and section citations.
- Deterministic planning for quarterly summary, cash/debt, Q3-to-Q4 revenue change, operating margin, and risk-note questions.
- Abstains from personalized investment recommendations and price targets.
- No API key, network request, non-standard dependency, or GPU required.

## Quick start

Python 3.10+ is recommended. The default implementation uses the Python standard library.

~~~bash
python -m unittest discover -s tests -v
python -m src.evaluate
python -m src.cli "What was ACME Systems operating margin in Q4?" --json
python -m src.cli "What are the main liquidity and leverage risks?"
python -m src.cli "Compare Q3 to Q4 revenue growth across the companies." --json
~~~

The first run creates data/financials_demo.sqlite with invented metrics. The generated SQLite file is ignored by Git and should not be committed.

## Google Colab

For a fresh runtime:

~~~python
%cd /content
!git clone https://github.com/chinthavishnupriya/ml-genai-enterprise-lab.git
%cd /content/ml-genai-enterprise-lab/07-financial-analyst-agent
!python -m pip install -r requirements.txt
!python -m unittest discover -s tests -v
!python -m src.evaluate
!python -m src.cli "What was ACME Systems operating margin in Q4?" --json
~~~

If the repository is already cloned in the current runtime, pull the latest main branch and enter the project directory instead of cloning again.

## Tool-permission boundaries

- The SQL tool connects with read-only URI mode and allows reads from quarterly_financials only.
- User questions are mapped to fixed, parameterized SQL templates; the agent does not generate arbitrary SQL from the user prompt.
- The calculator allows only numeric literals and basic arithmetic.
- The document search tool reads only local Markdown or TXT research notes.
- Every tool action is recorded with its input summary, observation, and data source.

## Safety and limitations

This project uses entirely synthetic financial data and fictional research notes. It is not financial, tax, accounting, legal, or investment advice. It does not fetch live prices, financial statements, or current economic news. The rules-based planner is deliberately narrow and may fail to understand questions outside its supported patterns. Before production use, add authorization, audit logging, data-access review, error handling, and tests against representative permissioned data.
