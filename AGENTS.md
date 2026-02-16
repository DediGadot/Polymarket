# Repository Guidelines

## Project Structure & Module Organization
The runtime entrypoint is `run.py`, with environment-driven settings in `config.py`. Core modules are split by responsibility:
- `client/`: API clients, auth, caching, WebSocket bridges, gas/oracle integrations.
- `scanner/`: opportunity detection, scoring, matching, and strategy logic.
- `executor/`: sizing, safety checks, order execution, unwind flows.
- `monitor/`: logging, status output, and PnL/session tracking.
- `state/` and `pipeline/`: checkpointing and shared runtime utilities.
- `benchmark/`: offline analysis and simulation tools.
- `tests/`: pytest suite for unit, integration, and pipeline behavior.

## Build, Test, and Development Commands
- `uv sync --all-extras`: install runtime + dev dependencies.
- `uv run python run.py --dry-run`: scan markets without wallet/execution.
- `uv run python run.py --scan-only`: full pipeline checks without order placement.
- `uv run python run.py`: paper trading mode.
- `uv run python run.py --live`: live trading (real orders).
- `uv run ruff check .`: lint code (matches CI).
- `PYTHONPATH=. uv run python -m pytest tests/ -v`: run test suite.
- `PYTHONPATH=. uv run python -m pytest tests/ --cov=. --cov-report=term-missing`: coverage report.

## Coding Style & Naming Conventions
Use Python 3.11+ with 4-space indentation, type hints, and clear docstrings on non-trivial logic. Follow existing naming:
- `snake_case` for modules, functions, and variables.
- `PascalCase` for classes/dataclasses.
- `UPPER_SNAKE_CASE` for constants.
Prefer small, composable functions and keep imports absolute from repo root (`PYTHONPATH=.`). Run `ruff` before opening a PR.

## Testing Guidelines
Testing uses `pytest`, `pytest-asyncio`, and `respx`. Name files `test_<feature>.py` and test functions `test_<behavior>`. Keep tests deterministic; mock network-facing dependencies. CI enforces coverage (`--cov-fail-under=85`), so new behavior should include tests and maintain the threshold.

## Commit & Pull Request Guidelines
Recent commits use short, direct subjects (often lowercase, e.g., `multiple markets`, `w/ kalshi`). Keep commit titles concise and specific; prefer imperative phrasing with optional scope (example: `scanner: tighten stale quote filters`).

For PRs, include:
- What changed and why.
- Risk notes (execution/safety/config impact).
- Test evidence (exact commands run and outcomes).
- Related issue/task links and screenshots for UI/report changes when relevant.

## Security & Configuration Tips
Never commit `.env`, API keys, private keys, or wallet secrets. Start from `.env.example`, and validate risky changes in `--dry-run` or paper mode before `--live`.
