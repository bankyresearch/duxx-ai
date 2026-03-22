# Contributing to Duxx AI

We welcome contributions! Here's how to get started.

## Setup

```bash
git clone https://github.com/bankyai/bankyai.git
cd bankyai
pip install -e ".[dev]"
```

## Development

1. Create a feature branch: `git checkout -b feature/my-feature`
2. Write code and tests
3. Run tests: `pytest tests/ -v`
4. Run linter: `ruff check .`
5. Run type checker: `mypy bankyai/`
6. Submit a PR

## Code Style

- We use `ruff` for formatting and linting
- Type hints are required for all public APIs
- Tests are required for all new features

## Areas for Contribution

- New guardrail implementations
- Additional LLM provider integrations
- Memory backend implementations (Redis, PostgreSQL, vector stores)
- Improved complexity estimation for the adaptive router
- Studio UI enhancements
- Documentation and examples
