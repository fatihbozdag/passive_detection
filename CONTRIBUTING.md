# Contributing to PassivePy

Thank you for your interest in contributing to PassivePy! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct.

## How to Contribute

1. Fork the repository
2. Create a new branch for your feature/fix
3. Make your changes
4. Run tests and ensure they pass
5. Submit a pull request

## Development Setup

1. Install development dependencies:
```bash
rye sync
```

2. Install pre-commit hooks:
```bash
pre-commit install
```

## Code Style

We use the following tools to maintain code quality:

- `ruff` for linting
- `black` for code formatting
- `mypy` for type checking

Run these tools before submitting a pull request:
```bash
ruff check .
black .
mypy .
```

## Testing

- Write tests for new features
- Ensure all tests pass
- Maintain or improve test coverage
- Use pytest for testing

## Documentation

- Update documentation for new features
- Add docstrings to new functions/classes
- Keep README.md up to date
- Document breaking changes

## Pull Request Process

1. Update the README.md with details of changes if needed
2. Update the CHANGELOG.md with details of changes
3. The PR must pass all CI checks
4. You may merge the PR once you have the sign-off of at least one other developer

## Questions?

Feel free to open an issue if you have any questions about contributing. 