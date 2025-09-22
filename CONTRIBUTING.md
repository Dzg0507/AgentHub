# Contributing to Agent Control Hub

Thank you for your interest in contributing to Agent Control Hub! This document provides guidelines and information for contributors.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contributing Guidelines](#contributing-guidelines)
- [Pull Request Process](#pull-request-process)
- [Issue Reporting](#issue-reporting)
- [Development Workflow](#development-workflow)

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/your-username/Agent_Control_Hub.git`
3. Create a new branch: `git checkout -b feature/your-feature-name`
4. Make your changes
5. Test your changes
6. Submit a pull request

## Development Setup

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Git

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/Agent_Control_Hub.git
   cd Agent_Control_Hub
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```bash
   cp env.example .env
   # Edit .env with your API keys and configuration
   ```

5. Run tests to verify installation:
   ```bash
   python -m pytest tests/
   ```

## Contributing Guidelines

### Code Style

- Follow PEP 8 style guidelines
- Use type hints where appropriate
- Write clear, descriptive variable and function names
- Add docstrings to all public functions and classes
- Keep functions small and focused on a single responsibility

### Testing

- Write unit tests for new functionality
- Ensure all tests pass before submitting a PR
- Aim for good test coverage (80%+)
- Use descriptive test names that explain what is being tested

### Documentation

- Update README.md if you add new features
- Add docstrings to new functions and classes
- Update API documentation if applicable
- Include examples in your docstrings

### Commit Messages

Use clear, descriptive commit messages:

```
feat: add support for OpenRouter API
fix: resolve issue with file path handling
docs: update installation instructions
test: add unit tests for LLM provider
refactor: improve error handling in pipeline
```

## Pull Request Process

1. **Create a feature branch** from `main`
2. **Make your changes** following the guidelines above
3. **Write tests** for your changes
4. **Update documentation** as needed
5. **Run the test suite** to ensure everything passes
6. **Submit a pull request** with a clear description

### Pull Request Template

When creating a pull request, please include:

- **Description**: What changes were made and why
- **Type**: Bug fix, feature, documentation, etc.
- **Testing**: How the changes were tested
- **Breaking Changes**: Any breaking changes (if applicable)
- **Checklist**: Confirm all requirements are met

## Issue Reporting

When reporting issues, please include:

- **Description**: Clear description of the issue
- **Steps to Reproduce**: How to reproduce the issue
- **Expected Behavior**: What should happen
- **Actual Behavior**: What actually happens
- **Environment**: OS, Python version, dependencies
- **Screenshots**: If applicable

## Development Workflow

### Adding New LLM Providers

1. Add provider support in `llm_provider.py`
2. Implement the `_chat_<provider>` method
3. Add provider to `get_available_models()`
4. Add tests for the new provider
5. Update documentation

### Adding New Agents

1. Create agent class in `agents/factory.py`
2. Add system message and configuration
3. Add to `create_agents()` function
4. Write tests for the new agent
5. Update documentation

### Adding New Pipeline Steps

1. Add step function in `services/pipeline.py`
2. Add step to `pipeline_steps` list
3. Implement error handling and logging
4. Write tests for the new step
5. Update documentation

## Testing

### Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_llm_provider.py

# Run with coverage
python -m pytest tests/ --cov=. --cov-report=html

# Run with verbose output
python -m pytest tests/ -v
```

### Test Structure

- Unit tests go in `tests/` directory
- Test files should be named `test_*.py`
- Test classes should be named `Test*`
- Test methods should be named `test_*`

## Code Review Process

1. All pull requests require review
2. At least one approval is required
3. All CI checks must pass
4. Code should follow project conventions
5. Tests should be included and passing

## Release Process

1. Update version numbers
2. Update CHANGELOG.md
3. Create release tag
4. Build and test release
5. Deploy to production

## Getting Help

- Check existing issues and discussions
- Join our community discussions
- Contact maintainers for urgent issues

## License

By contributing to Agent Control Hub, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to Agent Control Hub! 🚀
