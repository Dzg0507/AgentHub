# Contributing to Agent Control Hub

Thank you for your interest in contributing to Agent Control Hub! This document provides guidelines and information for contributors.

## 🤝 How to Contribute

### Reporting Issues

Before creating an issue, please:

1. **Search existing issues** to avoid duplicates
2. **Check the documentation** to see if your question is answered
3. **Use the issue templates** when available

When creating an issue, please include:

- **Clear description** of the problem or feature request
- **Steps to reproduce** (for bugs)
- **Expected vs actual behavior**
- **Environment details** (OS, Python version, etc.)
- **Screenshots** if applicable

### Suggesting Enhancements

We welcome enhancement suggestions! Please:

1. **Check existing discussions** first
2. **Provide clear use cases** and benefits
3. **Consider implementation complexity**
4. **Be specific** about the desired outcome

### Code Contributions

#### Getting Started

1. **Fork the repository**
2. **Clone your fork**:
   ```bash
   git clone https://github.com/your-username/AgentHub.git
   cd AgentHub
   ```

3. **Create a virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -e ".[dev]"
   ```

5. **Set up pre-commit hooks**:
   ```bash
   pre-commit install
   ```

#### Development Workflow

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following our coding standards

3. **Add tests** for new functionality

4. **Run tests** to ensure everything works:
   ```bash
   pytest tests/ -v
   ```

5. **Format and lint your code**:
   ```bash
   black .
   flake8 .
   mypy src/
   ```

6. **Commit your changes**:
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   ```

7. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

8. **Create a Pull Request**

## 📋 Coding Standards

### Python Code Style

- **Follow PEP 8** guidelines
- **Use Black** for code formatting
- **Use type hints** where appropriate
- **Write docstrings** for all functions and classes
- **Keep functions small** and focused
- **Use meaningful variable names**

### Code Formatting

We use **Black** for code formatting. Run it before committing:

```bash
black .
```

### Linting

We use **flake8** for linting. Run it to check for issues:

```bash
flake8 .
```

### Type Checking

We use **mypy** for type checking:

```bash
mypy src/
```

### Documentation

- **Write clear docstrings** for all public functions
- **Update README.md** if adding new features
- **Add examples** in docstrings where helpful
- **Keep comments up to date**

## 🧪 Testing

### Writing Tests

- **Write tests** for all new functionality
- **Test edge cases** and error conditions
- **Use descriptive test names**
- **Keep tests simple** and focused
- **Mock external dependencies**

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_basic.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run only failed tests
pytest --lf
```

### Test Structure

```python
def test_function_name():
    """Test description of what this test does"""
    # Arrange
    input_data = "test"
    
    # Act
    result = function_under_test(input_data)
    
    # Assert
    assert result == expected_output
```

## 📝 Commit Messages

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

### Format

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

### Types

- **feat**: A new feature
- **fix**: A bug fix
- **docs**: Documentation changes
- **style**: Code style changes (formatting, etc.)
- **refactor**: Code refactoring
- **test**: Adding or updating tests
- **chore**: Maintenance tasks

### Examples

```bash
feat: add support for Rust language
fix: resolve memory leak in pipeline processing
docs: update API documentation
test: add unit tests for LLM provider
chore: update dependencies
```

## 🔄 Pull Request Process

### Before Submitting

1. **Ensure tests pass** locally
2. **Format and lint** your code
3. **Update documentation** if needed
4. **Write a clear description** of your changes
5. **Link related issues** if applicable

### PR Description Template

```markdown
## Description
Brief description of the changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Tests pass locally
- [ ] New tests added (if applicable)
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or clearly documented)
```

### Review Process

1. **Automated checks** must pass
2. **Code review** by maintainers
3. **Address feedback** promptly
4. **Keep PRs focused** and reasonably sized
5. **Respond to comments** constructively

## 🏗️ Project Structure

Understanding the codebase:

```
src/
├── llm/           # LLM provider abstraction
├── ui/            # User interface components
└── main.py        # Main entry point

agents/            # Agent definitions and factory
services/          # Business logic services
core/              # Core configuration
models/            # Pydantic response models
routers/           # FastAPI routers
server/            # FastAPI application
utils/             # Utility functions
tests/             # Test files
```

## 🎯 Areas for Contribution

### High Priority

- **New Language Support**: Add support for additional programming languages
- **Agent Improvements**: Enhance existing agents or add new ones
- **UI/UX Enhancements**: Improve the Streamlit interface
- **Performance Optimization**: Speed up code generation and processing
- **Testing**: Increase test coverage

### Medium Priority

- **Documentation**: Improve docs and add examples
- **Error Handling**: Better error messages and recovery
- **Configuration**: More flexible configuration options
- **Monitoring**: Better logging and monitoring

### Low Priority

- **Code Cleanup**: Refactoring and code organization
- **Dependencies**: Update and optimize dependencies
- **CI/CD**: Improve automation and testing

## 🐛 Bug Reports

When reporting bugs, please include:

1. **Clear title** describing the issue
2. **Steps to reproduce** the problem
3. **Expected behavior** vs actual behavior
4. **Environment details**:
   - OS and version
   - Python version
   - Package versions
5. **Error messages** and logs
6. **Screenshots** if applicable

## 💡 Feature Requests

When suggesting features:

1. **Check existing issues** and discussions
2. **Describe the use case** clearly
3. **Explain the benefits** to users
4. **Consider implementation** complexity
5. **Provide examples** if possible

## 📞 Getting Help

- **GitHub Discussions**: For questions and general discussion
- **GitHub Issues**: For bug reports and feature requests
- **Code Review**: Ask questions in PR comments
- **Documentation**: Check the docs folder

## 🏆 Recognition

Contributors will be recognized in:

- **CONTRIBUTORS.md** file
- **Release notes** for significant contributions
- **GitHub contributor graph**

## 📄 License

By contributing, you agree that your contributions will be licensed under the same MIT License that covers the project.

## 🙏 Thank You

Thank you for contributing to Agent Control Hub! Your contributions help make this project better for everyone.

---

**Questions?** Feel free to open a discussion or issue if you need help getting started!