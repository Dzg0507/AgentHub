# Agent Control Hub Examples

This directory contains examples and tutorials for using Agent Control Hub.

## Quick Start Examples

### 1. Basic Python Project

Create a simple Python project with Agent Control Hub:

```bash
# Start the hub
python run.py both

# Create a new project with prompt:
# "Create a simple calculator with basic arithmetic operations"
```

### 2. Web Application

Generate a React + TypeScript web application:

```bash
# Use prompt:
# "Build a todo list app with React and TypeScript, including add, edit, delete, and mark complete functionality"
```

### 3. API Service

Create a FastAPI backend service:

```bash
# Use prompt:
# "Create a REST API for a blog with CRUD operations for posts and comments"
```

## Example Projects

### Python Examples

- **`python_calculator/`** - Basic calculator application
- **`python_web_scraper/`** - Web scraping tool
- **`python_data_analysis/`** - Data analysis script
- **`python_api_server/`** - FastAPI server

### Web Development Examples

- **`react_todo_app/`** - React + TypeScript todo application
- **`threejs_visualization/`** - Three.js 3D visualization
- **`nodejs_chat_app/`** - Real-time chat application

### Backend Examples

- **`go_microservice/`** - Go microservice
- **`rust_cli_tool/`** - Rust command-line tool
- **`java_spring_app/`** - Java Spring Boot application

## Tutorials

### Getting Started

1. **Installation and Setup**
   - [Installation Guide](tutorials/01-installation.md)
   - [Configuration](tutorials/02-configuration.md)
   - [First Project](tutorials/03-first-project.md)

2. **Core Features**
   - [Multi-Agent Pipeline](tutorials/04-multi-agent-pipeline.md)
   - [Language Support](tutorials/05-language-support.md)
   - [API Usage](tutorials/06-api-usage.md)

3. **Advanced Topics**
   - [Custom Agents](tutorials/07-custom-agents.md)
   - [LLM Providers](tutorials/08-llm-providers.md)
   - [Deployment](tutorials/09-deployment.md)

## Sample Prompts

### Python Projects

```text
# Simple Calculator
"Create a Python calculator with basic arithmetic operations, error handling, and a simple CLI interface"

# Web Scraper
"Build a web scraper that extracts product information from an e-commerce site and saves to CSV"

# Data Analysis
"Create a data analysis script that processes sales data and generates visualizations"

# API Server
"Build a FastAPI server for a library management system with books, authors, and borrowing records"
```

### Web Applications

```text
# React Todo App
"Create a React + TypeScript todo application with add, edit, delete, and filter functionality"

# E-commerce Site
"Build a modern e-commerce website with product catalog, shopping cart, and checkout"

# Dashboard
"Create an admin dashboard with charts, tables, and real-time data updates"

# Portfolio Site
"Build a personal portfolio website with animations and responsive design"
```

### Backend Services

```text
# Microservice
"Create a Go microservice for user authentication with JWT tokens and password hashing"

# CLI Tool
"Build a Rust CLI tool for file organization and duplicate detection"

# API Gateway
"Create a Java Spring Boot API gateway with rate limiting and authentication"

# Real-time Service
"Build a Node.js real-time chat service with WebSocket support"
```

## Best Practices

### Writing Effective Prompts

1. **Be Specific**: Include details about functionality, technology stack, and requirements
2. **Provide Context**: Explain the purpose and target audience
3. **Specify Constraints**: Mention any limitations or special requirements
4. **Include Examples**: Provide sample inputs/outputs when relevant

### Project Structure

1. **Organize Files**: Use clear directory structure and naming conventions
2. **Include Documentation**: Add README files and code comments
3. **Add Tests**: Include unit tests and integration tests
4. **Environment Setup**: Provide setup instructions and requirements

### Code Quality

1. **Follow Standards**: Use language-specific coding standards
2. **Error Handling**: Include proper error handling and validation
3. **Security**: Consider security best practices
4. **Performance**: Optimize for performance when needed

## Contributing Examples

We welcome contributions of new examples! Please see our [Contributing Guide](../CONTRIBUTING.md) for details.

### Adding a New Example

1. Create a new directory under the appropriate language folder
2. Include a README with description and usage instructions
3. Add the example to this README
4. Submit a pull request

### Example Structure

```
examples/
├── python/
│   ├── calculator/
│   │   ├── README.md
│   │   ├── requirements.txt
│   │   └── src/
│   └── web_scraper/
├── web/
│   ├── react_todo/
│   └── threejs_viz/
└── backend/
    ├── go_microservice/
    └── rust_cli/
```

## Support

If you have questions about the examples or need help getting started:

- Check the [documentation](../docs/)
- Open an [issue](https://github.com/Dzg0507/AgentHub/issues)
- Start a [discussion](https://github.com/Dzg0507/AgentHub/discussions)

## License

Examples are provided under the same MIT License as the main project.
