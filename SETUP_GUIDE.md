# 🚀 Agent Control Hub - Complete Setup Guide

This guide will walk you through setting up Agent Control Hub from scratch, ensuring everything works perfectly.

## 📋 Prerequisites

### System Requirements
- **Operating System**: Windows 10/11, macOS 10.15+, or Linux (Ubuntu 18.04+)
- **Python**: Version 3.8 or higher
- **Memory**: At least 4GB RAM (8GB recommended)
- **Storage**: 2GB free space
- **Internet**: Required for downloading dependencies and API calls

### Required Software
- **Python 3.8+**: [Download from python.org](https://www.python.org/downloads/)
- **Git**: [Download from git-scm.com](https://git-scm.com/downloads)
- **Code Editor** (optional): VS Code, PyCharm, or any text editor

## 🔧 Installation Steps

### Step 1: Clone the Repository

```bash
# Clone the repository
git clone https://github.com/Dzg0507/AgentHub.git
cd AgentHub

# Verify you're in the right directory
ls -la  # On Windows: dir
```

**Expected Output:**
```
README.md
requirements.txt
setup.py
src/
agents/
services/
...
```

### Step 2: Create a Virtual Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Verify activation (should show (.venv) in prompt)
python --version
```

**Expected Output:**
```
Python 3.8.10  # or your Python version
```

### Step 3: Install Dependencies

```bash
# Upgrade pip
python -m pip install --upgrade pip

# Install the package in development mode
pip install -e .

# Verify installation
pip list | grep agent-control-hub
```

**Expected Output:**
```
agent-control-hub    1.0.0    /path/to/AgentHub
```

### Step 4: Set Up Environment Variables

```bash
# Copy the example environment file
cp config/env.example .env

# Edit the .env file with your API keys
# On Windows:
notepad .env
# On macOS/Linux:
nano .env
```

**Example .env file:**
```env
# LLM Provider Configuration
LLM_PROVIDER=gemini
GOOGLE_API_KEY=your_google_api_key_here
TOGETHER_API_KEY=your_together_api_key_here
OPENROUTER_API_KEY=your_openrouter_api_key_here
LLM_MODEL=gemini-1.5-flash
```

### Step 5: Test the Installation

```bash
# Test basic imports
python -c "from src.llm.llm_provider import LLMProvider; print('✅ LLM Provider works')"
python -c "from agents.factory import create_agents; print('✅ Agent Factory works')"

# Test the launcher
python run.py help
```

**Expected Output:**
```
🤖 Agent Control Hub - Quick Launcher
========================================
Available commands:
  ui        - Start Streamlit UI only
  server    - Start FastAPI server only
  both      - Start both UI and server
  test      - Run the test suite
  help      - Show this help message
```

## 🎯 Quick Start

### Option 1: One-Command Startup (Recommended)

```bash
# Start everything with one command
python run.py both
```

This will:
- ✅ Start the FastAPI server on http://127.0.0.1:8000
- ✅ Start the Streamlit UI on http://localhost:8501
- ✅ Monitor both processes
- ✅ Provide unified shutdown with Ctrl+C

### Option 2: Start Components Separately

```bash
# Terminal 1: Start the server
python run.py server

# Terminal 2: Start the UI
python run.py ui
```

### Option 3: Use Platform Scripts

**Windows:**
```cmd
# Double-click or run:
scripts\start_hub.bat
```

**macOS/Linux:**
```bash
# Make executable and run:
chmod +x scripts/start_hub.sh
./scripts/start_hub.sh
```

## 🌐 Accessing the Application

### Web Interface
1. **Open your browser** and go to: http://localhost:8501
2. **You should see** the Agent Control Hub dashboard
3. **Create a new project** using the form on the main page

### API Documentation
1. **Open your browser** and go to: http://127.0.0.1:8000/docs
2. **Explore the API** using the interactive Swagger UI
3. **Test endpoints** directly from the browser

## 🧪 Testing Your Setup

### Run the Test Suite

```bash
# Run all tests
python run.py test

# Or run tests manually
python -m pytest tests/ -v
```

**Expected Output:**
```
============================= test session starts =============================
platform win32 -- Python 3.8.10, pytest-7.0.0, pluggy-1.0.0
collected 4 items

tests/test_basic.py::TestBasicFunctionality::test_agent_factory PASSED   [ 25%]
tests/test_basic.py::TestBasicFunctionality::test_llm_provider_initialization PASSED [ 50%]
tests/test_basic.py::TestBasicFunctionality::test_project_structure PASSED [ 75%]
tests/test_basic.py::TestBasicFunctionality::test_required_files PASSED [100%]

============================== 4 passed in 0.08s ==============================
```

### Test API Endpoints

```bash
# Test health endpoint
curl http://127.0.0.1:8000/health

# Test projects endpoint
curl http://127.0.0.1:8000/projects
```

## 🎨 Creating Your First Project

### 1. Open the Web Interface
- Go to http://localhost:8501
- You should see the dashboard

### 2. Create a New Project
- Click "Create New Project" or use the form
- Enter a project description, for example:
  ```
  Create a simple Python calculator with basic arithmetic operations
  ```
- Select your preferred language (Python, React, Node.js, etc.)
- Click "Create Project"

### 3. Monitor Progress
- Watch the real-time progress indicators
- See the multi-agent pipeline in action:
  - Prompt Enhancement
  - File Planning
  - Code Generation
  - Environment Setup

### 4. Download Your Project
- Once complete, download the project as a ZIP file
- Extract and run your generated code

## 🔧 Troubleshooting

### Common Issues

#### 1. Import Errors
```bash
# Error: ModuleNotFoundError
# Solution: Make sure you're in the project directory and virtual environment is activated
cd AgentHub
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

#### 2. Port Already in Use
```bash
# Error: Port 8000 or 8501 already in use
# Solution: Kill existing processes or change ports
# On Windows:
netstat -ano | findstr :8000
taskkill /PID <PID_NUMBER> /F

# On macOS/Linux:
lsof -ti:8000 | xargs kill -9
```

#### 3. API Key Issues
```bash
# Error: API key not found
# Solution: Check your .env file
cat .env  # On Windows: type .env
# Make sure GOOGLE_API_KEY is set correctly
```

#### 4. Permission Errors
```bash
# Error: Permission denied
# Solution: Check file permissions
chmod +x scripts/start_hub.sh  # On macOS/Linux
```

### Getting Help

1. **Check the logs**:
   ```bash
   # Server logs
   tail -f logs/agent_hub.log
   
   # Streamlit logs (check terminal output)
   ```

2. **Verify installation**:
   ```bash
   pip list | grep agent-control-hub
   python -c "import src.llm.llm_provider; print('OK')"
   ```

3. **Check system requirements**:
   ```bash
   python --version  # Should be 3.8+
   pip --version     # Should be recent
   ```

## 📚 Next Steps

### Learn More
- **Read the [README.md](README.md)** for comprehensive documentation
- **Check [examples/](examples/)** for sample projects and tutorials
- **Explore [docs/](docs/)** for detailed guides

### Customize Your Setup
- **Configure LLM providers** in your `.env` file
- **Modify prompts** in the `prompts/` directory
- **Add new languages** by extending the configuration

### Contribute
- **Report issues** on [GitHub Issues](https://github.com/Dzg0507/AgentHub/issues)
- **Suggest features** via [GitHub Discussions](https://github.com/Dzg0507/AgentHub/discussions)
- **Submit pull requests** following our [Contributing Guide](CONTRIBUTING.md)

## ✅ Verification Checklist

Before you start using Agent Control Hub, make sure:

- [ ] Python 3.8+ is installed
- [ ] Virtual environment is created and activated
- [ ] Package is installed successfully (`pip list | grep agent-control-hub`)
- [ ] Environment variables are set (`.env` file exists)
- [ ] Tests pass (`python run.py test`)
- [ ] UI starts successfully (`python run.py ui`)
- [ ] Server starts successfully (`python run.py server`)
- [ ] Web interface is accessible at http://localhost:8501
- [ ] API documentation is accessible at http://127.0.0.1:8000/docs

## 🎉 You're Ready!

If all checks pass, you're ready to start creating amazing projects with Agent Control Hub!

**Happy coding!** 🚀
