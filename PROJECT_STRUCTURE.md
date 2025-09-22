# Agent Control Hub - Project Structure

## 📁 Directory Organization

```
Agent_Control_Hub/
├── src/                          # Main source code
│   ├── __init__.py
│   ├── main.py                   # Main entry point
│   ├── agents/                   # Agent implementations
│   │   ├── __init__.py
│   │   └── factory.py
│   ├── core/                     # Core functionality
│   │   ├── __init__.py
│   │   └── config.py
│   ├── llm/                      # LLM provider abstraction
│   │   ├── __init__.py
│   │   └── llm_provider.py
│   ├── services/                 # Business logic services
│   │   ├── __init__.py
│   │   ├── pipeline.py
│   │   └── plan.py
│   ├── routers/                  # API routes
│   │   ├── __init__.py
│   │   └── projects.py
│   ├── models/                   # Data models
│   │   ├── __init__.py
│   │   └── responses.py
│   └── utils/                    # Utility functions
│       ├── __init__.py
│       ├── env.py
│       ├── files.py
│       ├── prompts.py
│       ├── server_manager.py
│       ├── complex_idea_generator.py
│       └── simple_idea_generator.py
├── ui/                           # User interface
│   ├── streamlit/               # Streamlit app
│   │   ├── __init__.py
│   │   ├── streamlit_app.py
│   │   ├── streamlit_config.toml
│   │   └── streamlit_styles.css
│   └── templates/               # HTML templates
│       └── ui.html
├── scripts/                      # Executable scripts
│   ├── start_hub.py
│   ├── start_hub.bat
│   └── start_hub.sh
├── config/                       # Configuration files
│   └── env.example
├── tests/                        # Test files
│   ├── __init__.py
│   ├── test_llm_provider.py
│   └── test_utils.py
├── examples/                     # Example files
│   ├── test_llm_provider.py
│   ├── test_server.py
│   ├── debug_server.py
│   └── demo_streamlit.py
├── docs/                         # Documentation
│   ├── agenthub_llm_integration_guide.md
│   └── agenthub_next_steps_and_future_improvements.md
├── workspace/                    # Agent workspace (renamed from agent_workspace)
│   ├── projects/
│   └── deploy/
├── logs/                         # Log files
│   └── *.log
├── prompts/                      # Prompt templates
│   ├── fileplan_prompt.txt
│   ├── quality_rubric.txt
│   └── tool_use_prompt.txt
├── server/                       # FastAPI server
│   ├── __init__.py
│   └── app.py
├── .github/                      # GitHub workflows
│   └── workflows/
│       └── ci.yml
├── README.md
├── README_AGENT_HUB.md
├── LICENSE
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
├── IMPLEMENTATION_SUMMARY.md
├── PROJECT_STRUCTURE.md
├── requirements.txt
├── setup.py
├── tools.py
└── .gitignore
```

## 🎯 Key Improvements

### 1. **Modular Source Code (`src/`)**
- All main source code organized under `src/`
- Clear separation of concerns
- Easy to package and distribute

### 2. **Organized UI (`ui/`)**
- Streamlit app in `ui/streamlit/`
- HTML templates in `ui/templates/`
- UI-specific configuration files

### 3. **Centralized Scripts (`scripts/`)**
- All executable scripts in one place
- Easy to find and run
- Cross-platform support

### 4. **Configuration Management (`config/`)**
- Environment examples in `config/`
- Easy to manage different environments
- Clear separation from source code

### 5. **Clean Workspace (`workspace/`)**
- Renamed from `agent_workspace` for clarity
- Contains all agent-generated projects
- Easy to backup and manage

### 6. **Organized Logs (`logs/`)**
- All log files in one location
- Easy to monitor and debug
- Clean root directory

### 7. **Example Files (`examples/`)**
- Test and demo files separated
- Easy to find and run examples
- Clean main directory

## 🚀 Usage

### Running the Application
```bash
# Start everything
python scripts/start_hub.py

# Start only server
python scripts/start_hub.py --server-only

# Start only UI
python scripts/start_hub.py --streamlit-only

# Using main entry point
python -m src.main streamlit
python -m src.main server
python -m src.main test
```

### Development
```bash
# Run tests
python -m pytest tests/

# Run specific test
python -m pytest tests/test_llm_provider.py

# Run examples
python examples/test_llm_provider.py
```

## 📦 Benefits

1. **Cleaner Structure**: Easy to navigate and understand
2. **Better Organization**: Related files grouped together
3. **Easier Maintenance**: Clear separation of concerns
4. **Scalable**: Easy to add new features and modules
5. **Professional**: Follows Python best practices
6. **CI/CD Ready**: Proper structure for automated testing

## 🔧 Migration Notes

- All import statements updated to use new paths
- Log files moved to `logs/` directory
- Workspace renamed from `agent_workspace` to `workspace`
- Scripts updated to use new file locations
- Configuration files organized in `config/`

The project is now much more organized and follows Python packaging best practices!
