# Agent Control Hub

A centralized hub for multi-agent code creation, enhancement, and deployment using Streamlit UI and FastAPI backend.

## Features

- **Multi-Agent Code Generation**: Collaborative agents for prompt enhancement, file planning, code generation, testing, and deployment
- **Language Support**: Python, Node.js, React+TypeScript, Three.js, Go, Rust, Java
- **Virtual Environment Management**: Per-project virtual environments with setup automation
- **Streamlit UI**: Modern web interface for project management
- **FastAPI Backend**: RESTful API with automatic documentation
- **File Generation**: Guaranteed file creation with fallback scaffolding
- **Project Management**: Create, monitor, download, and manage code generation projects

## Project Structure

```
Agent_Control_Hub/
├── src/                    # Main source code
│   ├── agents/            # Agent definitions and factory
│   ├── llm/               # LLM provider abstraction
│   ├── services/          # Business logic services
│   ├── ui/                # User interface components
│   └── utils/             # Utility functions
├── scripts/               # Executable scripts
├── config/                # Configuration files
├── tests/                 # Test files
├── examples/              # Example files
├── workspace/             # Agent workspace
├── logs/                  # Log files
└── docs/                  # Documentation
│   └── factory.py         # Creates and configures all agents
├── core/                  # Core configuration and constants
│   ├── __init__.py
│   └── config.py         # Centralized configuration
├── models/                # Pydantic response models
│   ├── __init__.py
│   └── responses.py      # API response models
├── prompts/               # Externalized prompts
│   ├── fileplan_prompt.txt
│   └── quality_rubric.txt
├── routers/               # FastAPI routers
│   ├── __init__.py
│   └── projects.py       # Project-related endpoints
├── server/                # FastAPI application
│   └── app.py            # Main FastAPI app
├── services/              # Business logic services
│   ├── __init__.py
│   ├── plan.py           # Project data models
│   └── pipeline.py       # Processing pipeline
├── templates/             # HTML templates
│   └── ui.html           # Built-in web UI
├── utils/                 # Utility functions
│   ├── env.py            # Environment and command utilities
│   ├── files.py          # File operations and snippet execution
│   └── prompts.py        # Prompt loading utilities
├── agent_hub_server.py   # Legacy entry point (backward compatibility)
├── streamlit_app.py      # Streamlit web UI
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd Agent_Control_Hub
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   ```bash
   # Copy the example environment file
   cp config/env.example .env
   
   # Edit .env with your API keys
   # You only need to set the keys for the providers you want to use
   ```
   
   Example `.env` file:
   ```env
   # LLM Provider Configuration
   LLM_PROVIDER=gemini
   GOOGLE_API_KEY=your_google_api_key_here
   TOGETHER_API_KEY=your_together_api_key_here
   OPENROUTER_API_KEY=your_openrouter_api_key_here
   LLM_MODEL=gemini-1.5-flash
   ```

## Usage

### Option 1: One-Command Startup (Recommended)

**Start everything with one command:**

```bash
# Quick launcher (recommended)
python run.py both

# Or use the full script
python scripts/start_hub.py
```

Or use the platform-specific scripts:
- **Windows**: Double-click `scripts/start_hub.bat` or run `scripts/start_hub.bat`
- **Unix/Linux/Mac**: `./scripts/start_hub.sh`

This will automatically:
- Start the FastAPI server on http://127.0.0.1:8000
- Start the Streamlit UI on http://localhost:8501
- Monitor both processes and restart if needed
- Provide unified shutdown with Ctrl+C

### Option 2: Enhanced Streamlit UI with Server Management

1. **Start just the Streamlit UI**:
   ```bash
   streamlit run streamlit_app.py
   ```

2. **Use the built-in server management**:
   - Open http://localhost:8501
   - Go to "Server Status" tab
   - Use "Start Server" button to start the FastAPI backend
   - Or use the "Start Everything" button in the sidebar

### Option 3: Manual Startup (Advanced)

1. **Start the FastAPI backend**:
   ```bash
   python agent_hub_server.py
   ```
   Or use the new modular version:
   ```bash
   python server/app.py
   ```

2. **Start the Enhanced Streamlit UI** (in a new terminal):
   ```bash
   streamlit run streamlit_app.py
   ```

3. **Access the UI**: Open http://localhost:8501 in your browser

#### Enhanced UI Features:
- **🏠 Dashboard**: Welcome page with project creation and recent projects
- **📋 Projects**: Comprehensive project management with filtering and sorting
- **🏥 Server Status**: Real-time server health monitoring with management controls
- **🛠️ Debug Tools**: Testing and debugging utilities
- **🚀 Pipeline Visualization**: Step-by-step process tracking with progress indicators
- **📁 File Management**: View generated files and download projects
- **⚡ Real-time Updates**: Auto-refresh with configurable intervals
- **🎨 Modern Design**: Clean, responsive interface with custom styling
- **🔧 Server Management**: Start, stop, and restart server directly from the UI
- **🚀 One-Click Startup**: Launch everything with a single command

### Option 2: Built-in Web UI

1. **Start the server**:
   ```bash
   python agent_hub_server.py
   ```

2. **Access the built-in UI**: Open http://127.0.0.1:8000/ui in your browser

### Option 3: Demo Mode

Try the enhanced UI features without a backend:

```bash
streamlit run demo_streamlit.py
```

This demo showcases all the UI enhancements including:
- Pipeline progress visualization
- Enhanced project cards
- Interactive forms
- Real-time status updates

### Option 4: API Only

1. **Start the server**:
   ```bash
   python agent_hub_server.py
   ```

2. **Access API documentation**: Open http://127.0.0.1:8000/docs in your browser

## API Endpoints

### Projects
- `POST /projects` - Create a new project
- `GET /projects` - List all projects
- `GET /projects/{id}` - Get project details
- `DELETE /projects/{id}` - Delete a project
- `GET /projects/{id}/download` - Download project as ZIP
- `GET /projects/{id}/files` - List project files
- `GET /projects/{id}/logs` - Get execution logs
- `POST /projects/{id}/retry` - Retry project processing
- `POST /projects/{id}/force-scaffold` - Force create minimal scaffold
- `POST /projects/{id}/venv` - Create virtual environment
- `POST /projects/{id}/setup` - Run setup commands

### System
- `GET /` - Root endpoint with server info
- `GET /health` - Health check
- `GET /ui` - Built-in web interface

## Supported Languages

The system supports code generation for multiple programming languages:

- **Python**: Standard library focus, with virtual environment support
- **Node.js**: JavaScript projects with npm package management
- **React+TypeScript**: Modern React applications with TypeScript
- **Three.js**: 3D web applications and visualizations
- **Go**: Go modules with standard toolchain
- **Rust**: Cargo-based projects with build automation
- **Java**: Gradle-based Java projects

## Project Workflow

1. **Create Project**: Submit a prompt describing what you want to build
2. **Prompt Enhancement**: AI enhances your prompt with technical requirements
3. **File Planning**: AI creates a structured file plan for the project
4. **Code Generation**: AI generates complete, runnable code
5. **Environment Setup**: Virtual environment and dependencies are configured
6. **Testing**: Generated code is tested for functionality
7. **Deployment**: Project is packaged and ready for deployment
8. **Download**: Get your complete project as a ZIP file

## Configuration

Key configuration options in `core/config.py`:

- `SERVER_HOST` and `SERVER_PORT`: Server binding address
- `WORKSPACE_DIR`: Directory for generated projects
- `TEMP_TTL_MINUTES`: Project retention time
- `LLM_CONFIG`: Language model configuration
- `SUPPORTED_LANGUAGES`: Available programming languages

## Development

### Running Tests
```bash
# Run utility tests
python -m pytest tests/ -v

# Run integration tests
python -m pytest tests/integration/ -v
```

### Adding New Languages

1. Add language to `SUPPORTED_LANGUAGES` in `core/config.py`
2. Update `_write_language_scaffold()` in `services/pipeline.py`
3. Add setup commands in `setup_project()` function
4. Update language selector in `streamlit_app.py`

### Adding New Agents

1. Define agent in `agents/factory.py`
2. Add agent to the returned dictionary
3. Use agent in pipeline functions in `services/pipeline.py`

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed and virtual environment is activated
2. **API Key Issues**: Verify `GOOGLE_API_KEY` is set in `.env` file
3. **Port Conflicts**: Change `SERVER_PORT` in `core/config.py` if port 8000 is in use
4. **File Generation Failures**: Use the "Force Scaffold" button to create minimal files

### Logs

- Server logs: `agent_hub.log`
- Project execution logs: Available via `/projects/{id}/logs` endpoint
- Streamlit logs: Check terminal output

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
