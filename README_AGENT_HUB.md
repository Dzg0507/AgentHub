# Advanced Agent Hub Server

A centralized hub for multi-agent code creation, enhancement, and deployment using AG2 (updated AutoGen).

## Features

- **Web Server**: FastAPI-based REST API with web UI
- **Multi-Agent Pipeline**: Seamless workflow from prompt to deployment
- **Real-time Status**: Track project progress in real-time
- **Code Generation**: Complete, production-ready code generation
- **Testing**: Automated testing and validation
- **Deployment**: Docker and deployment script generation
- **File Management**: Organized workspace with project isolation

## Quick Start

### Prerequisites
- Python 3.10+
- Google API Key (for Gemini)
- AG2 installed from the downloaded repository

### Installation
```bash
# Install dependencies
pip install fastapi uvicorn

# Set your Google API key
export GOOGLE_API_KEY="your_api_key_here"
# or add to .env file: GOOGLE_API_KEY=your_api_key_here
```

### Running the Server

#### Option 1: Direct Start
```bash
python agent_hub_server.py
```

#### Option 2: Setup and Start (Recommended)
```bash
python setup.py
python start_server.py
```

The server will start at: **http://127.0.0.1:8000**

## Usage

### Web Interface
1. Open your browser and go to: **http://127.0.0.1:8000/ui**
2. Enter your project description in the text area
3. Click "Create Project"
4. Watch the real-time status updates as agents process your request
5. Once completed, click "Download ZIP" to download your project files
6. Click "View Files" to see all files in the project

**Note**: Make sure to set your `GOOGLE_API_KEY` in the `.env` file before creating projects.

### Downloading Projects
- **ZIP Download**: Automatically packages all project files into a ZIP archive
- **File Listing**: View all files with sizes and modification dates
- **One-Click Download**: Download button appears when project is completed
- **Meaningful Names**: ZIP files are named based on your project description

### API Usage

#### Create a Project
```bash
curl -X POST "http://127.0.0.1:8000/projects" \
  -H "Content-Type: application/json" \
  -d '"Create a web application for task management"'
```

#### Check Project Status
```bash
curl -X GET "http://127.0.0.1:8000/projects/{project_id}"
```

#### List All Projects
```bash
curl -X GET "http://127.0.0.1:8000/projects"
```

#### Download Project as ZIP
```bash
curl -X GET "http://127.0.0.1:8000/projects/{project_id}/download" \
  -o my_project.zip
```

#### List Project Files
```bash
curl -X GET "http://127.0.0.1:8000/projects/{project_id}/files"
```

## Agent Pipeline

The system uses a sophisticated multi-agent pipeline:

### 1. Prompt Enhancement
- **Agent**: Prompt Enhancer
- **Role**: Takes user prompts and makes them comprehensive
- **Output**: Detailed technical specifications

### 2. Code Generation
- **Agent**: Solutions Architect
- **Role**: Generates complete, production-ready code
- **Output**: Full implementation with best practices

### 3. Code Execution
- **Agent**: Code Executor
- **Role**: Executes generated code and creates files
- **Output**: Working application files

### 4. Testing
- **Agent**: Quality Assurance
- **Role**: Generates and runs comprehensive tests
- **Output**: Test results and validation

### 5. Deployment
- **Agent**: Deployment Specialist
- **Role**: Creates deployment configurations and scripts
- **Output**: Docker files, deployment scripts

## Project Structure

```
agent_workspace/
├── projects/          # Individual project directories
│   └── {project_id}/
│       ├── app.py
│       ├── requirements.txt
│       ├── Dockerfile
│       └── ...
└── deploy/           # Deployment configurations
```

## Configuration

### Environment Variables
- `GOOGLE_API_KEY`: Your Google API key for Gemini
- Server runs on: `127.0.0.1:8000` (configurable in code)

### Customization
- **Workspace Directory**: Change `WORKSPACE_DIR` in the code
- **Server Host/Port**: Modify `SERVER_HOST` and `SERVER_PORT`
- **Agent Instructions**: Customize system messages in `create_agents()`

## Monitoring

### Logs
- Server logs: `agent_hub.log`
- Real-time console output
- Detailed error tracking

### Health Check
```bash
curl -X GET "http://127.0.0.1:8000/health"
```

## Example Projects

### Basic Web App
```
Create a web application for task management with user authentication
```

### Data Analysis Tool
```
Build a data analysis dashboard with charts and CSV upload functionality
```

### API Service
```
Create a REST API for managing inventory with CRUD operations
```

## Troubleshooting

### Unicode Issues
The server has been updated to handle Unicode encoding issues. If you still see encoding errors, they're cosmetic only - the server works fine.

### Clean Output
The server now provides cleaner console output without emoji encoding errors. All logging messages are properly formatted and readable.

### API Key Issues
- Ensure `GOOGLE_API_KEY` is set correctly
- Check API quotas and billing

### Port Issues
- Change `SERVER_PORT` if 8000 is in use
- Check firewall settings

## Advanced Features

### Project Management
- Delete projects via API or UI
- Download completed projects as ZIP files
- View detailed project information and file listings
- Real-time file size and modification tracking

### Real-time Updates
- Web UI refreshes every 5 seconds
- API provides current status
- Background processing with async support

### Error Recovery
- Comprehensive error handling
- Failed projects are marked and logged
- Detailed error messages for debugging

## Development

### Adding New Agents
1. Define agent in `create_agents()` function
2. Add to pipeline in `process_project_full_pipeline()`
3. Update UI status tracking

### Custom Workflows
- Modify pipeline steps in `process_project_full_pipeline()`
- Add new project status types
- Customize agent interactions

---

**Happy coding with your Advanced Agent Hub!**
