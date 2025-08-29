# Agent Foundry - Collaboration Guide

## 🚀 Quick Start for Codex

### Setup
```bash
git clone https://github.com/CryptoJym/agent-foundry.git
cd agent-foundry
pip install -r requirements.txt
```

### Check Task Status
```bash
# View all tasks
task-master get-tasks --projectRoot . --file .taskmaster/tasks/tasks.json

# Set a task to in-progress
task-master set-task-status --projectRoot . --id 5 --status in-progress --file .taskmaster/tasks/tasks.json
```

## 📋 Current Task Assignments

### Claude (In Progress)
- **Task 1**: Design model registry schema ⚡ IN PROGRESS
- **Task 2**: Build VRAM calculator engine
- **Task 3**: Implement latency estimation module

### Suggested for Codex
- **Task 5**: Develop Streamlit UI framework (no dependencies, can start immediately)
- **Task 12**: Create comprehensive test suite (can work in parallel)
- **Task 6**: Implement Model Atlas browser (after Task 5)

## 🏗️ Architecture Overview

```
agent-foundry/
├── src/
│   ├── core/              # Core calculation engines (Claude working here)
│   │   ├── __init__.py
│   │   ├── schema.py      # Model registry schema
│   │   ├── vram.py        # VRAM calculator
│   │   └── latency.py     # Latency estimator
│   ├── ui/                # UI components (Codex can work here)
│   │   ├── __init__.py
│   │   ├── components.py  # Reusable UI components
│   │   └── layouts.py     # Page layouts
│   └── utils/             # Shared utilities
│       ├── __init__.py
│       └── scoring.py     # Model scoring algorithm
├── tests/                 # Test suite (Codex can set up)
│   ├── test_core.py
│   └── test_ui.py
└── data/
    └── models_registry.json
```

## 🔄 Workflow

1. **Before starting work**: Pull latest changes
   ```bash
   git pull origin main
   ```

2. **Update task status**: Mark your task as in-progress
   ```bash
   task-master set-task-status --projectRoot . --id <TASK_ID> --status in-progress --file .taskmaster/tasks/tasks.json
   ```

3. **Create a branch**: Work on feature branches
   ```bash
   git checkout -b feature/task-<ID>-description
   ```

4. **Regular commits**: Commit progress frequently
   ```bash
   git add .
   git commit -m "Task <ID>: Description of changes"
   git push origin feature/task-<ID>-description
   ```

5. **Complete task**: Update status when done
   ```bash
   task-master set-task-status --projectRoot . --id <TASK_ID> --status done --file .taskmaster/tasks/tasks.json
   ```

## 💡 Key Interfaces

### Model Schema (Task 1 - Claude working on this)
```python
ModelSchema = {
    "name": str,
    "family": str,
    "size_b": float,
    "active_b": float,
    "context_k": int,
    "specialization": List[str],
    "license": str,
    "arch": Literal["dense", "moe"],
    "defaults": {
        "layers": int,
        "d_model": int,
        "kv_groups": int
    }
}
```

### Calculator Interfaces (Tasks 2-3)
```python
def calculate_weights_vram(active_b: float, bits: int, overhead: float = 1.2) -> float
def calculate_kv_cache(tokens: int, layers: int, d_model: int, kv_groups: int, kv_dtype: str) -> float
def estimate_latency(prompt_tokens: int, output_tokens: int, hardware: Dict) -> float
```

### UI Components (Task 5 - Suggested for Codex)
- ModelSelector: Dropdown/search for model selection
- ParameterSliders: Interactive controls for all parameters  
- ResultsDisplay: Tables and visualizations for recommendations
- ExportButtons: YAML/JSON download functionality

## 📝 Notes

- **Parallel Work**: Tasks 1-3 (core) and Task 5 (UI) can be developed independently
- **Testing**: Task 12 can start with basic structure even before implementations
- **Communication**: Update this file with progress notes or blockers
- **Schema First**: Claude will define the model schema first so UI can use it

## 🤝 Communication

Add notes here:
- [Claude] Starting with Task 1 - designing comprehensive model schema
- [Codex] _Add your status here_