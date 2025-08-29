# Codex Instance 1 Brief: UI Framework & Model Atlas

## Your Task
Build the Streamlit UI framework and implement the Model Atlas interface for Agent Foundry.

## Prerequisites
- Core engines have been reviewed and approved
- Repository: https://github.com/CryptoJym/agent-foundry
- Existing modules in `src/core/`: schema.py, vram.py, latency.py, scoring.py

## Deliverables

### 1. UI Framework (`src/ui/app.py`)
Create the main Streamlit application with:
- Clean, modern interface using Streamlit 1.28+
- Navigation between the 4 main modules (tabs or sidebar)
- Consistent styling and layout
- Session state management
- Error handling and loading states

### 2. Model Atlas Interface (`src/ui/model_atlas.py`)
Implement the model browser with:
- Model grid/card view showing key stats
- Filtering by:
  - Model family
  - Size range (parameter count)
  - Specialization (code, chat, etc.)
  - Context window
  - License type
- Search functionality
- Detailed model view with all attributes
- Comparison mode (select 2-3 models)

### 3. Shared Components (`src/ui/components.py`)
Create reusable components:
- Model card component
- Metric display widgets
- Filter controls
- Export buttons

## Integration Requirements
- Use `src.core.schema.load_registry()` to load models
- Display data from ModelSchema objects
- Handle the models_registry_v2.json format

## Code Style
- Follow existing patterns from app.py and app_v2.py
- Use type hints
- Add docstrings to all functions
- Handle errors gracefully
- Make it responsive and user-friendly

## Testing Checklist
- [ ] All models load and display correctly
- [ ] Filters work individually and in combination
- [ ] Search is fast and accurate
- [ ] Model comparison shows meaningful differences
- [ ] UI is responsive on different screen sizes

## Example Code Structure
```python
# src/ui/app.py
import streamlit as st
from src.core import load_registry
from .model_atlas import render_model_atlas

def main():
    st.set_page_config(page_title="Agent Foundry", layout="wide")
    
    # Navigation
    module = st.sidebar.selectbox(
        "Select Module",
        ["Model Atlas", "Physics Lab", "Solution Composer", "Tuning Triage"]
    )
    
    if module == "Model Atlas":
        render_model_atlas()
    # ... other modules
```

## Notes
- The existing app.py and app_v2.py in the root can be referenced for UI patterns
- Focus on making the Model Atlas exceptional - it's the entry point for users
- Ensure smooth performance even with many models loaded