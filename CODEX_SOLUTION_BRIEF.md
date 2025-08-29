# Codex Instance 3 Brief: Solution Composer & Tuning Triage

## Your Task
Implement the Solution Composer (model ranking) and Tuning Triage (optimization) interfaces.

## Prerequisites
- Core engines reviewed and approved
- Repository: https://github.com/CryptoJym/agent-foundry
- UI framework from Instance 1
- Test suite from Instance 2

## Deliverables

### 1. Solution Composer (`src/ui/solution_composer.py`)
Build the model recommendation interface:

#### Task Definition Panel
- Task type selector (dropdown with all TaskType options)
- Context requirements:
  - Minimum context (slider)
  - Target context (slider)
- Performance requirements:
  - Max TTFT (ms)
  - Target TPS
- Hardware constraints:
  - GPU selection (dropdown)
  - Available VRAM
  - Network condition
- License requirements

#### Results Dashboard
- Ranked list of models with:
  - Letter grades (A+, A, etc.)
  - Score breakdown (radar chart)
  - Key metrics (TTFT, TPS, VRAM)
  - Warnings/limitations
- Detailed comparison view
- Export recommendations

#### Scoring Customization
- Adjustable weights (sliders)
- Save/load weight presets
- Explain scoring methodology

### 2. Tuning Triage (`src/ui/tuning_triage.py`)
Optimization and deployment helper:

#### Model Optimizer
- Select target model
- Input hardware constraints
- Show optimization options:
  - Quantization recommendations
  - Batch size optimization
  - Context length tradeoffs
- Before/after comparison

#### Deployment Wizard
- Step-by-step deployment guide
- Configuration generator
- Server recommendation (vLLM, TGI, etc.)
- Cost estimator (cloud pricing)

#### Fine-tuning Advisor
- Dataset size calculator
- Training time estimator
- Hardware requirements
- LoRA vs full fine-tuning recommendation

### 3. Export Functionality (`src/ui/export_utils.py`)
Create export functions for:
- Model recommendations (PDF/JSON)
- Deployment configurations (YAML)
- Comparison reports (HTML)
- Team sharing links

## Code Structure

### Solution Composer Pattern
```python
# src/ui/solution_composer.py
import streamlit as st
from src.core import rank_models, ScoringCriteria, TaskType
from .components import render_model_score_card

def render_solution_composer():
    st.header("🎼 Solution Composer")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        criteria = build_criteria_form()
    
    with col2:
        if criteria:
            scores = rank_models(st.session_state.models, criteria)
            render_recommendations(scores)
```

### Integration Points
- Use `src.core.scoring` for all ranking logic
- Use `src.core.vram.optimize_for_vram_constraint()`
- Generate configs compatible with popular serving frameworks

## UI/UX Requirements
- Make complex choices simple with good defaults
- Provide educational tooltips
- Show visual comparisons (charts, not just tables)
- Enable "what-if" scenario testing
- Make exports immediately useful

## Testing Checklist
- [ ] All task types produce sensible recommendations
- [ ] Custom weights affect rankings appropriately
- [ ] Optimization suggestions are practical
- [ ] Exports contain all necessary information
- [ ] Deployment configs actually work
- [ ] UI responds quickly to parameter changes

## Advanced Features (if time permits)
- Multi-objective optimization (Pareto frontier)
- Team collaboration features
- Saved searches and preferences
- A/B testing recommendations
- Integration with model registries (HuggingFace)

## Notes
- Focus on making recommendations actionable
- Consider different user expertise levels
- Make the "why" behind recommendations clear
- Test with real-world scenarios from the examples