# Codex Coordination Plan

## Overview
This document coordinates the parallel development of Agent Foundry across multiple Codex instances.

## Phase 1: Code Review (Single Instance)
**Instance**: Codex Reviewer  
**Brief**: CODEX_REVIEW_BRIEF.md  
**Duration**: 30 minutes  
**Output**: Approval to proceed or list of fixes needed  

## Phase 2: Parallel Development (3 Instances)

### Instance 1: UI & Model Atlas
**Brief**: CODEX_UI_BRIEF.md  
**Dependencies**: Core engines approved  
**Key Files**:
- `src/ui/app.py` (main app)
- `src/ui/model_atlas.py` (model browser)
- `src/ui/components.py` (shared components)

### Instance 2: Tests & Physics Lab  
**Brief**: CODEX_TEST_BRIEF.md  
**Dependencies**: Core engines approved  
**Key Files**:
- `tests/test_*.py` (test suite)
- `src/ui/physics_lab.py` (calculators)

### Instance 3: Solution Composer & Tuning
**Brief**: CODEX_SOLUTION_BRIEF.md  
**Dependencies**: Core engines approved, UI framework (partial)  
**Key Files**:
- `src/ui/solution_composer.py` (ranking)
- `src/ui/tuning_triage.py` (optimization)
- `src/ui/export_utils.py` (exports)

## Integration Points

### Shared State
All instances should use:
```python
# Session state for models
if 'models' not in st.session_state:
    st.session_state.models = load_registry('data/models_registry_v2.json')
```

### UI Consistency
- Use the same Streamlit theme settings
- Follow component patterns from Instance 1
- Maintain consistent spacing and layout

### Testing Coverage
Instance 2's tests should cover:
- All core engine functions
- UI component rendering (using Streamlit testing)
- Integration between modules

## Communication Protocol

### Progress Updates
Each instance should create a progress file:
- `PROGRESS_UI.md` (Instance 1)
- `PROGRESS_TEST.md` (Instance 2)  
- `PROGRESS_SOLUTION.md` (Instance 3)

### Blocking Issues
If blocked, create an issue file:
- `BLOCKED_[INSTANCE].md` with:
  - What's blocking
  - What's needed
  - Suggested solution

### Code Reviews
Before integration:
1. Self-review against brief checklist
2. Cross-review another instance's work
3. Claude reviews all work
4. Integration testing

## Timeline
- Phase 1: 30 minutes (review)
- Phase 2: 2-3 hours (parallel development)
- Phase 3: 1 hour (integration)
- Phase 4: 30 minutes (final testing)

## Success Criteria
- [ ] All core engine tests pass
- [ ] UI loads without errors
- [ ] All 4 modules are functional
- [ ] Model recommendations are accurate
- [ ] Export functions work
- [ ]站Performance is acceptable (<1s response)
- [ ] Code is well-documented
- [ ] Ready for production use

## Next Steps After Completion
1. Deploy to Streamlit Cloud
2. Create demo video
3. Write user documentation
4. Gather feedback
5. Plan v2 features