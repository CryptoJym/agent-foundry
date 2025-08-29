# Codex Instance 2 Brief: Test Suite & Physics Lab

## Your Task
Build a comprehensive test suite for the core engines and implement the Physics Lab interface.

## Prerequisites
- Core engines reviewed and approved
- Repository: https://github.com/CryptoJym/agent-foundry
- UI framework being built by Instance 1

## Deliverables

### 1. Test Suite (`tests/`)
Create comprehensive tests for all core modules:

#### `tests/test_schema.py`
- Test model creation and validation
- Test serialization/deserialization
- Test registry loading/saving
- Test edge cases (missing fields, invalid data)

#### `tests/test_vram.py`
- Validate calculations against known benchmarks
- Test different quantization levels
- Test KV cache with various parameters
- Test GPU recommendations
- Test optimization algorithm

#### `tests/test_latency.py`
- Test prefill/decode calculations
- Validate hardware multipliers
- Test different network conditions
- Compare with real-world benchmarks

#### `tests/test_scoring.py`
- Test scoring algorithm with various criteria
- Validate task-to-specialization mapping
- Test ranking with multiple models
- Ensure deterministic results

### 2. Physics Lab Interface (`src/ui/physics_lab.py`)
Interactive calculators and visualizations:

#### VRAM Calculator
- Input controls for:
  - Model size (parameter slider)
  - Quantization method
  - Batch size
  - Sequence length
  - KV cache settings
- Real-time VRAM breakdown chart
- GPU compatibility matrix
- "Will it fit?" indicator

#### Latency Estimator
- Input controls for:
  - Model selection
  - Hardware type
  - Network condition
  - Input/output tokens
- Visualization of:
  - TTFT vs token count
  - Generation speed chart
  - Latency breakdown (pie chart)
- Deployment recommendations

#### Batch Calculator
- Find optimal batch size for hardware
- Show throughput vs latency tradeoff
- Memory usage visualization

## Code Requirements

### Testing Framework
- Use pytest for all tests
- Aim for >90% code coverage
- Include both unit and integration tests
- Add performance benchmarks

### Example Test Structure
```python
# tests/test_vram.py
import pytest
from src.core.vram import calculate_weights_vram, calculate_total_vram

class TestVRAMCalculations:
    def test_weight_calculation_fp16(self):
        """Test FP16 weight calculation for 7B model."""
        vram_gb = calculate_weights_vram(7.0, quantization="fp16")
        expected = 7.0 * 2 * 1.2 / 1024  # 7B * 2 bytes * overhead / GiB
        assert abs(vram_gb - expected) < 0.1
        
    def test_kv_cache_scaling(self):
        """Test KV cache scales with sequence length."""
        # Test implementation
```

### Physics Lab UI Pattern
```python
# src/ui/physics_lab.py
import streamlit as st
import plotly.graph_objects as go
from src.core import calculate_total_vram, estimate_latency

def render_physics_lab():
    st.header("🔬 Physics Lab")
    
    tab1, tab2, tab3 = st.tabs(["VRAM Calculator", "Latency Estimator", "Batch Optimizer"])
    
    with tab1:
        render_vram_calculator()
    # ...
```

## Testing Checklist
- [ ] All core functions have unit tests
- [ ] Edge cases are covered
- [ ] Performance benchmarks pass
- [ ] Physics Lab calculations match core engine results
- [ ] Visualizations update smoothly
- [ ] UI handles invalid inputs gracefully

## Notes
- Reference the existing calculations in the test sections of each core module
- Make visualizations interactive and educational
- Include tooltips explaining the physics concepts
- Consider adding "presets" for common scenarios