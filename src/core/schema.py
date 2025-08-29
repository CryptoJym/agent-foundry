"""Model registry schema and validation for Agent Foundry."""

import json
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union
from dataclasses import dataclass, field, asdict
from enum import Enum


class Architecture(str, Enum):
    """Model architecture types."""
    DENSE = "dense"
    MOE = "moe"  # Mixture of Experts
    SPARSE = "sparse"


class License(str, Enum):
    """Common model license types."""
    APACHE_2_0 = "Apache-2.0"
    MIT = "MIT"
    OPENRAIL_M = "OpenRAIL-M"
    OPEN_WEIGHT = "Open-Weight"
    COMMUNITY = "Community"
    PERMISSIVE = "Permissive"
    CUSTOM = "Custom"


class Specialization(str, Enum):
    """Model specialization areas."""
    GENERAL = "general"
    CODE = "code"
    SQL = "sql"
    MATH = "math"
    VISION = "vision"
    RAG = "rag"
    PLANNING = "planning"
    TOOL_USE = "tool-use"
    CHAT = "chat"
    INSTRUCT = "instruct"


@dataclass
class ModelDefaults:
    """Default configuration parameters for a model."""
    layers: int
    d_model: int  # Model dimension/hidden size
    kv_groups: int = 1  # Number of KV attention groups (for GQA)
    n_heads: Optional[int] = None  # Number of attention heads
    vocab_size: Optional[int] = None  # Vocabulary size
    rope_theta: Optional[float] = None  # RoPE theta for positional encoding
    

@dataclass
class QuantizationSupport:
    """Supported quantization methods and their characteristics."""
    int4: bool = True
    int8: bool = True
    fp16: bool = True
    gptq: bool = False
    awq: bool = False
    gguf: bool = False
    optimal_quant: Optional[str] = "int4"  # Recommended quantization
    

@dataclass
class HardwareRequirements:
    """Minimum and recommended hardware requirements."""
    min_vram_gb: float  # Minimum VRAM for basic operation
    recommended_vram_gb: float  # Recommended for good performance
    min_ram_gb: Optional[float] = None  # System RAM requirements
    optimal_gpu: Optional[List[str]] = None  # Best GPUs for this model
    

@dataclass
class PerformanceMetrics:
    """Expected performance characteristics."""
    typical_prefill_tps: Optional[float] = None  # Tokens per second (prefill)
    typical_decode_tps: Optional[float] = None  # Tokens per second (decode)
    batch_size_impact: Optional[str] = None  # How batch size affects performance
    

@dataclass
class ModelSchema:
    """Comprehensive schema for model registry entries."""
    # Basic identification
    name: str
    family: str
    
    # Size and architecture
    size_b: float  # Total parameters in billions
    active_b: float  # Active parameters per token (for MoE)
    context_k: int  # Maximum context length in thousands
    arch: Architecture = Architecture.DENSE
    version: Optional[str] = None
    
    # Capabilities
    specialization: List[Specialization] = field(default_factory=list)
    license: Union[License, str] = License.CUSTOM
    
    # Technical details
    defaults: ModelDefaults = field(default_factory=ModelDefaults)
    quantization: QuantizationSupport = field(default_factory=QuantizationSupport)
    
    # Requirements and performance
    hardware: Optional[HardwareRequirements] = None
    performance: Optional[PerformanceMetrics] = None
    
    # Additional metadata
    notes: str = ""
    release_date: Optional[str] = None
    paper_url: Optional[str] = None
    model_card_url: Optional[str] = None
    base_model: Optional[str] = None  # For fine-tuned models
    training_data: Optional[str] = None  # Brief description
    
    # Serving compatibility
    vllm_compatible: bool = True
    tgi_compatible: bool = True
    tensorrt_compatible: bool = False
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        # Convert enums to strings
        data['arch'] = self.arch.value
        if isinstance(self.license, License):
            data['license'] = self.license.value
        data['specialization'] = [s.value if isinstance(s, Specialization) else s 
                                  for s in self.specialization]
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ModelSchema':
        """Create ModelSchema from dictionary."""
        # Handle nested dataclasses
        if 'defaults' in data and isinstance(data['defaults'], dict):
            data['defaults'] = ModelDefaults(**data['defaults'])
        if 'quantization' in data and isinstance(data['quantization'], dict):
            data['quantization'] = QuantizationSupport(**data['quantization'])
        if 'hardware' in data and data['hardware'] and isinstance(data['hardware'], dict):
            data['hardware'] = HardwareRequirements(**data['hardware'])
        if 'performance' in data and data['performance'] and isinstance(data['performance'], dict):
            data['performance'] = PerformanceMetrics(**data['performance'])
            
        # Handle enums
        if 'arch' in data:
            data['arch'] = Architecture(data['arch'])
        if 'license' in data:
            try:
                data['license'] = License(data['license'])
            except ValueError:
                # Keep as string if not a standard license
                pass
        if 'specialization' in data:
            specs = []
            for spec in data['specialization']:
                try:
                    specs.append(Specialization(spec))
                except ValueError:
                    specs.append(spec)  # Keep as string if not standard
            data['specialization'] = specs
            
        return cls(**data)


def validate_model(model: Union[ModelSchema, Dict]) -> tuple[bool, List[str]]:
    """
    Validate a model entry against the schema.
    
    Returns:
        tuple: (is_valid, list_of_errors)
    """
    errors = []
    
    if isinstance(model, dict):
        try:
            model = ModelSchema.from_dict(model)
        except Exception as e:
            return False, [f"Failed to parse model: {str(e)}"]
    
    # Validation rules
    if not model.name:
        errors.append("Model name is required")
    
    if model.size_b <= 0:
        errors.append("Model size must be positive")
        
    if model.active_b <= 0 or model.active_b > model.size_b:
        errors.append("Active parameters must be positive and <= total parameters")
        
    if model.context_k <= 0:
        errors.append("Context length must be positive")
        
    if not model.specialization:
        errors.append("At least one specialization is required")
        
    if model.defaults.layers <= 0:
        errors.append("Number of layers must be positive")
        
    if model.defaults.d_model <= 0:
        errors.append("Model dimension must be positive")
        
    if model.defaults.kv_groups < 1:
        errors.append("KV groups must be >= 1")
        
    # Hardware validation if provided
    if model.hardware:
        if model.hardware.min_vram_gb <= 0:
            errors.append("Minimum VRAM must be positive")
        if model.hardware.recommended_vram_gb < model.hardware.min_vram_gb:
            errors.append("Recommended VRAM must be >= minimum VRAM")
            
    return len(errors) == 0, errors


def load_registry(registry_path: Union[str, Path]) -> Dict[str, ModelSchema]:
    """
    Load model registry from JSON file.
    
    Returns:
        Dict mapping model names to ModelSchema objects
    """
    registry_path = Path(registry_path)
    if not registry_path.exists():
        raise FileNotFoundError(f"Registry file not found: {registry_path}")
        
    with open(registry_path, 'r') as f:
        data = json.load(f)
    
    models = {}
    for model_data in data:
        try:
            # Handle legacy format conversion
            if 'defaults' not in model_data and all(k in model_data for k in ['layers', 'd_model']):
                # Extract defaults from flat structure
                model_data['defaults'] = {
                    'layers': model_data.pop('layers', 1),
                    'd_model': model_data.pop('d_model', 1),
                    'kv_groups': model_data.pop('kv_groups', 1)
                }
            
            model = ModelSchema.from_dict(model_data)
            models[model.name] = model
        except Exception as e:
            print(f"Warning: Failed to load model {model_data.get('name', 'unknown')}: {str(e)}")
            
    return models


def save_registry(models: Dict[str, ModelSchema], registry_path: Union[str, Path]):
    """Save model registry to JSON file."""
    registry_path = Path(registry_path)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    
    model_list = [model.to_dict() for model in models.values()]
    
    with open(registry_path, 'w') as f:
        json.dump(model_list, f, indent=2)


# Example usage and testing
if __name__ == "__main__":
    # Example model creation
    qwen_72b = ModelSchema(
        name="Qwen2.5-72B-Instruct",
        family="Qwen2.5",
        size_b=72,
        active_b=72,
        context_k=128,
        arch=Architecture.DENSE,
        specialization=[Specialization.GENERAL, Specialization.PLANNING, Specialization.TOOL_USE],
        license=License.OPEN_WEIGHT,
        defaults=ModelDefaults(
            layers=80,
            d_model=8192,
            kv_groups=8,
            n_heads=64,
            vocab_size=152064
        ),
        quantization=QuantizationSupport(
            int4=True,
            int8=True,
            fp16=True,
            gptq=True,
            awq=True,
            optimal_quant="int4"
        ),
        hardware=HardwareRequirements(
            min_vram_gb=36,
            recommended_vram_gb=80,
            min_ram_gb=64,
            optimal_gpu=["A100-80GB", "H100-80GB"]
        ),
        performance=PerformanceMetrics(
            typical_prefill_tps=3000,
            typical_decode_tps=150,
            batch_size_impact="Linear scaling up to batch 8"
        ),
        notes="Long context, strong structured output capabilities",
        release_date="2024-09",
        model_card_url="https://huggingface.co/Qwen/Qwen2.5-72B-Instruct",
        vllm_compatible=True,
        tgi_compatible=True,
        tensorrt_compatible=True
    )
    
    # Validate
    is_valid, errors = validate_model(qwen_72b)
    print(f"Qwen2.5-72B validation: {'PASSED' if is_valid else 'FAILED'}")
    if errors:
        for error in errors:
            print(f"  - {error}")
            
    # Test serialization
    model_dict = qwen_72b.to_dict()
    reconstructed = ModelSchema.from_dict(model_dict)
    print(f"Serialization test: {'PASSED' if reconstructed.name == qwen_72b.name else 'FAILED'}")