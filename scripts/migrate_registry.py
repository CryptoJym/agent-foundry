"""Migrate existing model registry to new comprehensive schema."""

import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.schema import (
    ModelSchema, ModelDefaults, Architecture, License, 
    Specialization, QuantizationSupport, HardwareRequirements,
    save_registry
)


def migrate_legacy_model(legacy_data: dict) -> ModelSchema:
    """Convert legacy model format to new schema."""
    
    # Extract defaults
    defaults_data = legacy_data.get('defaults', {})
    defaults = ModelDefaults(
        layers=defaults_data.get('layers', 1),
        d_model=defaults_data.get('d_model', 1),
        kv_groups=defaults_data.get('kv_groups', 1)
    )
    
    # Map specializations
    specialization_map = {
        'general': Specialization.GENERAL,
        'code': Specialization.CODE,
        'sql': Specialization.SQL,
        'math': Specialization.MATH,
        'vision': Specialization.VISION,
        'rag': Specialization.RAG,
        'planning': Specialization.PLANNING,
        'tool-use': Specialization.TOOL_USE
    }
    
    specs = []
    for spec in legacy_data.get('specialization', []):
        if spec in specialization_map:
            specs.append(specialization_map[spec])
        else:
            specs.append(spec)  # Keep as string if not mapped
    
    # Determine architecture
    arch = Architecture.MOE if legacy_data.get('arch') == 'moe' else Architecture.DENSE
    
    # Estimate hardware requirements based on model size
    size_b = legacy_data.get('size_b', 0)
    if size_b <= 15:
        min_vram = 8
        rec_vram = 16
    elif size_b <= 35:
        min_vram = 16
        rec_vram = 24
    elif size_b <= 75:
        min_vram = 40
        rec_vram = 80
    else:
        min_vram = 80
        rec_vram = 160
    
    hardware = HardwareRequirements(
        min_vram_gb=min_vram,
        recommended_vram_gb=rec_vram,
        min_ram_gb=min_vram * 2
    )
    
    # Create new model schema
    return ModelSchema(
        name=legacy_data.get('name', ''),
        family=legacy_data.get('family', ''),
        size_b=size_b,
        active_b=legacy_data.get('active_b', size_b),
        context_k=legacy_data.get('context_k', 32),
        arch=arch,
        specialization=specs,
        license=legacy_data.get('license', 'Custom'),
        defaults=defaults,
        quantization=QuantizationSupport(),
        hardware=hardware,
        notes=legacy_data.get('notes', '')
    )


def main():
    """Migrate the existing registry."""
    # Paths
    legacy_path = Path(__file__).parent.parent / "data" / "models_registry.json"
    new_path = Path(__file__).parent.parent / "data" / "models_registry_v2.json"
    
    # Load legacy registry
    with open(legacy_path, 'r') as f:
        legacy_models = json.load(f)
    
    # Migrate each model
    migrated_models = {}
    for model_data in legacy_models:
        try:
            model = migrate_legacy_model(model_data)
            migrated_models[model.name] = model
            print(f"✓ Migrated {model.name}")
        except Exception as e:
            print(f"✗ Failed to migrate {model_data.get('name', 'unknown')}: {e}")
    
    # Save new registry
    save_registry(migrated_models, new_path)
    print(f"\nMigrated {len(migrated_models)} models to {new_path}")
    
    # Also create a backward-compatible version
    compat_path = legacy_path.with_suffix('.backup')
    legacy_path.rename(compat_path)
    print(f"Backed up original to {compat_path}")
    
    # Save a simplified version for backward compatibility
    simple_models = []
    for model in migrated_models.values():
        simple = {
            "name": model.name,
            "family": model.family,
            "size_b": model.size_b,
            "active_b": model.active_b,
            "context_k": model.context_k,
            "specialization": [s.value if isinstance(s, Specialization) else s 
                              for s in model.specialization],
            "license": model.license.value if isinstance(model.license, License) else model.license,
            "arch": model.arch.value,
            "notes": model.notes,
            "defaults": {
                "layers": model.defaults.layers,
                "d_model": model.defaults.d_model,
                "kv_groups": model.defaults.kv_groups
            }
        }
        simple_models.append(simple)
    
    with open(legacy_path, 'w') as f:
        json.dump(simple_models, f, indent=2)
    print(f"Created backward-compatible registry at {legacy_path}")


if __name__ == "__main__":
    main()