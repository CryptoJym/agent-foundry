"""VRAM calculation engine for Agent Foundry."""

from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class Quantization(str, Enum):
    """Quantization bit widths."""
    INT4 = "int4"
    INT8 = "int8"
    FP16 = "fp16"
    FP32 = "fp32"
    

class KVDType(str, Enum):
    """KV cache data types."""
    FP8 = "fp8"
    FP16 = "fp16"
    FP32 = "fp32"


@dataclass
class VRAMBreakdown:
    """Detailed breakdown of VRAM usage."""
    weights_gib: float
    kv_cache_gib: float
    activation_gib: float
    overhead_gib: float
    total_gib: float
    
    def fits_gpu(self, gpu_vram_gib: float) -> bool:
        """Check if model fits within GPU VRAM."""
        return self.total_gib <= gpu_vram_gib
    
    def utilization_percent(self, gpu_vram_gib: float) -> float:
        """Calculate VRAM utilization percentage."""
        return (self.total_gib / gpu_vram_gib) * 100 if gpu_vram_gib > 0 else 0


def get_bytes_per_param(quantization: Quantization) -> float:
    """Get bytes per parameter for different quantization levels."""
    mapping = {
        Quantization.INT4: 0.5,    # 4 bits = 0.5 bytes
        Quantization.INT8: 1.0,    # 8 bits = 1 byte
        Quantization.FP16: 2.0,    # 16 bits = 2 bytes
        Quantization.FP32: 4.0,    # 32 bits = 4 bytes
    }
    return mapping.get(quantization, 2.0)


def get_kv_bytes(kv_dtype: KVDType) -> float:
    """Get bytes per KV cache element."""
    mapping = {
        KVDType.FP8: 1.0,     # 8 bits = 1 byte
        KVDType.FP16: 2.0,    # 16 bits = 2 bytes
        KVDType.FP32: 4.0,    # 32 bits = 4 bytes
    }
    return mapping.get(kv_dtype, 2.0)


def calculate_weights_vram(
    active_params_b: float,
    quantization: Quantization = Quantization.FP16,
    overhead_factor: float = 1.2
) -> float:
    """
    Calculate VRAM required for model weights.
    
    Args:
        active_params_b: Active parameters in billions
        quantization: Quantization type
        overhead_factor: Overhead multiplier for framework/buffer overhead
        
    Returns:
        VRAM required in GiB
    """
    bytes_per_param = get_bytes_per_param(quantization)
    bytes_total = active_params_b * 1e9 * bytes_per_param
    bytes_with_overhead = bytes_total * overhead_factor
    return bytes_with_overhead / (1024**3)  # Convert to GiB


def calculate_kv_cache(
    total_tokens: int,
    num_layers: int,
    d_model: int,
    kv_groups: int = 1,
    kv_dtype: KVDType = KVDType.FP16,
    batch_size: int = 1
) -> float:
    """
    Calculate VRAM required for KV cache.
    
    For transformer models with attention mechanisms:
    - Each layer stores K and V matrices
    - With grouped-query attention (GQA), KV heads are shared
    
    Args:
        total_tokens: Sum of input and output tokens
        num_layers: Number of transformer layers
        d_model: Model hidden dimension
        kv_groups: Number of KV groups for GQA (1 = MHA, >1 = GQA)
        kv_dtype: Data type for KV cache
        batch_size: Batch size for inference
        
    Returns:
        KV cache VRAM in GiB
    """
    # KV cache size per token per layer
    # Factor of 2 for K and V, divided by kv_groups for GQA
    kv_size_per_token_per_layer = (2.0 * d_model / max(1, kv_groups))
    
    # Total elements
    total_elements = total_tokens * num_layers * kv_size_per_token_per_layer * batch_size
    
    # Total bytes
    bytes_per_element = get_kv_bytes(kv_dtype)
    total_bytes = total_elements * bytes_per_element
    
    return total_bytes / (1024**3)  # Convert to GiB


def calculate_activation_memory(
    batch_size: int,
    sequence_length: int,
    d_model: int,
    num_layers: int,
    is_training: bool = False
) -> float:
    """
    Calculate activation memory (intermediate tensors during forward pass).
    
    Args:
        batch_size: Batch size
        sequence_length: Maximum sequence length
        d_model: Model dimension
        num_layers: Number of layers
        is_training: Whether training (needs gradients)
        
    Returns:
        Activation memory in GiB
    """
    # Rough estimate: ~4x hidden size per token per layer
    # (attention scores, FFN intermediates, layer norm, etc.)
    memory_per_token = d_model * 4 * 4  # 4 bytes per float32
    
    if is_training:
        memory_per_token *= 2  # Gradients double memory
        
    total_tokens = batch_size * sequence_length
    total_bytes = total_tokens * memory_per_token * num_layers
    
    return total_bytes / (1024**3)


def calculate_total_vram(
    active_params_b: float,
    total_tokens: int,
    model_config: Dict,
    quantization: Quantization = Quantization.FP16,
    kv_dtype: KVDType = KVDType.FP16,
    batch_size: int = 1,
    overhead_factor: float = 1.2,
    include_activation: bool = True
) -> VRAMBreakdown:
    """
    Calculate total VRAM requirements with detailed breakdown.
    
    Args:
        active_params_b: Active parameters in billions
        total_tokens: Total tokens (input + output)
        model_config: Model configuration with layers, d_model, kv_groups
        quantization: Weight quantization
        kv_dtype: KV cache data type
        batch_size: Batch size
        overhead_factor: General overhead factor
        include_activation: Include activation memory
        
    Returns:
        VRAMBreakdown with detailed memory usage
    """
    # Weights
    weights_vram = calculate_weights_vram(
        active_params_b, 
        quantization, 
        overhead_factor
    )
    
    # KV Cache
    kv_vram = calculate_kv_cache(
        total_tokens,
        model_config.get('layers', 32),
        model_config.get('d_model', 4096),
        model_config.get('kv_groups', 1),
        kv_dtype,
        batch_size
    )
    
    # Activations
    activation_vram = 0
    if include_activation:
        activation_vram = calculate_activation_memory(
            batch_size,
            total_tokens,
            model_config.get('d_model', 4096),
            model_config.get('layers', 32),
            is_training=False
        )
    
    # Framework overhead (CUDA kernels, buffers, etc.)
    framework_overhead = 2.0  # ~2 GiB typical overhead
    
    # Total
    total = weights_vram + kv_vram + activation_vram + framework_overhead
    
    return VRAMBreakdown(
        weights_gib=weights_vram,
        kv_cache_gib=kv_vram,
        activation_gib=activation_vram,
        overhead_gib=framework_overhead,
        total_gib=total
    )


def recommend_gpu(vram_required_gib: float, safety_margin: float = 1.1) -> Dict[str, bool]:
    """
    Recommend suitable GPUs based on VRAM requirements.
    
    Args:
        vram_required_gib: Required VRAM in GiB
        safety_margin: Safety margin multiplier
        
    Returns:
        Dict of GPU names to whether they're suitable
    """
    # Common GPU configurations
    gpus = {
        "RTX 4090 24GB": 24,
        "RTX A5000 24GB": 24,
        "L4 24GB": 24,
        "L40S 48GB": 48,
        "A40 48GB": 48,
        "A100 40GB": 40,
        "A100 80GB": 80,
        "H100 80GB": 80,
        "H200 141GB": 141,
    }
    
    required_with_margin = vram_required_gib * safety_margin
    
    recommendations = {}
    for gpu_name, gpu_vram in gpus.items():
        recommendations[gpu_name] = gpu_vram >= required_with_margin
        
    return recommendations


def optimize_for_vram_constraint(
    target_vram_gib: float,
    model_params_b: float,
    model_config: Dict,
    max_tokens: int
) -> Dict[str, any]:
    """
    Find optimal settings to fit within VRAM constraint.
    
    Args:
        target_vram_gib: Target VRAM limit
        model_params_b: Model parameters in billions
        model_config: Model configuration
        max_tokens: Maximum tokens needed
        
    Returns:
        Dict with recommended settings
    """
    recommendations = {
        "quantization": None,
        "kv_dtype": None,
        "max_batch_size": 1,
        "fits": False,
        "vram_usage": None
    }
    
    # Try different combinations
    quant_options = [Quantization.INT4, Quantization.INT8, Quantization.FP16]
    kv_options = [KVDType.FP8, KVDType.FP16]
    
    for quant in quant_options:
        for kv_dtype in kv_options:
            vram = calculate_total_vram(
                model_params_b,
                max_tokens,
                model_config,
                quantization=quant,
                kv_dtype=kv_dtype,
                batch_size=1
            )
            
            if vram.total_gib <= target_vram_gib:
                recommendations["quantization"] = quant
                recommendations["kv_dtype"] = kv_dtype
                recommendations["fits"] = True
                recommendations["vram_usage"] = vram
                
                # Try to find max batch size
                for batch in range(1, 33):  # Up to 32
                    test_vram = calculate_total_vram(
                        model_params_b,
                        max_tokens,
                        model_config,
                        quantization=quant,
                        kv_dtype=kv_dtype,
                        batch_size=batch
                    )
                    if test_vram.total_gib > target_vram_gib:
                        recommendations["max_batch_size"] = batch - 1
                        break
                    recommendations["max_batch_size"] = batch
                
                return recommendations
    
    # If nothing fits, return best attempt
    vram = calculate_total_vram(
        model_params_b,
        max_tokens,
        model_config,
        quantization=Quantization.INT4,
        kv_dtype=KVDType.FP8,
        batch_size=1
    )
    recommendations["quantization"] = Quantization.INT4
    recommendations["kv_dtype"] = KVDType.FP8
    recommendations["vram_usage"] = vram
    
    return recommendations


# Testing
if __name__ == "__main__":
    # Test with Qwen2.5-72B
    model_config = {
        "layers": 80,
        "d_model": 8192,
        "kv_groups": 8
    }
    
    # Calculate for different scenarios
    print("=== Qwen2.5-72B VRAM Requirements ===\n")
    
    scenarios = [
        ("FP16 weights, FP16 KV", Quantization.FP16, KVDType.FP16),
        ("INT8 weights, FP16 KV", Quantization.INT8, KVDType.FP16),
        ("INT4 weights, FP8 KV", Quantization.INT4, KVDType.FP8),
    ]
    
    for name, quant, kv_dtype in scenarios:
        vram = calculate_total_vram(
            active_params_b=72,
            total_tokens=4096,
            model_config=model_config,
            quantization=quant,
            kv_dtype=kv_dtype,
            batch_size=1
        )
        
        print(f"{name}:")
        print(f"  Weights: {vram.weights_gib:.2f} GiB")
        print(f"  KV Cache: {vram.kv_cache_gib:.2f} GiB")
        print(f"  Activations: {vram.activation_gib:.2f} GiB")
        print(f"  Overhead: {vram.overhead_gib:.2f} GiB")
        print(f"  Total: {vram.total_gib:.2f} GiB")
        print(f"  Fits A100-80GB: {'✓' if vram.fits_gpu(80) else '✗'}")
        print()
    
    # Test optimization
    print("=== Optimization for RTX 4090 24GB ===")
    recommendations = optimize_for_vram_constraint(
        target_vram_gib=24,
        model_params_b=72,
        model_config=model_config,
        max_tokens=2048
    )
    
    print(f"Recommended settings:")
    print(f"  Quantization: {recommendations['quantization']}")
    print(f"  KV dtype: {recommendations['kv_dtype']}")
    print(f"  Max batch size: {recommendations['max_batch_size']}")
    print(f"  Fits: {recommendations['fits']}")
    if recommendations['vram_usage']:
        print(f"  Total VRAM: {recommendations['vram_usage'].total_gib:.2f} GiB")