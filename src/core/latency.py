"""Latency estimation engine for Agent Foundry."""

from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class HardwareType(str, Enum):
    """Hardware accelerator types."""
    CPU = "cpu"
    GPU_CONSUMER = "gpu_consumer"  # RTX 4090, etc.
    GPU_DATACENTER = "gpu_datacenter"  # A100, H100
    GPU_CLOUD = "gpu_cloud"  # L4, T4
    TPU = "tpu"
    

class NetworkCondition(str, Enum):
    """Network connection quality."""
    LOCAL = "local"  # Same machine
    LAN = "lan"  # Local network (<1ms)
    DATACENTER = "datacenter"  # Same DC (~1-5ms)
    REGIONAL = "regional"  # Same region (~10-50ms)
    GLOBAL = "global"  # Cross-region (~100-300ms)


@dataclass
class LatencyBreakdown:
    """Detailed breakdown of inference latency."""
    prefill_ms: float  # Time to process input tokens
    decode_ms: float  # Time to generate output tokens
    network_ms: float  # Network round-trip time
    overhead_ms: float  # Framework/API overhead
    total_ms: float
    
    @property
    def ttft_ms(self) -> float:
        """Time to first token (prefill + network + overhead)."""
        return self.prefill_ms + self.network_ms + self.overhead_ms
    
    @property
    def tps(self) -> float:
        """Tokens per second during generation."""
        if self.decode_ms > 0:
            return 1000.0 / self.decode_ms
        return 0
    
    def format_summary(self) -> str:
        """Human-readable latency summary."""
        return (
            f"TTFT: {self.ttft_ms:.0f}ms | "
            f"Generation: {self.tps:.1f} tok/s | "
            f"Total: {self.total_ms:.0f}ms"
        )


def get_hardware_multiplier(hardware: HardwareType) -> Dict[str, float]:
    """Get performance multipliers for different hardware."""
    multipliers = {
        HardwareType.CPU: {
            "prefill_speed": 0.05,  # 5% of GPU speed
            "decode_speed": 0.1,    # 10% of GPU speed
            "overhead": 2.0         # 2x overhead
        },
        HardwareType.GPU_CONSUMER: {
            "prefill_speed": 0.7,   # 70% of datacenter GPU
            "decode_speed": 0.8,    # 80% of datacenter GPU
            "overhead": 1.2         # 20% more overhead
        },
        HardwareType.GPU_DATACENTER: {
            "prefill_speed": 1.0,   # Baseline
            "decode_speed": 1.0,    # Baseline
            "overhead": 1.0         # Baseline
        },
        HardwareType.GPU_CLOUD: {
            "prefill_speed": 0.5,   # 50% of datacenter GPU
            "decode_speed": 0.6,    # 60% of datacenter GPU
            "overhead": 1.3         # 30% more overhead
        },
        HardwareType.TPU: {
            "prefill_speed": 1.5,   # 150% for large batches
            "decode_speed": 0.9,    # 90% for single stream
            "overhead": 0.8         # Optimized overhead
        }
    }
    return multipliers.get(hardware, multipliers[HardwareType.GPU_DATACENTER])


def get_network_latency(condition: NetworkCondition) -> float:
    """Get typical network latency in milliseconds."""
    latencies = {
        NetworkCondition.LOCAL: 0.1,
        NetworkCondition.LAN: 1.0,
        NetworkCondition.DATACENTER: 3.0,
        NetworkCondition.REGIONAL: 25.0,
        NetworkCondition.GLOBAL: 200.0
    }
    return latencies.get(condition, 50.0)


def estimate_prefill_time(
    input_tokens: int,
    model_params_b: float,
    batch_size: int = 1,
    hardware: HardwareType = HardwareType.GPU_DATACENTER,
    is_quantized: bool = False
) -> float:
    """
    Estimate prefill (prompt processing) time.
    
    Prefill is memory-bandwidth bound and processes all tokens in parallel.
    
    Args:
        input_tokens: Number of input tokens
        model_params_b: Model size in billions of parameters
        batch_size: Inference batch size
        hardware: Hardware type
        is_quantized: Whether model is quantized (faster)
        
    Returns:
        Prefill time in milliseconds
    """
    # Base prefill speed (tokens/second) for 7B model on A100
    base_prefill_tps = 3000.0
    
    # Adjust for model size (larger = slower due to memory bandwidth)
    size_factor = 7.0 / model_params_b if model_params_b > 0 else 1.0
    
    # Adjust for batch size (some parallelism benefit)
    batch_factor = 1.0 + (batch_size - 1) * 0.3  # 30% benefit per additional batch
    
    # Adjust for quantization
    quant_factor = 1.5 if is_quantized else 1.0
    
    # Get hardware multiplier
    hw_mult = get_hardware_multiplier(hardware)
    
    # Calculate effective TPS
    effective_tps = (
        base_prefill_tps * 
        size_factor * 
        batch_factor * 
        quant_factor * 
        hw_mult["prefill_speed"]
    )
    
    # Calculate time
    prefill_ms = (input_tokens / effective_tps) * 1000.0
    
    return max(1.0, prefill_ms)  # At least 1ms


def estimate_decode_time(
    output_tokens: int,
    model_params_b: float,
    batch_size: int = 1,
    hardware: HardwareType = HardwareType.GPU_DATACENTER,
    is_quantized: bool = False,
    use_kv_cache: bool = True
) -> float:
    """
    Estimate decode (generation) time.
    
    Decode is compute-bound and processes tokens sequentially.
    
    Args:
        output_tokens: Number of tokens to generate
        model_params_b: Model size in billions of parameters
        batch_size: Inference batch size
        hardware: Hardware type
        is_quantized: Whether model is quantized
        use_kv_cache: Whether KV cache is used (should always be True)
        
    Returns:
        Total decode time in milliseconds
    """
    # Base decode speed (tokens/second) for 7B model on A100
    base_decode_tps = 150.0
    
    # Adjust for model size (larger = slower due to compute)
    size_factor = 7.0 / model_params_b if model_params_b > 0 else 1.0
    
    # Batch size has minimal impact on decode (sequential)
    batch_penalty = 1.0 / (1.0 + (batch_size - 1) * 0.8)  # 80% penalty per batch
    
    # Quantization benefit
    quant_factor = 1.8 if is_quantized else 1.0
    
    # KV cache is essential
    kv_factor = 1.0 if use_kv_cache else 0.1  # 10x slower without KV cache
    
    # Hardware multiplier
    hw_mult = get_hardware_multiplier(hardware)
    
    # Calculate effective TPS
    effective_tps = (
        base_decode_tps * 
        size_factor * 
        batch_penalty * 
        quant_factor * 
        kv_factor *
        hw_mult["decode_speed"]
    )
    
    # Calculate time
    decode_ms = (output_tokens / effective_tps) * 1000.0
    
    return max(1.0, decode_ms)


def calculate_ttft(
    input_tokens: int,
    model_params_b: float,
    hardware: HardwareType = HardwareType.GPU_DATACENTER,
    network: NetworkCondition = NetworkCondition.DATACENTER,
    is_quantized: bool = False,
    batch_size: int = 1
) -> float:
    """
    Calculate Time To First Token (TTFT).
    
    This is the latency users experience before seeing any output.
    
    Args:
        input_tokens: Number of input tokens
        model_params_b: Model size in billions
        hardware: Hardware type
        network: Network condition
        is_quantized: Whether model is quantized
        batch_size: Inference batch size
        
    Returns:
        TTFT in milliseconds
    """
    prefill_ms = estimate_prefill_time(
        input_tokens, 
        model_params_b, 
        batch_size, 
        hardware, 
        is_quantized
    )
    
    network_ms = get_network_latency(network)
    
    # API/framework overhead
    hw_mult = get_hardware_multiplier(hardware)
    overhead_ms = 5.0 * hw_mult["overhead"]  # Base 5ms overhead
    
    return prefill_ms + network_ms + overhead_ms


def estimate_latency(
    input_tokens: int,
    output_tokens: int,
    model_params_b: float,
    hardware: HardwareType = HardwareType.GPU_DATACENTER,
    network: NetworkCondition = NetworkCondition.DATACENTER,
    is_quantized: bool = False,
    batch_size: int = 1
) -> LatencyBreakdown:
    """
    Estimate end-to-end latency for inference.
    
    Args:
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens to generate
        model_params_b: Model size in billions of parameters
        hardware: Hardware type
        network: Network condition
        is_quantized: Whether model is quantized
        batch_size: Inference batch size
        
    Returns:
        LatencyBreakdown with detailed timing
    """
    # Calculate components
    prefill_ms = estimate_prefill_time(
        input_tokens,
        model_params_b,
        batch_size,
        hardware,
        is_quantized
    )
    
    decode_ms = estimate_decode_time(
        output_tokens,
        model_params_b,
        batch_size,
        hardware,
        is_quantized
    )
    
    network_ms = get_network_latency(network)
    
    # Overhead
    hw_mult = get_hardware_multiplier(hardware)
    overhead_ms = 5.0 * hw_mult["overhead"]
    
    # Total
    total_ms = prefill_ms + decode_ms + network_ms + overhead_ms
    
    return LatencyBreakdown(
        prefill_ms=prefill_ms,
        decode_ms=decode_ms,
        network_ms=network_ms,
        overhead_ms=overhead_ms,
        total_ms=total_ms
    )


def compare_configurations(
    input_tokens: int,
    output_tokens: int,
    model_params_b: float,
    configurations: list[Dict]
) -> list[Tuple[str, LatencyBreakdown]]:
    """
    Compare latency across different deployment configurations.
    
    Args:
        input_tokens: Input token count
        output_tokens: Output token count
        model_params_b: Model size in billions
        configurations: List of config dicts with hardware, network, etc.
        
    Returns:
        List of (config_name, latency_breakdown) tuples sorted by total latency
    """
    results = []
    
    for config in configurations:
        latency = estimate_latency(
            input_tokens,
            output_tokens,
            model_params_b,
            hardware=config.get("hardware", HardwareType.GPU_DATACENTER),
            network=config.get("network", NetworkCondition.DATACENTER),
            is_quantized=config.get("quantized", False),
            batch_size=config.get("batch_size", 1)
        )
        results.append((config.get("name", "Unknown"), latency))
    
    # Sort by total latency
    results.sort(key=lambda x: x[1].total_ms)
    
    return results


def recommend_deployment(
    model_params_b: float,
    target_ttft_ms: float,
    target_tps: float,
    typical_input_tokens: int = 512,
    typical_output_tokens: int = 512
) -> Dict[str, any]:
    """
    Recommend deployment configuration to meet latency targets.
    
    Args:
        model_params_b: Model size in billions
        target_ttft_ms: Target time to first token in ms
        target_tps: Target tokens per second
        typical_input_tokens: Typical input length
        typical_output_tokens: Typical output length
        
    Returns:
        Dict with recommended configuration
    """
    recommendations = {
        "model_size": model_params_b,
        "targets": {
            "ttft_ms": target_ttft_ms,
            "tps": target_tps
        },
        "configs": []
    }
    
    # Test different configurations
    test_configs = [
        {
            "name": "Local GPU FP16",
            "hardware": HardwareType.GPU_CONSUMER,
            "network": NetworkCondition.LOCAL,
            "quantized": False
        },
        {
            "name": "Local GPU INT8",
            "hardware": HardwareType.GPU_CONSUMER,
            "network": NetworkCondition.LOCAL,
            "quantized": True
        },
        {
            "name": "Cloud A100 FP16",
            "hardware": HardwareType.GPU_DATACENTER,
            "network": NetworkCondition.REGIONAL,
            "quantized": False
        },
        {
            "name": "Cloud A100 INT8",
            "hardware": HardwareType.GPU_DATACENTER,
            "network": NetworkCondition.REGIONAL,
            "quantized": True
        },
        {
            "name": "Edge L4 INT8",
            "hardware": HardwareType.GPU_CLOUD,
            "network": NetworkCondition.LAN,
            "quantized": True
        }
    ]
    
    for config in test_configs:
        latency = estimate_latency(
            typical_input_tokens,
            typical_output_tokens,
            model_params_b,
            hardware=config["hardware"],
            network=config["network"],
            is_quantized=config["quantized"]
        )
        
        meets_ttft = latency.ttft_ms <= target_ttft_ms
        meets_tps = latency.tps >= target_tps
        
        recommendations["configs"].append({
            "name": config["name"],
            "hardware": config["hardware"],
            "network": config["network"],
            "quantized": config["quantized"],
            "metrics": {
                "ttft_ms": round(latency.ttft_ms, 1),
                "tps": round(latency.tps, 1),
                "total_ms": round(latency.total_ms, 1)
            },
            "meets_targets": meets_ttft and meets_tps,
            "notes": []
        })
        
        # Add notes
        if not meets_ttft:
            recommendations["configs"][-1]["notes"].append(
                f"TTFT {latency.ttft_ms:.0f}ms exceeds target {target_ttft_ms}ms"
            )
        if not meets_tps:
            recommendations["configs"][-1]["notes"].append(
                f"TPS {latency.tps:.1f} below target {target_tps}"
            )
    
    # Sort by whether targets are met, then by total latency
    recommendations["configs"].sort(
        key=lambda x: (not x["meets_targets"], x["metrics"]["total_ms"])
    )
    
    return recommendations


# Alias for backward compatibility
calculate_decode_time = estimate_decode_time


# Testing
if __name__ == "__main__":
    # Test with a 72B model
    print("=== Qwen2.5-72B Latency Estimation ===\n")
    
    # Common workload
    input_tokens = 1024
    output_tokens = 512
    model_size = 72
    
    # Test different deployments
    configs = [
        {
            "name": "RTX 4090 Local INT4",
            "hardware": HardwareType.GPU_CONSUMER,
            "network": NetworkCondition.LOCAL,
            "quantized": True
        },
        {
            "name": "A100 Datacenter FP16",
            "hardware": HardwareType.GPU_DATACENTER,
            "network": NetworkCondition.DATACENTER,
            "quantized": False
        },
        {
            "name": "L4 Edge INT8",
            "hardware": HardwareType.GPU_CLOUD,
            "network": NetworkCondition.LAN,
            "quantized": True
        },
        {
            "name": "CPU Local",
            "hardware": HardwareType.CPU,
            "network": NetworkCondition.LOCAL,
            "quantized": True
        }
    ]
    
    results = compare_configurations(
        input_tokens,
        output_tokens,
        model_size,
        configs
    )
    
    for name, latency in results:
        print(f"{name}:")
        print(f"  TTFT: {latency.ttft_ms:.0f}ms")
        print(f"  Generation: {latency.tps:.1f} tokens/sec")
        print(f"  Total time: {latency.total_ms/1000:.1f}s")
        print()
    
    # Test recommendations
    print("=== Deployment Recommendations ===")
    recs = recommend_deployment(
        model_params_b=72,
        target_ttft_ms=500,  # 500ms TTFT target
        target_tps=50,       # 50 tokens/sec target
        typical_input_tokens=1024,
        typical_output_tokens=512
    )
    
    print(f"\nFinding configs for {recs['model_size']}B model:")
    print(f"Target TTFT: {recs['targets']['ttft_ms']}ms")
    print(f"Target TPS: {recs['targets']['tps']}")
    
    for config in recs["configs"]:
        status = "✓" if config["meets_targets"] else "✗"
        print(f"\n{status} {config['name']}:")
        print(f"  TTFT: {config['metrics']['ttft_ms']}ms")
        print(f"  TPS: {config['metrics']['tps']}")
        if config["notes"]:
            for note in config["notes"]:
                print(f"  ⚠️  {note}")