"""Model scoring and ranking engine for Agent Foundry."""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import math

from .schema import ModelSchema, Specialization
from .vram import calculate_total_vram, VRAMBreakdown
from .latency import estimate_latency, LatencyBreakdown, HardwareType, NetworkCondition


class TaskType(str, Enum):
    """High-level task categories for model selection."""
    GENERAL_CHAT = "general_chat"
    CODE_GENERATION = "code_generation"
    CODE_COMPLETION = "code_completion"
    SQL_QUERIES = "sql_queries"
    MATH_PROBLEMS = "math_problems"
    CREATIVE_WRITING = "creative_writing"
    TECHNICAL_WRITING = "technical_writing"
    SUMMARIZATION = "summarization"
    TRANSLATION = "translation"
    RAG_QA = "rag_qa"
    AGENT_PLANNING = "agent_planning"
    TOOL_USE = "tool_use"
    VISION_TASKS = "vision_tasks"
    STRUCTURED_OUTPUT = "structured_output"


@dataclass
class ScoringWeights:
    """Configurable weights for scoring algorithm."""
    task_fit: float = 0.40      # How well model matches task
    context_fit: float = 0.15   # Context window adequacy
    latency_score: float = 0.15 # Speed requirements
    vram_fit: float = 0.15      # Hardware fit
    license_fit: float = 0.10   # License compatibility
    safety_margin: float = 0.05 # Conservative choices
    
    def normalize(self):
        """Ensure weights sum to 1.0."""
        total = (self.task_fit + self.context_fit + self.latency_score + 
                self.vram_fit + self.license_fit + self.safety_margin)
        if total > 0:
            self.task_fit /= total
            self.context_fit /= total
            self.latency_score /= total
            self.vram_fit /= total
            self.license_fit /= total
            self.safety_margin /= total


@dataclass
class ScoringCriteria:
    """Criteria for scoring models."""
    task_type: TaskType
    min_context_k: int = 4  # Minimum context in thousands
    target_context_k: Optional[int] = None  # Ideal context length
    max_ttft_ms: Optional[float] = None  # Max time to first token
    target_tps: Optional[float] = None  # Target tokens per second
    gpu_vram_gb: Optional[float] = None  # Available VRAM
    require_open_license: bool = False  # Require permissive license
    hardware_type: HardwareType = HardwareType.GPU_DATACENTER
    network_condition: NetworkCondition = NetworkCondition.DATACENTER
    prefer_quantized: bool = True  # Prefer quantized models
    weights: ScoringWeights = field(default_factory=ScoringWeights)


@dataclass
class ModelScore:
    """Detailed scoring breakdown for a model."""
    model_name: str
    total_score: float  # 0-100
    task_fit_score: float
    context_score: float
    latency_score: float
    vram_score: float
    license_score: float
    safety_score: float
    
    # Detailed metrics
    can_fit_vram: bool
    meets_latency: bool
    has_specialization: bool
    warnings: List[str] = field(default_factory=list)
    
    # Performance estimates
    estimated_ttft_ms: Optional[float] = None
    estimated_tps: Optional[float] = None
    estimated_vram_gb: Optional[float] = None
    
    def get_grade(self) -> str:
        """Convert score to letter grade."""
        if self.total_score >= 90:
            return "A+"
        elif self.total_score >= 85:
            return "A"
        elif self.total_score >= 80:
            return "A-"
        elif self.total_score >= 75:
            return "B+"
        elif self.total_score >= 70:
            return "B"
        elif self.total_score >= 65:
            return "B-"
        elif self.total_score >= 60:
            return "C+"
        elif self.total_score >= 55:
            return "C"
        else:
            return "D"


def get_task_specializations(task_type: TaskType) -> List[Specialization]:
    """Map task types to relevant model specializations."""
    mapping = {
        TaskType.GENERAL_CHAT: [Specialization.GENERAL, Specialization.CHAT],
        TaskType.CODE_GENERATION: [Specialization.CODE, Specialization.INSTRUCT],
        TaskType.CODE_COMPLETION: [Specialization.CODE],
        TaskType.SQL_QUERIES: [Specialization.SQL, Specialization.CODE],
        TaskType.MATH_PROBLEMS: [Specialization.MATH, Specialization.GENERAL],
        TaskType.CREATIVE_WRITING: [Specialization.GENERAL, Specialization.CHAT],
        TaskType.TECHNICAL_WRITING: [Specialization.GENERAL, Specialization.INSTRUCT],
        TaskType.SUMMARIZATION: [Specialization.GENERAL, Specialization.INSTRUCT],
        TaskType.TRANSLATION: [Specialization.GENERAL],
        TaskType.RAG_QA: [Specialization.RAG, Specialization.GENERAL],
        TaskType.AGENT_PLANNING: [Specialization.PLANNING, Specialization.TOOL_USE],
        TaskType.TOOL_USE: [Specialization.TOOL_USE, Specialization.INSTRUCT],
        TaskType.VISION_TASKS: [Specialization.VISION],
        TaskType.STRUCTURED_OUTPUT: [Specialization.INSTRUCT, Specialization.TOOL_USE],
    }
    return mapping.get(task_type, [Specialization.GENERAL])


def calculate_task_fit_score(
    model: ModelSchema,
    task_type: TaskType
) -> Tuple[float, bool]:
    """
    Calculate how well a model fits the task.
    
    Returns:
        Tuple of (score 0-100, has_specialization)
    """
    required_specs = get_task_specializations(task_type)
    
    # Check exact matches
    exact_matches = sum(1 for spec in required_specs if spec in model.specialization)
    has_specialization = exact_matches > 0
    
    # Base score
    if exact_matches >= len(required_specs):
        base_score = 100
    elif exact_matches > 0:
        base_score = 80 + (20 * exact_matches / len(required_specs))
    elif Specialization.GENERAL in model.specialization:
        base_score = 60  # General models can do most tasks
    else:
        base_score = 40  # Mismatched specialization
    
    # Bonus for instruction-tuned models
    if Specialization.INSTRUCT in model.specialization:
        base_score = min(100, base_score + 5)
    
    # Penalty for using vision model on non-vision tasks
    if (Specialization.VISION in model.specialization and 
        task_type != TaskType.VISION_TASKS):
        base_score *= 0.8
    
    return base_score, has_specialization


def calculate_context_score(
    model: ModelSchema,
    criteria: ScoringCriteria
) -> float:
    """Calculate context window fit score."""
    model_context_k = model.context_k
    
    # Fail if below minimum
    if model_context_k < criteria.min_context_k:
        return 0
    
    # Perfect if exactly at target
    if criteria.target_context_k and model_context_k == criteria.target_context_k:
        return 100
    
    # Score based on how well it meets needs
    if criteria.target_context_k:
        if model_context_k >= criteria.target_context_k:
            # Bonus for extra context, but diminishing returns
            excess_ratio = model_context_k / criteria.target_context_k
            return min(100, 85 + (15 * math.log(excess_ratio)))
        else:
            # Penalty for insufficient context
            ratio = model_context_k / criteria.target_context_k
            return max(50, 100 * ratio)
    else:
        # No specific target, score based on absolute context
        if model_context_k >= 128:
            return 100
        elif model_context_k >= 32:
            return 85
        elif model_context_k >= 8:
            return 70
        else:
            return 60


def calculate_latency_score(
    model: ModelSchema,
    criteria: ScoringCriteria,
    estimated_ttft: float,
    estimated_tps: float
) -> Tuple[float, bool]:
    """
    Calculate latency score based on performance targets.
    
    Returns:
        Tuple of (score 0-100, meets_requirements)
    """
    meets_requirements = True
    scores = []
    
    # TTFT scoring
    if criteria.max_ttft_ms:
        if estimated_ttft <= criteria.max_ttft_ms:
            # Better than target
            ratio = criteria.max_ttft_ms / estimated_ttft
            ttft_score = min(100, 70 + (30 * math.log(ratio)))
        else:
            # Worse than target
            ratio = estimated_ttft / criteria.max_ttft_ms
            ttft_score = max(0, 100 - (50 * (ratio - 1)))
            meets_requirements = False
        scores.append(ttft_score)
    
    # TPS scoring
    if criteria.target_tps:
        if estimated_tps >= criteria.target_tps:
            # Better than target
            ratio = estimated_tps / criteria.target_tps
            tps_score = min(100, 70 + (30 * math.log(ratio)))
        else:
            # Worse than target
            ratio = criteria.target_tps / estimated_tps
            tps_score = max(0, 100 - (50 * (ratio - 1)))
            meets_requirements = False
        scores.append(tps_score)
    
    # If no specific requirements, score based on absolute performance
    if not scores:
        if estimated_tps >= 100:
            tps_score = 90
        elif estimated_tps >= 50:
            tps_score = 80
        elif estimated_tps >= 20:
            tps_score = 70
        else:
            tps_score = 50
        scores.append(tps_score)
        
        if estimated_ttft <= 500:
            ttft_score = 90
        elif estimated_ttft <= 1000:
            ttft_score = 80
        elif estimated_ttft <= 2000:
            ttft_score = 70
        else:
            ttft_score = 50
        scores.append(ttft_score)
    
    return sum(scores) / len(scores), meets_requirements


def calculate_vram_score(
    model: ModelSchema,
    criteria: ScoringCriteria,
    vram_breakdown: VRAMBreakdown
) -> Tuple[float, bool]:
    """
    Calculate VRAM fit score.
    
    Returns:
        Tuple of (score 0-100, fits_in_vram)
    """
    if not criteria.gpu_vram_gb:
        # No VRAM constraint
        return 85, True
    
    total_vram = vram_breakdown.total_gib
    available_vram = criteria.gpu_vram_gb
    
    if total_vram > available_vram:
        # Doesn't fit
        return 0, False
    
    # Calculate utilization
    utilization = total_vram / available_vram
    
    if utilization <= 0.7:
        # Excellent - lots of headroom
        return 100, True
    elif utilization <= 0.85:
        # Good - reasonable headroom
        return 85, True
    elif utilization <= 0.95:
        # Tight but workable
        return 70, True
    else:
        # Very tight, risky
        return 50, True


def calculate_license_score(
    model: ModelSchema,
    criteria: ScoringCriteria
) -> float:
    """Calculate license compatibility score."""
    if not criteria.require_open_license:
        # No license requirement
        return 90
    
    # Check if license is permissive
    permissive_licenses = ["Apache-2.0", "MIT", "Permissive"]
    
    if any(lic in str(model.license) for lic in permissive_licenses):
        return 100
    elif "OpenRAIL" in str(model.license):
        return 80  # Some restrictions but generally open
    else:
        return 40  # Restrictive or unclear license


def score_model(
    model: ModelSchema,
    criteria: ScoringCriteria,
    typical_input_tokens: int = 512,
    typical_output_tokens: int = 512
) -> ModelScore:
    """
    Score a model against criteria.
    
    Args:
        model: Model to score
        criteria: Scoring criteria
        typical_input_tokens: Typical input length for latency estimation
        typical_output_tokens: Typical output length for latency estimation
        
    Returns:
        ModelScore with detailed breakdown
    """
    # Ensure weights are normalized
    criteria.weights.normalize()
    
    # Task fit
    task_score, has_spec = calculate_task_fit_score(model, criteria.task_type)
    
    # Context fit
    context_score = calculate_context_score(model, criteria)
    
    # Estimate performance
    latency = estimate_latency(
        typical_input_tokens,
        typical_output_tokens,
        model.active_b,  # Use active params for latency
        hardware=criteria.hardware_type,
        network=criteria.network_condition,
        is_quantized=criteria.prefer_quantized
    )
    
    latency_score, meets_latency = calculate_latency_score(
        model, 
        criteria,
        latency.ttft_ms,
        latency.tps
    )
    
    # VRAM calculation
    vram = calculate_total_vram(
        model.active_b,
        typical_input_tokens + typical_output_tokens,
        {
            "layers": model.defaults.layers,
            "d_model": model.defaults.d_model,
            "kv_groups": model.defaults.kv_groups
        },
        batch_size=1
    )
    
    vram_score, fits_vram = calculate_vram_score(model, criteria, vram)
    
    # License score
    license_score = calculate_license_score(model, criteria)
    
    # Safety margin (prefer well-tested models)
    safety_score = 85  # Base safety score
    if model.family in ["Llama", "Qwen", "Mistral"]:
        safety_score = 95  # Well-tested families
    elif "experimental" in model.notes.lower():
        safety_score = 60
    
    # Calculate weighted total
    total_score = (
        criteria.weights.task_fit * task_score +
        criteria.weights.context_fit * context_score +
        criteria.weights.latency_score * latency_score +
        criteria.weights.vram_fit * vram_score +
        criteria.weights.license_fit * license_score +
        criteria.weights.safety_margin * safety_score
    )
    
    # Compile warnings
    warnings = []
    if not fits_vram:
        warnings.append(f"Requires {vram.total_gib:.1f}GB VRAM, exceeds {criteria.gpu_vram_gb}GB available")
    if not meets_latency:
        warnings.append(f"TTFT {latency.ttft_ms:.0f}ms or TPS {latency.tps:.1f} below targets")
    if not has_spec:
        warnings.append(f"No specific optimization for {criteria.task_type.value}")
    if context_score < 70:
        warnings.append(f"Context window may be insufficient")
    
    return ModelScore(
        model_name=model.name,
        total_score=total_score,
        task_fit_score=task_score,
        context_score=context_score,
        latency_score=latency_score,
        vram_score=vram_score,
        license_score=license_score,
        safety_score=safety_score,
        can_fit_vram=fits_vram,
        meets_latency=meets_latency,
        has_specialization=has_spec,
        warnings=warnings,
        estimated_ttft_ms=latency.ttft_ms,
        estimated_tps=latency.tps,
        estimated_vram_gb=vram.total_gib
    )


def rank_models(
    models: List[ModelSchema],
    criteria: ScoringCriteria,
    top_n: Optional[int] = None,
    min_score: float = 0.0
) -> List[ModelScore]:
    """
    Rank models by score.
    
    Args:
        models: List of models to rank
        criteria: Scoring criteria
        top_n: Return only top N models
        min_score: Minimum score threshold
        
    Returns:
        List of ModelScore objects sorted by total score
    """
    scores = []
    
    for model in models:
        try:
            score = score_model(model, criteria)
            if score.total_score >= min_score:
                scores.append(score)
        except Exception as e:
            # Skip models that fail scoring
            print(f"Warning: Failed to score {model.name}: {e}")
            continue
    
    # Sort by total score descending
    scores.sort(key=lambda x: x.total_score, reverse=True)
    
    if top_n:
        scores = scores[:top_n]
    
    return scores


def generate_recommendation_report(
    scores: List[ModelScore],
    criteria: ScoringCriteria
) -> str:
    """Generate human-readable recommendations."""
    if not scores:
        return "No models meet the specified criteria."
    
    report = []
    report.append(f"=== Model Recommendations for {criteria.task_type.value} ===\n")
    
    # Top recommendation
    top = scores[0]
    report.append(f"🏆 Top Choice: {top.model_name}")
    report.append(f"   Grade: {top.get_grade()} ({top.total_score:.1f}/100)")
    report.append(f"   TTFT: {top.estimated_ttft_ms:.0f}ms | TPS: {top.estimated_tps:.1f}")
    report.append(f"   VRAM: {top.estimated_vram_gb:.1f}GB")
    
    if top.warnings:
        report.append("   ⚠️  Warnings:")
        for warning in top.warnings:
            report.append(f"      - {warning}")
    
    report.append("")
    
    # Alternatives
    if len(scores) > 1:
        report.append("Alternative Options:")
        for i, score in enumerate(scores[1:4], 1):  # Next 3
            report.append(f"{i+1}. {score.model_name} - {score.get_grade()} ({score.total_score:.1f}/100)")
            if score.warnings:
                report.append(f"   ⚠️  {score.warnings[0]}")
    
    return "\n".join(report)


# Testing
if __name__ == "__main__":
    from .schema import ModelDefaults, Architecture, License
    
    # Create test models
    test_models = [
        ModelSchema(
            name="Qwen2.5-72B-Instruct",
            family="Qwen2.5",
            size_b=72,
            active_b=72,
            context_k=128,
            arch=Architecture.DENSE,
            specialization=[Specialization.GENERAL, Specialization.INSTRUCT, Specialization.TOOL_USE],
            license=License.OPEN_WEIGHT,
            defaults=ModelDefaults(layers=80, d_model=8192, kv_groups=8),
            notes="Excellent for structured output and tool use"
        ),
        ModelSchema(
            name="Llama-3-70B",
            family="Llama",
            size_b=70,
            active_b=70,
            context_k=8,
            arch=Architecture.DENSE,
            specialization=[Specialization.GENERAL, Specialization.CHAT],
            license=License.COMMUNITY,
            defaults=ModelDefaults(layers=80, d_model=8192, kv_groups=8),
            notes="Popular open model"
        ),
        ModelSchema(
            name="Codestral-22B",
            family="Mistral",
            size_b=22,
            active_b=22,
            context_k=32,
            arch=Architecture.DENSE,
            specialization=[Specialization.CODE],
            license=License.CUSTOM,
            defaults=ModelDefaults(layers=56, d_model=6144, kv_groups=8),
            notes="Specialized for code"
        )
    ]
    
    # Test criteria
    criteria = ScoringCriteria(
        task_type=TaskType.CODE_GENERATION,
        min_context_k=8,
        target_context_k=32,
        max_ttft_ms=1000,
        target_tps=50,
        gpu_vram_gb=24,  # RTX 4090
        hardware_type=HardwareType.GPU_CONSUMER,
        network_condition=NetworkCondition.LOCAL,
        prefer_quantized=True
    )
    
    # Rank models
    scores = rank_models(test_models, criteria)
    
    # Print results
    print(generate_recommendation_report(scores, criteria))
    print("\n=== Detailed Scores ===")
    for score in scores:
        print(f"\n{score.model_name}:")
        print(f"  Task Fit: {score.task_fit_score:.1f}")
        print(f"  Context: {score.context_score:.1f}")
        print(f"  Latency: {score.latency_score:.1f}")
        print(f"  VRAM: {score.vram_score:.1f}")
        print(f"  License: {score.license_score:.1f}")
        print(f"  Safety: {score.safety_score:.1f}")