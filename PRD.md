# Product Requirements Document: Agent Foundry

## Executive Summary

Agent Foundry is an interactive web application that transforms how engineering teams select and deploy open-weight language models. By making the "physics" of models tangible—VRAM requirements, latency calculations, quantization impacts—it enables informed decisions and prevents costly deployment failures.

## Problem Statement

### Current Challenges
- **Knowledge Gap**: Engineers treat models as black boxes, lacking understanding of memory requirements, latency implications, and hardware constraints
- **Poor Selection**: Teams choose models based on benchmark scores alone, leading to deployment failures when models don't fit hardware or meet latency SLOs
- **Premature Fine-tuning**: Without understanding model capabilities, teams waste resources fine-tuning when they should use larger models or RAG
- **Fragmented Information**: Model details scattered across papers, model cards, and blog posts with no unified comparison tool

### Impact
- Failed deployments waste weeks of engineering time
- Over-provisioned hardware increases cloud costs by 40-60%
- Suboptimal model choices degrade product quality
- Teams lack confidence in their decisions

## Solution Overview

Agent Foundry provides four integrated modules:

1. **Model Atlas**: Searchable catalog of open-weight models with standardized comparisons
2. **Physics Lab**: Interactive calculators that visualize VRAM usage, latency, and throughput
3. **Solution Composer**: Guided wizard that generates deployment configurations based on constraints
4. **Tuning Triage**: Decision framework for when to fine-tune vs. use larger models

## User Personas

### Primary: ML Engineer
- **Goal**: Deploy models that meet performance SLOs within hardware budget
- **Pain**: Uncertainty about model requirements leads to over-provisioning
- **Gain**: Confident model selection with predictable performance

### Secondary: Tech Lead
- **Goal**: Make informed build-vs-buy decisions for AI capabilities
- **Pain**: Can't evaluate feasibility without deep ML knowledge
- **Gain**: Data-driven recommendations for technical planning

### Tertiary: Solutions Architect
- **Goal**: Design scalable AI systems for clients
- **Pain**: Each client has different constraints requiring custom analysis
- **Gain**: Rapid prototyping tool that exports production configs

## Core Features

### MVP (Weeks 1-4)

#### 1. Model Registry
- 10+ popular open-weight models (Qwen, Llama, Mixtral, etc.)
- Standardized schema: size, architecture, context length, specializations
- JSON-based for easy updates
- Filtering by task, license, architecture

#### 2. Physics Calculators
- **Weights VRAM**: `params × bits/8 × overhead`
- **KV Cache**: `tokens × layers × 2 × d_model / kv_groups × bytes`
- **Latency**: `TTFT + decode_time` with throughput estimates
- Real-time updates as parameters change

#### 3. Scoring Algorithm
```
Score = 40×TaskFit + 15×ContextFit + 15×LatencyFit 
        + 15×VRAMFit + 10×LicenseFit + 5×MarginBonus
```

#### 4. Export Functionality
- YAML deployment configs for vLLM/TGI
- JSON for programmatic integration
- Human-readable recommendations

#### 5. Educational Layer
- Tooltips explaining each parameter
- "Why this matters" micro-lessons
- Interactive exercises

### Post-MVP (Months 2-3)

#### Enhanced Features
- 50+ models including specialized variants
- Multi-GPU deployment planning
- Cost calculator for AWS/GCP/Azure
- A/B testing configurations
- Fine-tuning dataset size estimator

#### Integrations
- Import from HuggingFace model cards
- Export to serving frameworks
- CI/CD webhook for automated selection
- Slack/Teams notifications

#### Team Features
- Shared model evaluations
- Decision history tracking
- Learning progress gamification
- Custom scoring weights per team

## Technical Architecture

### Frontend
- **Framework**: Streamlit (MVP) → React (Production)
- **Components**: Modular, reusable calculators
- **State**: Session-based with optional persistence
- **Export**: Native file generation

### Backend
- **Language**: Python 3.8+
- **Dependencies**: Minimal (pandas, pyyaml)
- **Model Registry**: JSON with schema validation
- **Calculations**: Numpy-optimized functions

### Deployment
- **MVP**: Single Streamlit instance
- **Production**: Containerized with Kubernetes
- **Database**: Optional PostgreSQL for team features
- **CDN**: Static assets and model registry

### Architecture Principles
- **Modularity**: Separate engines for each calculator
- **Extensibility**: Plugin system for new models
- **Performance**: Client-side calculations when possible
- **Accuracy**: Validated against real deployments

## Implementation Plan

### Phase 1: Core Engine (Weeks 1-2)
- [ ] Design model registry schema
- [ ] Implement VRAM calculator
- [ ] Implement latency estimator  
- [ ] Create scoring algorithm
- [ ] Unit test all calculations

### Phase 2: Streamlit UI (Weeks 3-4)
- [ ] Build Model Atlas browser
- [ ] Create Physics Lab interface
- [ ] Implement Solution Composer wizard
- [ ] Add Tuning Triage logic
- [ ] Enable YAML/JSON export

### Phase 3: Educational Content (Week 5)
- [ ] Write parameter explanations
- [ ] Create interactive exercises
- [ ] Design challenge scenarios
- [ ] Add visual indicators

### Phase 4: Testing & Launch (Week 6)
- [ ] Internal testing with ML team
- [ ] Gather feedback from 5 partner teams
- [ ] Refine calculations based on real data
- [ ] Public release with documentation

## Success Metrics

### Usage Metrics
- Daily active users: 100+ within 3 months
- Models evaluated: 1,000+ per month
- Configs exported: 200+ per month
- Return visitor rate: >60%

### Learning Metrics
- Time to first successful deployment: <2 hours
- Reduction in failed deployments: 50%
- Increase in model diversity: 3x
- Fine-tuning decision accuracy: 80%

### Business Impact
- Hardware cost savings: 30-40%
- Deployment time reduction: 60%
- Model selection confidence: 85%
- Team adoption rate: 80%

## Risks & Mitigations

### Technical Risks
- **Risk**: Calculations don't match reality
- **Mitigation**: Validate against production deployments

- **Risk**: Model registry becomes outdated
- **Mitigation**: Automated updates from HuggingFace

### Adoption Risks
- **Risk**: Teams prefer manual selection
- **Mitigation**: Embed in deployment pipelines

- **Risk**: Too complex for beginners
- **Mitigation**: Progressive disclosure UI

## Future Roadmap

### Quarter 2
- Model performance benchmarks
- Batch inference calculator
- Training cost estimator
- Model merging assistant

### Quarter 3  
- AutoML integration
- Deployment monitoring
- Performance regression alerts
- Custom model import

### Quarter 4
- Enterprise features
- Private model registry
- Compliance checking
- ROI reporting

## Conclusion

Agent Foundry transforms model selection from guesswork to engineering. By making the invisible visible—memory usage, latency, costs—it empowers teams to deploy with confidence. The interactive approach ensures teams don't just get recommendations, but understand why, building lasting expertise.

## Appendix

### Model Registry Schema
```json
{
  "name": "string",
  "family": "string", 
  "size_b": "number",
  "active_b": "number",
  "context_k": "number",
  "specialization": ["string"],
  "license": "string",
  "arch": "enum[dense,moe]",
  "notes": "string",
  "defaults": {
    "layers": "number",
    "d_model": "number",
    "kv_groups": "number"
  }
}
```

### Export Config Schema
```yaml
model:
  name: string
  quantization: int
hardware:
  gpu: string
  count: int
serving:
  runtime: string
  batch_size: int
  features: [string]
performance:
  expected_latency_ms: int
  expected_throughput_qps: int
```