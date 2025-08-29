# 🛠️ Agent Foundry

An interactive tool for selecting and understanding open-weight language models. Turn the complex physics of model deployment into an intuitive, educational experience for your team.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/agent-foundry.git
cd agent-foundry

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py

# Or run the enhanced version
streamlit run src/app_v2.py
```

Visit http://localhost:8501 to start exploring models!

## 🎯 What is Agent Foundry?

Agent Foundry is a lightweight web application that helps teams:

1. **Understand** the physics of language models (VRAM, latency, throughput)
2. **Select** the right open-weight model for their use case
3. **Configure** deployment parameters with confidence
4. **Export** production-ready configuration files

### Key Modules

- **🔍 Model Atlas**: Browse and compare open-weight models
- **🧮 Physics Lab**: Calculate VRAM, latency, and deployment requirements
- **🎯 Solution Composer**: Get recommendations based on your constraints
- **🔧 Tuning Triage**: Decide when to use LoRA/QLoRA vs. larger models

## 📊 Features

### Model Selection
- Filter by task (general, code, RAG, planning)
- Enforce license constraints (permissive, open-weight)
- Match hardware requirements (GPU VRAM)
- Meet latency SLOs

### Physics Calculations
- **Weights VRAM**: `parameters × bits/8 × overhead`
- **KV Cache**: `tokens × layers × 2 × d_model / kv_groups × bytes`
- **Latency**: `TTFT + decode_time`

### Export Formats
- YAML configuration for deployment
- JSON for programmatic use
- Human-readable recommendations

## 🏗️ Architecture

```
agent-foundry/
├── app.py                 # Original Streamlit MVP
├── src/
│   └── app_v2.py         # Enhanced version with external registry
├── data/
│   └── models_registry.json  # Model database
├── docs/
│   └── guides/           # User guides and tutorials
├── scripts/
│   └── update_models.py  # Registry maintenance
└── requirements.txt      # Python dependencies
```

## 📚 Model Registry

The tool includes popular open-weight models:

- **Qwen2.5** family (72B, 32B-Coder)
- **Llama 3.1** (70B)
- **Mixtral** (8x22B MoE)
- **DBRX** (132B MoE)
- **DeepSeek-Coder-V2** (16B)
- **Gemma2** (27B)
- **StarCoder2** (15B)

To add new models, edit `data/models_registry.json`:

```json
{
  "name": "ModelName-XXB",
  "family": "ModelFamily",
  "size_b": 70,
  "active_b": 70,
  "context_k": 32,
  "specialization": ["general", "code"],
  "license": "Apache-2.0",
  "arch": "dense",
  "notes": "Key characteristics",
  "defaults": {
    "layers": 80,
    "d_model": 8192,
    "kv_groups": 8
  }
}
```

## 🎓 Educational Features

### Interactive Learning
- Tooltips explain each parameter
- Real-time VRAM calculations
- Visual GPU fit indicators
- Score breakdowns show trade-offs

### Challenge Exercises
1. **Make it Fit**: Fit a 70B model on a 24GB GPU
2. **Latency Hunt**: Achieve <500ms latency with maximum quality
3. **Cost Optimize**: Balance performance and cloud costs

### Team Workshops
- Model selection races
- VRAM optimization challenges
- Fine-tuning decision trees

## 🔧 Configuration

### Environment Variables
```bash
# Optional: Use external model registry
export MODEL_REGISTRY_PATH="/path/to/custom/models.json"

# Optional: Set default GPU
export DEFAULT_GPU="A100 80GB"
```

### Customization
- Modify scoring weights in `rank_models()`
- Add new GPU types to `GPUS` array
- Extend specialization categories
- Customize recommendation logic

## 📈 Roadmap

- [ ] More models (Phi, Yi, Falcon)
- [ ] Cost calculator for cloud providers
- [ ] Multi-GPU deployment planning
- [ ] Fine-tuning dataset estimator
- [ ] A/B testing configurations
- [ ] Integration with serving frameworks
- [ ] Model performance benchmarks
- [ ] Team learning tracking

## 🤝 Contributing

We welcome contributions! To add models or features:

1. Fork the repository
2. Create a feature branch
3. Add your changes
4. Submit a pull request

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## 📝 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

Built with:
- [Streamlit](https://streamlit.io/) for the web interface
- Model parameters from official model cards
- Inspiration from the vLLM and TGI communities

---

**Remember**: The best model is the one that fits your constraints! 🎯