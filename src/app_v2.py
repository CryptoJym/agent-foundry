# app_v2.py — Agent Foundry with External Registry
import math, io, json, yaml, os
import pandas as pd
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Agent Foundry", layout="wide", page_icon="🛠️")

# Load models from external JSON
@st.cache_data
def load_models():
    registry_path = Path(__file__).parent.parent / "data" / "models_registry.json"
    try:
        with open(registry_path, 'r') as f:
            return json.load(f)
    except:
        # Fallback to hardcoded if file not found
        return []

MODELS = load_models()

GPUS = [
    {"name": "A100 80GB", "vram_gib": 80, "prefill_tps_estimate": 3000, "decode_tps_estimate": 200},
    {"name": "H100 80GB", "vram_gib": 80, "prefill_tps_estimate": 5000, "decode_tps_estimate": 300},
    {"name": "L40S 48GB", "vram_gib": 48, "prefill_tps_estimate": 2500, "decode_tps_estimate": 180},
    {"name": "RTX 4090 24GB", "vram_gib": 24, "prefill_tps_estimate": 2000, "decode_tps_estimate": 150},
    {"name": "T4 16GB", "vram_gib": 16, "prefill_tps_estimate": 800, "decode_tps_estimate": 80}
]

# ---------- Helpers ----------
def weights_vram_gib(active_b, bits, overhead=1.2):
    bytes_per_weight = bits / 8.0
    return active_b * 1e9 * bytes_per_weight * overhead / (1024**3)

def kv_cache_gib(tokens_total, layers, d_model, kv_groups, bytes_kv):
    per_token_per_layer_bytes = (2.0 * d_model / max(1, kv_groups)) * bytes_kv
    return tokens_total * layers * per_token_per_layer_bytes / (1024**3)

def est_latency_s(prompt_toks, output_toks, prefill_tps, decode_tps):
    ttft = prompt_toks / max(1e-6, prefill_tps)
    decode = output_toks / max(1e-6, decode_tps)
    return ttft + decode

def fit_score(task, model):
    if task in model["specialization"]:
        return 1.0
    if task == "general" and "general" in model["specialization"]:
        return 1.0
    return 0.6 if ("general" in model["specialization"]) else 0.0

def license_ok(policy, lic):
    if policy == "Permissive-only":
        return lic.lower() in ["apache-2.0","permissive"]
    if policy == "Allow open-weight":
        return lic.lower() in ["apache-2.0","permissive","open-weight","community","openrail-m"]
    return True

def rank_models(models, req, hw, perf):
    rows = []
    for m in models:
        bits = req["weights_bits"]
        kv_bytes = 2 if req["kv_dtype"] == "fp16" else 1
        L = m["defaults"]["layers"]; D = m["defaults"]["d_model"]; G = m["defaults"]["kv_groups"]
        tokens_total = req["prompt_tokens"] + req["output_tokens"]
        w_vram = weights_vram_gib(m["active_b"], bits)
        kv_vram = kv_cache_gib(tokens_total, L, D, G, kv_bytes)
        total_vram = w_vram + kv_vram + req["vram_overhead_gib"]

        meets_vram = total_vram <= hw["vram_gib"]
        meets_ctx = m["context_k"] >= math.ceil(tokens_total/1000)
        meets_lic = license_ok(req["license_policy"], m["license"])
        lat_s = est_latency_s(req["prompt_tokens"], req["output_tokens"], perf["prefill_tps"], perf["decode_tps"])
        meets_slo = lat_s*1000 <= req["latency_ms"]

        taskfit = fit_score(req["task"], m)
        vramfit = 1.0 if meets_vram else 0.0
        ctxfit  = 1.0 if meets_ctx else 0.0
        licfit  = 1.0 if meets_lic else 0.0
        latfit  = min(1.0, req["latency_ms"] / max(1.0, lat_s*1000))
        margin  = max(0.0, (hw["vram_gib"] - total_vram) / hw["vram_gib"])
        score = 40*taskfit + 15*ctxfit + 15*latfit + 15*vramfit + 10*licfit + 5*margin

        rows.append({
            "Model": m["name"], "Family": m["family"], "Arch": m["arch"],
            "Active B": m["active_b"], "Context (k)": m["context_k"],
            "License": m["license"],
            "Weights VRAM (GiB)": round(w_vram,2),
            "KV VRAM (GiB)": round(kv_vram,2),
            "Total VRAM (GiB)": round(total_vram,2),
            "Fits GPU?": "✅" if meets_vram else "❌",
            "Est. Latency (ms)": int(lat_s*1000),
            "Meets SLO?": "✅" if meets_slo else "❌",
            "Task Fit": taskfit,
            "Score": round(score,1),
            "Notes": m["notes"]
        })
    df = pd.DataFrame(rows).sort_values("Score", ascending=False)
    return df

def tuning_recommendation(gap_pct, structured, data_k, slo_ms, lat_ms):
    if lat_ms > slo_ms:
        return "Prefer smaller/faster base; revisit tuning after hitting SLO."
    if gap_pct <= 10 and structured and data_k >= 10:
        return "Proceed with LoRA/QLoRA SFT; consider light DPO for preference."
    if gap_pct <= 15 and structured and data_k >= 5:
        return "Pilot LoRA with small dataset; add retrieval or teacher distillation."
    return "Hold off on tuning; expand data, add RAG, or use larger base temporarily."

# ---------- UI ----------
st.title("🛠️ Agent Foundry — Model Selection & Physics Lab")
st.markdown("*An interactive tool for selecting and understanding open-weight models*")

# Sidebar: requirements
with st.sidebar:
    st.header("🎯 Requirements")
    task = st.selectbox("Primary task", ["general","rag","code","sql","vision","planning"], 
                       help="What will this model primarily be used for?")
    latency_ms = st.slider("Target p95 latency (ms)", 100, 8000, 1500, 50,
                          help="Maximum acceptable latency for 95% of requests")
    
    col1, col2 = st.columns(2)
    with col1:
        prompt_tokens = st.number_input("Prompt tokens", 16, 200000, 2048, 16,
                                      help="Average input size")
    with col2:
        output_tokens = st.number_input("Output tokens", 16, 200000, 512, 16,
                                      help="Average output size")
    
    st.markdown("---")
    st.header("⚖️ Constraints")
    license_policy = st.selectbox("License policy", 
                                ["Allow open-weight", "Permissive-only", "Anything (internal PoC)"],
                                help="Licensing restrictions for your use case")
    
    col3, col4 = st.columns(2)
    with col3:
        weights_bits = st.selectbox("Weight quantization", [4,8,16], index=0,
                                   help="Lower bits = smaller model, slight quality loss")
    with col4:
        kv_dtype = st.selectbox("KV dtype", ["fp16","fp8"], index=1,
                               help="fp8 reduces KV cache memory by 50%")
    
    vram_overhead_gib = st.slider("Extra VRAM overhead (GiB)", 0.0, 8.0, 2.0, 0.5,
                                 help="Buffer for activations, gradients, etc.")
    
    st.markdown("---")
    st.header("🖥️ Hardware")
    gpu = st.selectbox("Target GPU", [g["name"] for g in GPUS], index=0)
    gpu_info = next(g for g in GPUS if g["name"] == gpu)
    
    st.info(f"**{gpu_info['name']}**: {gpu_info['vram_gib']} GiB VRAM")
    
    st.markdown("---")
    st.header("⚡ Throughput")
    st.markdown("*Estimates or measured values*")
    
    use_gpu_defaults = st.checkbox("Use GPU defaults", value=True)
    if use_gpu_defaults:
        prefill_tps = gpu_info["prefill_tps_estimate"]
        decode_tps = gpu_info["decode_tps_estimate"]
        st.write(f"Prefill: ~{prefill_tps} tok/s")
        st.write(f"Decode: ~{decode_tps} tok/s")
    else:
        prefill_tps = st.number_input("Prefill tokens/sec", 100.0, 20000.0, 3000.0, 50.0)
        decode_tps  = st.number_input("Decode tokens/sec", 20.0, 5000.0, 200.0, 5.0)
    
    st.markdown("---")
    run_analysis = st.button("🚀 Rank Models", type="primary", use_container_width=True)

# Main content area
if run_analysis:
    req = {
        "task": task, "latency_ms": latency_ms,
        "prompt_tokens": int(prompt_tokens), "output_tokens": int(output_tokens),
        "license_policy": license_policy, "weights_bits": int(weights_bits),
        "kv_dtype": kv_dtype, "vram_overhead_gib": float(vram_overhead_gib)
    }
    perf = {"prefill_tps": float(prefill_tps), "decode_tps": float(decode_tps)}
    
    df = rank_models(MODELS, req, gpu_info, perf)
    
    # Results section
    col_main, col_side = st.columns([3, 1])
    
    with col_main:
        st.subheader("📊 Recommended Models")
        st.dataframe(df, use_container_width=True, height=400)
    
    with col_side:
        st.subheader("🏆 Top Pick")
        if not df.empty:
            top = df.iloc[0].to_dict()
            st.metric("Model", top['Model'])
            st.metric("Score", f"{top['Score']}/100")
            st.metric("Latency", f"{top['Est. Latency (ms)']} ms")
            st.metric("VRAM", f"{top['Total VRAM (GiB)']} GiB")
    
    # Detailed analysis
    if not df.empty:
        st.markdown("---")
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.markdown("### 📋 Suggested Plan")
            st.markdown(f"**Recommended Model:** `{top['Model']}` ({top['Arch']}, active {top['Active B']}B)")
            st.markdown(f"**Why this model?**")
            st.markdown(f"- Task fit: {round(top['Task Fit']*100)}%")
            st.markdown(f"- Latency: {top['Est. Latency (ms)']} ms (target: {latency_ms} ms)")
            st.markdown(f"- VRAM usage: {top['Total VRAM (GiB)']} / {gpu_info['vram_gib']} GiB")
            st.markdown(f"- Architecture: {top['Arch']}")
            if top['Notes']:
                st.info(f"**Note:** {top['Notes']}")
        
        with col_right:
            st.markdown("### 🔧 Tuning Triage")
            gap = st.slider("Gap to teacher model (%)", 0, 40, 8, 1,
                           help="Performance difference vs. larger model on your metric")
            structured = st.checkbox("Structured outputs?", value=True,
                                   help="JSON/SQL/code/templates")
            data_k = st.slider("Training examples (thousands)", 0, 200, 25, 1)
            
            rec = tuning_recommendation(gap, structured, data_k, latency_ms, top['Est. Latency (ms)'])
            
            if "Proceed" in rec:
                st.success(rec)
            elif "Pilot" in rec:
                st.warning(rec)
            else:
                st.error(rec)
        
        # Export section
        st.markdown("---")
        st.subheader("💾 Export Configuration")
        
        solution = {
            "timestamp": str(pd.Timestamp.now()),
            "use_case": task,
            "hardware": {"gpu": gpu_info["name"], "vram_gib": gpu_info["vram_gib"]},
            "requirements": {
                "latency_slo_ms": latency_ms,
                "prompt_tokens": prompt_tokens,
                "output_tokens": output_tokens,
                "license_policy": license_policy
            },
            "model": {
                "name": top["Model"],
                "family": top["Family"],
                "architecture": top["Arch"],
                "active_parameters_b": top["Active B"],
                "quantization": {
                    "weights_bits": int(weights_bits),
                    "kv_dtype": kv_dtype
                }
            },
            "serving": {
                "runtime": "vLLM",
                "features": [
                    "continuous-batching",
                    "paged-attention",
                    "speculative-decoding (optional)"
                ],
                "estimated_performance": {
                    "prefill_tps": prefill_tps,
                    "decode_tps": decode_tps,
                    "p95_latency_ms": top['Est. Latency (ms)']
                }
            },
            "rag": {
                "enabled": task in ["rag", "general"],
                "notes": "Prefer retrieval over very long contexts."
            },
            "tuning": {
                "recommendation": rec,
                "method": "LoRA/QLoRA if recommended",
                "validation": ["JSON schema compliance", "unit tests", "A/B testing"]
            }
        }
        
        col_yaml, col_json = st.columns(2)
        
        with col_yaml:
            yml = yaml.dump(solution, sort_keys=False, default_flow_style=False)
            st.download_button(
                "📄 Download solution.yaml",
                data=yml.encode(),
                file_name="solution.yaml",
                mime="text/yaml"
            )
        
        with col_json:
            jsn = json.dumps(solution, indent=2)
            st.download_button(
                "📄 Download solution.json",
                data=jsn.encode(),
                file_name="solution.json",
                mime="application/json"
            )

# Tabs for additional content
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Model Atlas", "🧮 Physics Lab", "📚 Learning Resources", "ℹ️ About"])

with tab1:
    st.markdown("### Model Atlas — Browse & Compare")
    
    search_col, filter_col = st.columns([2, 1])
    with search_col:
        search_term = st.text_input("Search models", placeholder="e.g., Qwen, code, MoE...")
    with filter_col:
        arch_filter = st.selectbox("Architecture", ["All", "dense", "moe"])
    
    filtered_models = MODELS
    if search_term:
        filtered_models = [m for m in filtered_models if search_term.lower() in str(m).lower()]
    if arch_filter != "All":
        filtered_models = [m for m in filtered_models if m["arch"] == arch_filter]
    
    if filtered_models:
        df = pd.DataFrame([{
            "Model": m["name"],
            "Family": m["family"],
            "Architecture": m["arch"],
            "Total Params (B)": m["size_b"],
            "Active Params (B)": m["active_b"],
            "Context (k)": m["context_k"],
            "Specialization": ", ".join(m["specialization"]),
            "License": m["license"]
        } for m in filtered_models])
        st.dataframe(df, use_container_width=True, height=600)
    else:
        st.info("No models match your search criteria")

with tab2:
    st.markdown("### Physics Lab — Understanding Model Requirements")
    
    col_calc, col_explain = st.columns([1, 1])
    
    with col_calc:
        st.markdown("#### 🧮 Quick Calculator")
        
        calc_model = st.selectbox("Select model", [m["name"] for m in MODELS])
        calc_model_info = next(m for m in MODELS if m["name"] == calc_model)
        
        calc_bits = st.select_slider("Quantization", [4, 8, 16], value=4)
        calc_tokens = st.slider("Total tokens (prompt + output)", 1000, 100000, 10000, 1000)
        calc_kv_fp8 = st.checkbox("Use FP8 KV cache", value=True)
        
        # Calculate
        w_vram = weights_vram_gib(calc_model_info["active_b"], calc_bits)
        kv_vram = kv_cache_gib(
            calc_tokens,
            calc_model_info["defaults"]["layers"],
            calc_model_info["defaults"]["d_model"],
            calc_model_info["defaults"]["kv_groups"],
            1 if calc_kv_fp8 else 2
        )
        total = w_vram + kv_vram + 2.0  # overhead
        
        st.markdown("##### Results:")
        st.metric("Weights VRAM", f"{w_vram:.2f} GiB")
        st.metric("KV Cache VRAM", f"{kv_vram:.2f} GiB")
        st.metric("Total VRAM (with overhead)", f"{total:.2f} GiB")
        
        # GPU fit check
        st.markdown("##### Can it fit?")
        for gpu in GPUS:
            if total <= gpu["vram_gib"]:
                st.success(f"✅ Fits on {gpu['name']}")
            else:
                st.error(f"❌ Too large for {gpu['name']}")
    
    with col_explain:
        st.markdown("#### 📖 Key Concepts")
        
        with st.expander("**Latency Calculation**"):
            st.markdown("""
            Total latency = Prefill + Generation
            
            - **Prefill (TTFT)**: `prompt_tokens / prefill_tokens_per_second`
            - **Generation**: `output_tokens / decode_tokens_per_second`
            
            Prefill is memory-bandwidth bound and can process many tokens in parallel.
            Generation is compute-bound and processes tokens sequentially.
            """)
        
        with st.expander("**VRAM Requirements**"):
            st.markdown("""
            VRAM = Weights + KV Cache + Overhead
            
            - **Weights**: `parameters × bits_per_weight / 8`
            - **KV Cache**: `tokens × layers × 2 × d_model / kv_groups × bytes_per_value`
            - **Overhead**: Activations, optimizer states, temporary buffers (~20% of weights)
            
            Quantization (4-bit, 8-bit) reduces weight memory linearly.
            FP8 KV cache cuts cache memory in half vs FP16.
            """)
        
        with st.expander("**MoE vs Dense Models**"):
            st.markdown("""
            **Mixture of Experts (MoE)**:
            - Total params != active params (only some experts activate per token)
            - Use active params for latency/VRAM calculations
            - Better quality/compute ratio but more complex to fine-tune
            
            **Dense Models**:
            - All parameters activate for every token
            - Simpler to fine-tune and deploy
            - More predictable performance characteristics
            """)
        
        with st.expander("**When to Fine-tune**"):
            st.markdown("""
            **Good candidates for LoRA/QLoRA**:
            - Gap to teacher <10% on your metric
            - Structured outputs (JSON, SQL, code)
            - 10k+ high-quality examples
            - Meeting latency requirements
            
            **Better to wait**:
            - Large quality gap (>15%)
            - Limited training data (<5k examples)
            - Failing latency SLOs
            - Knowledge gaps (use RAG instead)
            """)

with tab3:
    st.markdown("### Learning Resources")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📚 Essential Reading")
        st.markdown("""
        - [Transformer Memory Requirements](https://github.com/vllm-project/vllm/blob/main/docs/source/design/kernel/paged_attention.rst)
        - [Quantization Explained](https://huggingface.co/blog/4bit-transformers-bitsandbytes)
        - [MoE Architecture Guide](https://huggingface.co/blog/moe)
        - [LoRA/QLoRA Paper](https://arxiv.org/abs/2106.09685)
        """)
        
        st.markdown("#### 🎯 Interactive Exercises")
        with st.expander("Exercise 1: Make it Fit"):
            st.markdown("""
            **Challenge**: Make Llama-3.1-70B fit on a single RTX 4090 (24GB)
            
            **Hints**:
            - Check quantization options
            - Consider KV cache optimization
            - Look at context length limits
            
            **Solution**: Use 4-bit quantization (~35GB → ~10GB), FP8 KV cache,
            limit context to 4k tokens.
            """)
    
    with col2:
        st.markdown("#### 🛠️ Serving Stack Guide")
        st.markdown("""
        **vLLM** (Recommended for most cases):
        - PagedAttention for efficient KV cache
        - Continuous batching
        - Tensor parallelism support
        
        **TGI** (HuggingFace):
        - Great integration with HF ecosystem
        - Built-in quantization support
        - Good for standard deployments
        
        **TensorRT-LLM** (NVIDIA):
        - Maximum performance on NVIDIA GPUs
        - Complex but powerful optimization
        - Best for production at scale
        """)
        
        st.markdown("#### 🎓 Team Training")
        st.info("""
        **Workshop Ideas**:
        1. Model selection race - who can find the best model for a use case?
        2. VRAM optimization challenge - fit the biggest model possible
        3. Latency hunt - achieve target latency with maximum quality
        """)

with tab4:
    st.markdown("### About Agent Foundry")
    
    st.markdown("""
    Agent Foundry is an interactive tool designed to help teams understand and select open-weight models
    for their specific use cases. It combines practical physics calculations with educational resources
    to build intuition about model deployment.
    
    #### Key Features:
    - 🔍 **Model Atlas**: Browse and compare open-weight models
    - 🧮 **Physics Lab**: Calculate VRAM, latency, and deployment requirements
    - 🎯 **Smart Selection**: Get recommendations based on your constraints
    - 📚 **Learning Mode**: Build intuition through interactive examples
    - 💾 **Export Configs**: Generate deployment-ready YAML/JSON
    
    #### Coming Soon:
    - More models (Gemma, Phi, Mistral variants)
    - Cost calculator for cloud deployments
    - Fine-tuning dataset size estimator
    - Multi-GPU deployment planner
    - A/B testing configuration generator
    
    #### Contributing:
    To add new models, edit `data/models_registry.json` and submit a PR!
    """)
    
    st.markdown("---")
    st.markdown("*Built with ❤️ for teams deploying open-weight models in production*")