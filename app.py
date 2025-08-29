# app.py — Agent Foundry (MVP)
# Run: pip install streamlit pyyaml pandas
import math, io, json, yaml
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Agent Foundry", layout="wide")

# ---------- Seed registry (replace with external JSON file later) ----------
MODELS = [
    {
        "name": "Qwen2.5-72B-Instruct",
        "family": "Qwen2.5",
        "size_b": 72, "active_b": 72, "context_k": 128,
        "specialization": ["general","planning","tool-use"],
        "license": "Open-Weight", "arch": "dense",
        "notes": "Long context; strong structured output",
        "defaults": {"layers": 80, "d_model": 8192, "kv_groups": 8}
    },
    {
        "name": "Qwen2.5-Coder-32B",
        "family": "Qwen2.5-Coder",
        "size_b": 32, "active_b": 32, "context_k": 128,
        "specialization": ["code","sql"],
        "license": "Open-Weight", "arch": "dense",
        "notes": "Repo-level, long-context code tasks",
        "defaults": {"layers": 48, "d_model": 6656, "kv_groups": 8}
    },
    {
        "name": "DBRX Instruct",
        "family": "DBRX",
        "size_b": 132, "active_b": 36, "context_k": 32,
        "specialization": ["general","code","math"],
        "license": "Permissive", "arch": "moe",
        "notes": "MoE; good throughput per quality",
        "defaults": {"layers": 64, "d_model": 7168, "kv_groups": 8}
    },
    {
        "name": "Llama-3.1-70B-Instruct",
        "family": "Llama",
        "size_b": 70, "active_b": 70, "context_k": 64,
        "specialization": ["general","rag"],
        "license": "Community", "arch": "dense",
        "notes": "Strong generalist; check license terms",
        "defaults": {"layers": 80, "d_model": 8192, "kv_groups": 8}
    },
    {
        "name": "DeepSeek-Coder-V2-Lite-16B",
        "family": "DeepSeek-Coder-V2",
        "size_b": 16, "active_b": 16, "context_k": 128,
        "specialization": ["code","sql","math"],
        "license": "Permissive", "arch": "dense",
        "notes": "Code specialist; long context",
        "defaults": {"layers": 40, "d_model": 5120, "kv_groups": 8}
    },
    {
        "name": "Gemma2-27B",
        "family": "Gemma2",
        "size_b": 27, "active_b": 27, "context_k": 8,
        "specialization": ["general","rag"],
        "license": "Open-Weight", "arch": "dense",
        "notes": "Efficient dense model",
        "defaults": {"layers": 48, "d_model": 5632, "kv_groups": 8}
    },
    {
        "name": "Mixtral-8x22B",
        "family": "Mixtral",
        "size_b": 176, "active_b": 44, "context_k": 32,
        "specialization": ["general","planning"],
        "license": "Apache-2.0", "arch": "moe",
        "notes": "MoE: high quality, mid-latency",
        "defaults": {"layers": 64, "d_model": 6144, "kv_groups": 8}
    },
    {
        "name": "StarCoder2-15B",
        "family": "StarCoder2",
        "size_b": 15, "active_b": 15, "context_k": 16,
        "specialization": ["code"],
        "license": "OpenRAIL-M", "arch": "dense",
        "notes": "Great FIM/insertion for IDEs",
        "defaults": {"layers": 40, "d_model": 5120, "kv_groups": 8}
    }
]

GPUS = [
    {"name": "A100 80GB", "vram_gib": 80},
    {"name": "H100 80GB", "vram_gib": 80},
    {"name": "L40S 48GB", "vram_gib": 48},
    {"name": "RTX 4090 24GB", "vram_gib": 24},
    {"name": "T4 16GB", "vram_gib": 16}
]

# ---------- Helpers ----------
def weights_vram_gib(active_b, bits, overhead=1.2):
    bytes_per_weight = bits / 8.0
    return active_b * 1e9 * bytes_per_weight * overhead / (1024**3)

def kv_cache_gib(tokens_total, layers, d_model, kv_groups, bytes_kv):
    # Simple transformer KV estimate; assumes K and V caches per layer, grouped-query attention
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
    # partial match
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
        kv_bytes = 2 if req["kv_dtype"] == "fp16" else 1  # fp8=1, fp16=2 bytes
        L = m["defaults"]["layers"]; D = m["defaults"]["d_model"]; G = m["defaults"]["kv_groups"]
        tokens_total = req["prompt_tokens"] + req["output_tokens"]
        w_vram = weights_vram_gib(m["active_b"], bits)
        kv_vram = kv_cache_gib(tokens_total, L, D, G, kv_bytes)
        total_vram = w_vram + kv_vram + req["vram_overhead_gib"]

        meets_vram = total_vram <= hw["vram_gib"]
        meets_ctx = m["context_k"] >= math.ceil(tokens_total/1000)
        meets_lic = license_ok(req["license_policy"], m["license"])
        # latency
        lat_s = est_latency_s(req["prompt_tokens"], req["output_tokens"], perf["prefill_tps"], perf["decode_tps"])
        meets_slo = lat_s*1000 <= req["latency_ms"]

        # scores
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
    # Simple rubric
    if lat_ms > slo_ms:  # failing latency => prefer smaller base or quantize
        return "Prefer smaller/faster base; revisit tuning after hitting SLO."
    if gap_pct <= 10 and structured and data_k >= 10:
        return "Proceed with LoRA/QLoRA SFT; consider light DPO for preference."
    if gap_pct <= 15 and structured and data_k >= 5:
        return "Pilot LoRA with small dataset; add retrieval or teacher distillation."
    return "Hold off on tuning; expand data, add RAG, or use larger base temporarily."

# ---------- UI ----------
st.title("🛠️ Agent Foundry — Model Selection & Physics Lab")

# Sidebar: requirements
st.sidebar.header("Requirements")
task = st.sidebar.selectbox("Primary task", ["general","rag","code","sql","vision","planning"])
latency_ms = st.sidebar.slider("Target p95 latency (ms)", 100, 8000, 1500, 50)
prompt_tokens = st.sidebar.number_input("Prompt tokens", 16, 200000, 2048, 16)
output_tokens = st.sidebar.number_input("Output tokens", 16, 200000, 512, 16)
license_policy = st.sidebar.selectbox("License policy", ["Allow open-weight", "Permissive-only", "Anything (internal PoC)"])
weights_bits = st.sidebar.selectbox("Weight quantization", [4,8,16], index=0)
kv_dtype = st.sidebar.selectbox("KV dtype", ["fp16","fp8"], index=1)
vram_overhead_gib = st.sidebar.slider("Extra VRAM overhead (GiB)", 0.0, 8.0, 2.0, 0.5)

gpu = st.sidebar.selectbox("Target GPU", [g["name"] for g in GPUS], index=0)
gpu_info = next(g for g in GPUS if g["name"] == gpu)

st.sidebar.header("Throughput (measure or estimate)")
prefill_tps = st.sidebar.number_input("Prefill tokens/sec", 100.0, 20000.0, 3000.0, 50.0)
decode_tps  = st.sidebar.number_input("Decode tokens/sec", 20.0, 5000.0, 200.0, 5.0)

if st.sidebar.button("Rank models"):
    req = {
        "task": task, "latency_ms": latency_ms,
        "prompt_tokens": int(prompt_tokens), "output_tokens": int(output_tokens),
        "license_policy": license_policy, "weights_bits": int(weights_bits),
        "kv_dtype": kv_dtype, "vram_overhead_gib": float(vram_overhead_gib)
    }
    perf = {"prefill_tps": float(prefill_tps), "decode_tps": float(decode_tps)}
    df = rank_models(MODELS, req, gpu_info, perf)
    st.subheader("Recommended models")
    st.dataframe(df, use_container_width=True)
    top = df.iloc[0].to_dict()

    st.markdown("### Suggested plan")
    st.markdown(f"- **Pick:** `{top['Model']}` ({top['Arch']}, active {top['Active B']}B)\n"
                f"- **Why:** Task fit {round(top['Task Fit']*100)}%, latency ≈ {top['Est. Latency (ms)']} ms, "
                f"VRAM total ≈ {top['Total VRAM (GiB)']} GiB on {gpu_info['name']}.\n"
                f"- **Caveats:** {top['Notes']}")

    # Tuning triage
    st.markdown("### Tuning triage")
    gap = st.slider("Observed gap to large teacher on your metric (%)", 0, 40, 8, 1)
    structured = st.checkbox("Outputs are structured (JSON/SQL/code/templates)?", value=True)
    data_k = st.slider("High-quality task examples available (thousands)", 0, 200, 25, 1)
    rec = tuning_recommendation(gap, structured, data_k, latency_ms, top['Est. Latency (ms)'])
    st.info(rec)

    # Export solution.yaml
    solution = {
        "use_case": task,
        "hardware": {"gpu": gpu_info["name"], "vram_gib": gpu_info["vram_gib"]},
        "latency_slo_ms": latency_ms,
        "traffic_assumptions": {"prefill_tps": prefill_tps, "decode_tps": decode_tps,
                                "prompt_tokens": prompt_tokens, "output_tokens": output_tokens},
        "model": {"name": top["Model"], "quant_bits": int(weights_bits), "kv_dtype": kv_dtype},
        "serving": {"runtime": "vLLM", "features": ["continuous-batching","paged-attention","speculative-decoding (optional)"]},
        "rag": {"enabled": task in ["rag","general"], "notes": "Prefer retrieval over very long contexts."},
        "tuning": {"recommended": rec, "method": "LoRA/QLoRA if recommended", "checks": ["JSON schema", "unit tests (code/sql)"]}
    }
    yml = yaml.dump(solution, sort_keys=False)
    st.download_button("Download solution.yaml", data=yml.encode(), file_name="solution.yaml", mime="text/yaml")

# Tabs: Atlas & Physics explanations
tab1, tab2, tab3 = st.tabs(["Model Atlas","Physics Lab (Explainers)","About & How-To"])

with tab1:
    st.write("Filter and inspect models. (Edit the registry in code or switch to external JSON.)")
    q = st.text_input("Search", "")
    show = [m for m in MODELS if q.lower() in m["name"].lower()]
    df = pd.DataFrame([{
        "Model": m["name"], "Family": m["family"], "Arch": m["arch"], "Active B": m["active_b"],
        "Context (k)": m["context_k"], "Specialization": ", ".join(m["specialization"]),
        "License": m["license"], "Notes": m["notes"]
    } for m in show])
    st.dataframe(df, use_container_width=True)

with tab2:
    st.markdown("#### Latency")
    st.markdown("**Total ≈ (Prompt tokens / prefill TPS) + (Output tokens / decode TPS)**. "
                "Prefill is memory‑bandwidth bound; decode is serialized but can be sped up with speculative decoding.")
    st.markdown("#### VRAM")
    st.markdown("- **Weights**: `params × bits/8` (+ overhead). Quantize to 4 or 8‑bit to fit bigger models.\n"
                "- **KV cache**: grows with `(input+output tokens) × layers × (2 × d_model / kv_groups)`; "
                "use FP8 KV and keep contexts short or use RAG.")
    st.markdown("#### MoE vs Dense")
    st.markdown("MoE activates fewer parameters per token (use **active parameters** for latency/VRAM estimates); "
                "dense is simpler to fine‑tune.")

with tab3:
    st.markdown("**How to use this app**")
    st.markdown("1. Set your task, SLOs, tokens, and GPU.\n"
                "2. Click **Rank models** to get a shortlist and a YAML you can drop into infra.\n"
                "3. Use **Tuning triage** to decide on LoRA/QLoRA.\n"
                "4. Update the in‑code registry or point to an external JSON for fresh models.")