import streamlit as st
import numpy as np

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NeuroLab · Neural Network Toolbox",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@200;300;400;600;700;900&family=Space+Mono:ital,wght@0,400;0,700;1,400&family=JetBrains+Mono:wght@300;400;700&display=swap');

:root {
  --bg:#020408;--bg2:#060d14;--bg3:#0a1520;--panel:#0d1e2e;
  --border:#1a3550;--accent:#00e5ff;--accent2:#7c3aed;
  --accent3:#10b981;--accent4:#f59e0b;--danger:#ef4444;
  --text:#e2f0ff;--muted:#5a7a9a;
  --glow:0 0 20px rgba(0,229,255,.25);
}

*,*::before,*::after{box-sizing:border-box;margin:0}
html,body,[data-testid="stAppViewContainer"]{
  background:var(--bg)!important;color:var(--text);
  font-family:'Outfit',sans-serif;
}
[data-testid="stAppViewContainer"]{
  background:
    radial-gradient(ellipse 80% 50% at 20% 10%,rgba(0,229,255,.04) 0%,transparent 60%),
    radial-gradient(ellipse 60% 40% at 80% 80%,rgba(124,58,237,.05) 0%,transparent 60%),
    var(--bg)!important;
}
[data-testid="stAppViewContainer"]::before{
  content:'';position:fixed;inset:0;pointer-events:none;z-index:0;
  background-image:linear-gradient(rgba(0,229,255,.018) 1px,transparent 1px),
                   linear-gradient(90deg,rgba(0,229,255,.018) 1px,transparent 1px);
  background-size:48px 48px;
}
[data-testid="stHeader"]{background:transparent!important}
[data-testid="block-container"]{padding-top:.8rem!important;position:relative;z-index:1}

/* SIDEBAR */
[data-testid="stSidebar"]{
  background:linear-gradient(180deg,#05101a 0%,#020810 100%)!important;
  border-right:1px solid var(--border);
  box-shadow:4px 0 40px rgba(0,229,255,.06);
}
[data-testid="stSidebar"]>div{padding:1.5rem 1rem}

.sb-logo{text-align:center;padding:1.2rem 0 1.8rem;border-bottom:1px solid var(--border);margin-bottom:1.4rem}
.sb-logo .mark{
  width:58px;height:58px;border-radius:16px;
  background:linear-gradient(135deg,var(--accent),var(--accent2));
  margin:0 auto .9rem;display:flex;align-items:center;justify-content:center;
  font-size:26px;box-shadow:0 0 30px rgba(0,229,255,.35);
  animation:pglow 3s ease-in-out infinite;
}
@keyframes pglow{0%,100%{box-shadow:0 0 20px rgba(0,229,255,.3)}50%{box-shadow:0 0 44px rgba(0,229,255,.65),0 0 60px rgba(124,58,237,.3)}}
.sb-logo h2{font-family:'Outfit',sans-serif;font-weight:700;font-size:1.05rem;color:var(--text);letter-spacing:1px}
.sb-logo p{font-family:'Space Mono',monospace;font-size:.62rem;color:var(--muted);letter-spacing:2px;text-transform:uppercase}
.nav-label{font-family:'Space Mono',monospace;font-size:.58rem;letter-spacing:3px;text-transform:uppercase;color:var(--muted);padding:.4rem .4rem .2rem;margin-top:1rem}

/* radio → nav */
[data-testid="stRadio"]>div{gap:3px!important}
[data-testid="stRadio"]>div>label{
  background:transparent;border:1px solid transparent;border-radius:10px;
  padding:.55rem .75rem!important;cursor:pointer;transition:all .2s;
  font-family:'Outfit',sans-serif;font-size:.85rem;color:var(--muted)!important;
  display:flex;align-items:center;gap:8px;
}
[data-testid="stRadio"]>div>label:hover{background:rgba(0,229,255,.05);border-color:rgba(0,229,255,.2);color:var(--text)!important}
[data-testid="stRadio"]>div>label:has(input:checked){
  background:linear-gradient(90deg,rgba(0,229,255,.12),rgba(124,58,237,.08));
  border-color:var(--accent)!important;color:var(--accent)!important;box-shadow:var(--glow);
}

/* HERO */
.hero{text-align:center;padding:3rem 2rem 2rem;position:relative;overflow:hidden}
.hero::before{content:'';position:absolute;inset:0;background:radial-gradient(ellipse 70% 80% at 50% 0%,rgba(0,229,255,.07) 0%,transparent 70%);pointer-events:none}
.hero-eye{font-family:'Space Mono',monospace;font-size:.68rem;letter-spacing:4px;color:var(--accent);text-transform:uppercase;margin-bottom:.9rem;animation:fiu .6s ease both}
.hero h1{font-family:'Outfit',sans-serif;font-weight:900;font-size:clamp(2.2rem,6vw,4.2rem);line-height:1.05;
  background:linear-gradient(135deg,#fff 0%,var(--accent) 50%,var(--accent2) 100%);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
  margin-bottom:1rem;animation:fiu .6s .15s ease both}
.hero p{font-size:1rem;color:var(--muted);max-width:560px;margin:0 auto 1.8rem;line-height:1.7;animation:fiu .6s .3s ease both}
.hero-badges{display:flex;flex-wrap:wrap;gap:9px;justify-content:center;animation:fiu .6s .45s ease both}
@keyframes fiu{from{opacity:0;transform:translateY(20px)}to{opacity:1;transform:translateY(0)}}

.badge{display:inline-flex;align-items:center;gap:5px;padding:5px 13px;border-radius:100px;font-family:'Space Mono',monospace;font-size:.68rem;letter-spacing:1px;border:1px solid}
.bc{color:var(--accent);border-color:rgba(0,229,255,.3);background:rgba(0,229,255,.06)}
.bp{color:#a78bfa;border-color:rgba(124,58,237,.3);background:rgba(124,58,237,.06)}
.bg{color:var(--accent3);border-color:rgba(16,185,129,.3);background:rgba(16,185,129,.06)}
.ba{color:var(--accent4);border-color:rgba(245,158,11,.3);background:rgba(245,158,11,.06)}

/* MODULE CARDS */
.mgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(270px,1fr));gap:18px;padding:.8rem 0 1.8rem}
.mcard{
  background:var(--panel);border:1px solid var(--border);border-radius:20px;
  padding:1.8rem;cursor:pointer;transition:all .3s cubic-bezier(.34,1.56,.64,1);
  position:relative;overflow:hidden;animation:cin .5s ease both;
}
.mcard::before{content:'';position:absolute;inset:0;opacity:0;transition:opacity .3s;border-radius:inherit}
.mcard:hover{transform:translateY(-6px) scale(1.02)}
.mcard.c::before{background:radial-gradient(circle at 50% 0%,rgba(0,229,255,.08),transparent 70%)}
.mcard.p::before{background:radial-gradient(circle at 50% 0%,rgba(124,58,237,.1),transparent 70%)}
.mcard.g::before{background:radial-gradient(circle at 50% 0%,rgba(16,185,129,.08),transparent 70%)}
.mcard.a::before{background:radial-gradient(circle at 50% 0%,rgba(245,158,11,.08),transparent 70%)}
.mcard.r::before{background:radial-gradient(circle at 50% 0%,rgba(239,68,68,.08),transparent 70%)}
.mcard:hover::before{opacity:1}
.mcard.c:hover{border-color:rgba(0,229,255,.5);box-shadow:0 20px 60px rgba(0,229,255,.12)}
.mcard.p:hover{border-color:rgba(124,58,237,.5);box-shadow:0 20px 60px rgba(124,58,237,.15)}
.mcard.g:hover{border-color:rgba(16,185,129,.5);box-shadow:0 20px 60px rgba(16,185,129,.12)}
.mcard.a:hover{border-color:rgba(245,158,11,.5);box-shadow:0 20px 60px rgba(245,158,11,.12)}
.mcard.r:hover{border-color:rgba(239,68,68,.5);box-shadow:0 20px 60px rgba(239,68,68,.12)}
@keyframes cin{from{opacity:0;transform:translateY(28px)}to{opacity:1;transform:translateY(0)}}
.mcard:nth-child(2){animation-delay:.1s}.mcard:nth-child(3){animation-delay:.2s}
.mcard:nth-child(4){animation-delay:.3s}.mcard:nth-child(5){animation-delay:.4s}
.ci{width:52px;height:52px;border-radius:13px;display:flex;align-items:center;justify-content:center;font-size:24px;margin-bottom:1.1rem;position:relative;z-index:1}
.ci.c{background:linear-gradient(135deg,rgba(0,229,255,.2),rgba(0,229,255,.05));border:1px solid rgba(0,229,255,.3)}
.ci.p{background:linear-gradient(135deg,rgba(124,58,237,.2),rgba(124,58,237,.05));border:1px solid rgba(124,58,237,.3)}
.ci.g{background:linear-gradient(135deg,rgba(16,185,129,.2),rgba(16,185,129,.05));border:1px solid rgba(16,185,129,.3)}
.ci.a{background:linear-gradient(135deg,rgba(245,158,11,.2),rgba(245,158,11,.05));border:1px solid rgba(245,158,11,.3)}
.ci.r{background:linear-gradient(135deg,rgba(239,68,68,.2),rgba(239,68,68,.05));border:1px solid rgba(239,68,68,.3)}
.mcard h3{font-family:'Outfit',sans-serif;font-weight:700;font-size:1.05rem;margin-bottom:.45rem;position:relative;z-index:1}
.mcard p{font-size:.83rem;color:var(--muted);line-height:1.6;position:relative;z-index:1}
.ctag{display:inline-block;margin-top:.9rem;font-family:'Space Mono',monospace;font-size:.63rem;letter-spacing:1.5px;text-transform:uppercase;padding:3px 10px;border-radius:100px;position:relative;z-index:1}
.ctag.c{color:var(--accent);background:rgba(0,229,255,.08);border:1px solid rgba(0,229,255,.2)}
.ctag.p{color:#a78bfa;background:rgba(124,58,237,.08);border:1px solid rgba(124,58,237,.2)}
.ctag.g{color:var(--accent3);background:rgba(16,185,129,.08);border:1px solid rgba(16,185,129,.2)}
.ctag.a{color:var(--accent4);background:rgba(245,158,11,.08);border:1px solid rgba(245,158,11,.2)}
.ctag.r{color:#f87171;background:rgba(239,68,68,.08);border:1px solid rgba(239,68,68,.2)}

/* SECTION HEADER */
.shead{display:flex;align-items:center;gap:14px;margin-bottom:1.8rem;padding-bottom:.9rem;border-bottom:1px solid var(--border)}
.sicon{width:44px;height:44px;border-radius:12px;display:flex;align-items:center;justify-content:center;font-size:20px;flex-shrink:0}
.shead h2{font-family:'Outfit',sans-serif;font-weight:700;font-size:1.45rem}
.shead p{font-size:.83rem;color:var(--muted);margin-top:2px}

/* PANELS */
.panel{background:var(--panel);border:1px solid var(--border);border-radius:16px;padding:1.4rem;margin-bottom:1.1rem}
.ptitle{font-family:'Space Mono',monospace;font-size:.7rem;letter-spacing:2px;text-transform:uppercase;color:var(--muted);margin-bottom:.9rem;display:flex;align-items:center;gap:8px}
.ptitle::before{content:'';width:3px;height:13px;border-radius:2px;background:var(--accent)}

/* METRICS */
.mrow{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:11px;margin-bottom:1.1rem}
.mc{background:rgba(0,229,255,.04);border:1px solid rgba(0,229,255,.15);border-radius:12px;padding:.9rem;text-align:center;transition:all .2s}
.mc:hover{border-color:var(--accent);box-shadow:var(--glow)}
.mv{font-family:'JetBrains Mono',monospace;font-size:1.5rem;font-weight:700;color:var(--accent)}
.ml{font-size:.73rem;color:var(--muted);margin-top:2px}

/* TRUTH TABLE */
.tt{width:100%;border-collapse:collapse;font-family:'JetBrains Mono',monospace;font-size:.84rem}
.tt th{background:rgba(0,229,255,.1);color:var(--accent);padding:9px 15px;text-align:center;font-size:.72rem;letter-spacing:1px;border:1px solid var(--border)}
.tt td{padding:8px 15px;text-align:center;border:1px solid var(--border);color:var(--text);transition:background .2s}
.tt tr:hover td{background:rgba(0,229,255,.04)}
.t1{color:var(--accent3)}.t0{color:var(--danger)}
.ok{background:rgba(16,185,129,.1)}.bad{background:rgba(239,68,68,.1)}

/* NN DIAGRAM */
.nnd{text-align:center;padding:1rem 0}
.nnlayer{display:inline-flex;flex-direction:column;align-items:center;gap:11px;margin:0 1.8rem;vertical-align:middle}
.nnn{width:40px;height:40px;border-radius:50%;border:2px solid;display:flex;align-items:center;justify-content:center;font-family:'JetBrains Mono',monospace;font-size:.62rem;transition:all .3s}
.nnn:hover{transform:scale(1.2)}
.nnn.i{border-color:var(--accent);color:var(--accent);background:rgba(0,229,255,.08)}
.nnn.h{border-color:var(--accent2);color:#a78bfa;background:rgba(124,58,237,.08)}
.nnn.o{border-color:var(--accent3);color:var(--accent3);background:rgba(16,185,129,.08)}
.nnlbl{font-family:'Space Mono',monospace;font-size:.6rem;letter-spacing:1px;color:var(--muted);text-transform:uppercase}

/* FORMULA */
.formula{background:rgba(0,0,0,.4);border:1px solid var(--border);border-left:3px solid var(--accent2);border-radius:10px;padding:.9rem 1.1rem;font-family:'JetBrains Mono',monospace;font-size:.82rem;color:#a78bfa;margin:.7rem 0;overflow-x:auto}

/* CHIP */
.chip{display:inline-flex;align-items:center;gap:5px;padding:3px 10px;border-radius:100px;font-family:'JetBrains Mono',monospace;font-size:.7rem}
.cc{color:var(--accent);background:rgba(0,229,255,.1);border:1px solid rgba(0,229,255,.25)}
.cg{color:var(--accent3);background:rgba(16,185,129,.1);border:1px solid rgba(16,185,129,.25)}
.ca{color:var(--accent4);background:rgba(245,158,11,.1);border:1px solid rgba(245,158,11,.25)}
.cr{color:#f87171;background:rgba(239,68,68,.1);border:1px solid rgba(239,68,68,.25)}

/* SCANLINE */
.scanline{position:fixed;top:0;left:0;right:0;height:2px;z-index:9999;pointer-events:none;background:linear-gradient(90deg,transparent,var(--accent),transparent);animation:scan 5s linear infinite;opacity:.3}
@keyframes scan{0%{top:0%}100%{top:100%}}

/* STATUS */
.sdot{display:inline-block;width:7px;height:7px;border-radius:50%;background:var(--accent3);box-shadow:0 0 7px var(--accent3);animation:blink 2s ease-in-out infinite;margin-right:5px}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.3}}

/* FOOTER */
.footer{text-align:center;padding:1.8rem 0 .8rem;border-top:1px solid var(--border);margin-top:2.5rem;font-family:'Space Mono',monospace;font-size:.62rem;color:var(--muted);letter-spacing:2px}

/* STREAMLIT OVERRIDES */
[data-testid="stSlider"]>div>div>div{background:var(--border)!important}
[data-testid="stSlider"]>div>div>div>div{background:linear-gradient(90deg,var(--accent),var(--accent2))!important}
[data-testid="stSelectbox"]>div>div{background:var(--panel)!important;border:1px solid var(--border)!important;border-radius:10px!important;color:var(--text)!important}
[data-testid="stButton"]>button{background:linear-gradient(135deg,var(--accent),var(--accent2))!important;color:#000!important;border:none!important;border-radius:10px!important;font-family:'Outfit',sans-serif!important;font-weight:700!important;font-size:.88rem!important;padding:.52rem 1.4rem!important;transition:all .2s!important}
[data-testid="stButton"]>button:hover{transform:translateY(-2px)!important;box-shadow:0 8px 28px rgba(0,229,255,.4)!important}
[data-testid="stNumberInput"]>div>div>input,[data-testid="stTextInput"]>div>div>input{background:var(--panel)!important;border:1px solid var(--border)!important;border-radius:8px!important;color:var(--text)!important}
[data-testid="stTextArea"]>div>textarea{background:var(--panel)!important;border:1px solid var(--border)!important;border-radius:8px!important;color:var(--text)!important;font-family:'JetBrains Mono',monospace!important;font-size:.83rem!important}
[data-testid="stPlotlyChart"]{border:1px solid var(--border);border-radius:14px;overflow:hidden;background:var(--panel)!important}
[data-testid="stExpander"]{background:var(--panel)!important;border:1px solid var(--border)!important;border-radius:12px!important}
hr{border-color:var(--border)!important}
label,.stSelectbox label,.stSlider label{color:var(--muted)!important;font-family:'Outfit',sans-serif!important;font-size:.83rem!important}
::-webkit-scrollbar{width:4px;height:4px}
::-webkit-scrollbar-track{background:var(--bg2)}
::-webkit-scrollbar-thumb{background:var(--border);border-radius:10px}
::-webkit-scrollbar-thumb:hover{background:var(--accent)}
code,pre{font-family:'JetBrains Mono',monospace!important;font-size:.81rem!important}

/* TABS */
[data-testid="stTabs"] [data-testid="stTab"]{
  font-family:'Outfit',sans-serif!important;font-size:.88rem!important;
  color:var(--muted)!important;background:transparent!important;
  border:none!important;padding:.5rem 1.2rem!important;
}
[data-testid="stTabs"] [data-testid="stTab"][aria-selected="true"]{
  color:var(--accent)!important;
  border-bottom:2px solid var(--accent)!important;
}
</style>
<div class="scanline"></div>
""", unsafe_allow_html=True)


# ── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sb-logo">
      <div class="mark">⚡</div>
      <h2>NeuroLab</h2>
      <p>Neural Network Toolbox</p>
    </div>
    <div class="nav-label">🧩 Modules</div>
    """, unsafe_allow_html=True)

    page = st.radio("nav", [
        "🏠  Overview",
        "🧠  Perceptron & Logic Gates",
        "➡️  Forward Propagation",
        "⬅️  Backpropagation",
        "👁️  Vision Hub (YOLO + MP)",
        "📝  Sentiment RNN (LSTM)",
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown('<div class="nav-label">⚙️ Settings</div>', unsafe_allow_html=True)
    show_formulas = st.toggle("Show Math Formulas", value=True)
    st.markdown("""
    <div style="margin-top:1.5rem">
      <div style="font-family:'Space Mono',monospace;font-size:.6rem;color:var(--muted);letter-spacing:1px">
        <span class="sdot"></span>v1.0 · All systems go
      </div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if "Overview" in page:
    st.markdown("""
    <div class="hero">
      <div class="hero-eye">⚡ Interactive Machine Learning Laboratory</div>
      <h1>Neural Network<br>Lab Toolbox</h1>
      <p>Explore, train, and visualize neural networks from first principles —
         perceptrons to LSTMs, logic gates to computer vision.</p>
      <div class="hero-badges">
        <span class="badge bc">⚡ PyTorch</span>
        <span class="badge bp">🧠 Deep Learning</span>
        <span class="badge bg">👁️ Computer Vision</span>
        <span class="badge ba">📝 NLP / LSTM</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="mgrid">', unsafe_allow_html=True)
    mods = [
        ("c","🧠","Perceptron & Logic Gates","Train a single-layer perceptron on AND, OR, NAND, XOR. Watch weights converge in real time.","Classification"),
        ("p","➡️","Forward Propagation","Step through a multi-layer feedforward network. Inspect activations and outputs layer by layer.","Feed-Forward"),
        ("g","⬅️","Backpropagation","Visualize gradient flow and weight updates. Understand how errors propagate backward.","Gradient Descent"),
        ("a","👁️","Vision Hub","Real-time object detection with YOLOv8 and pose estimation with MediaPipe.","Computer Vision"),
        ("r","📝","Sentiment RNN","Train an LSTM on custom text data. Predict sentiment and visualize hidden state dynamics.","NLP · LSTM"),
    ]
    for col, icon, title, desc, tag in mods:
        st.markdown(f"""
        <div class="mcard {col}">
          <div class="ci {col}">{icon}</div>
          <h3>{title}</h3><p>{desc}</p>
          <span class="ctag {col}">{tag}</span>
        </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="mrow">
      <div class="mc"><div class="mv">5</div><div class="ml">Modules</div></div>
      <div class="mc"><div class="mv">∞</div><div class="ml">Experiments</div></div>
      <div class="mc"><div class="mv" style="font-size:1.1rem">Real-Time</div><div class="ml">Training Viz</div></div>
      <div class="mc"><div class="mv" style="font-size:1.1rem">GPU</div><div class="ml">PyTorch Backend</div></div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: PERCEPTRON
# ═══════════════════════════════════════════════════════════════════════════════
elif "Perceptron" in page:
    from modules.perceptron import Perceptron
    import plotly.graph_objects as go

    st.markdown("""
    <div class="shead">
      <div class="sicon c" style="background:rgba(0,229,255,.1);border:1px solid rgba(0,229,255,.3)">🧠</div>
      <div><h2 style="color:var(--accent)">Perceptron & Logic Gates</h2>
           <p>Single-layer binary classifier trained with the perceptron learning rule</p></div>
    </div>""", unsafe_allow_html=True)

    if show_formulas:
        st.markdown('<div class="formula">ŷ = step(w₁x₁ + w₂x₂ + b) &nbsp;|&nbsp; wᵢ ← wᵢ + η·(y−ŷ)·xᵢ &nbsp;|&nbsp; b ← b + η·(y−ŷ)</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1.6])
    with col1:
        st.markdown('<div class="panel"><div class="ptitle">Configuration</div>', unsafe_allow_html=True)
        gate   = st.selectbox("Logic Gate", ["AND","OR","NAND","NOR","XOR"])
        lr     = st.slider("Learning Rate (η)", 0.01, 1.0, 0.1, 0.01)
        epochs = st.slider("Epochs", 1, 200, 50)
        st.markdown('</div>', unsafe_allow_html=True)
        if gate == "XOR":
            st.warning("⚠️ XOR is **not linearly separable** — a single perceptron cannot learn it.")
        run = st.button("⚡ Train Perceptron")

    with col2:
        st.markdown("""
        <div class="panel"><div class="ptitle">Network Architecture</div>
        <div class="nnd">
          <div style="display:inline-flex;align-items:center;gap:18px">
            <div class="nnlayer">
              <div class="nnn i">x₁</div><div class="nnn i">x₂</div>
              <div class="nnlbl">Input</div>
            </div>
            <div style="color:var(--muted);font-size:1.4rem">→</div>
            <div class="nnlayer">
              <div class="nnn o">∑+σ</div>
              <div class="nnlbl">Output</div>
            </div>
          </div>
        </div></div>""", unsafe_allow_html=True)

    if run:
        p = Perceptron(lr=lr)
        losses, _ = p.train(gate=gate, epochs=epochs)

        cA, cB = st.columns(2)
        with cA:
            st.markdown('<div class="panel"><div class="ptitle">Training Loss</div>', unsafe_allow_html=True)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=list(range(len(losses))), y=losses,
                mode='lines', line=dict(color='#00e5ff', width=2.5),
                fill='tozeroy', fillcolor='rgba(0,229,255,.07)'))
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#5a7a9a', family='JetBrains Mono'),
                xaxis=dict(title='Epoch', gridcolor='#1a3550', color='#5a7a9a'),
                yaxis=dict(title='Loss',  gridcolor='#1a3550', color='#5a7a9a'),
                margin=dict(l=10,r=10,t=10,b=10), height=230)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with cB:
            st.markdown('<div class="panel"><div class="ptitle">Truth Table</div>', unsafe_allow_html=True)
            preds, correct = p.evaluate(gate)
            inputs  = [(0,0),(0,1),(1,0),(1,1)]
            targets = p.get_targets(gate)
            rows = ""
            for (a,b_), t, pred in zip(inputs, targets, preds):
                cls = "ok" if pred == t else "bad"
                icon = "✓" if pred == t else "✗"
                rows += f'<tr class="{cls}"><td>{a}</td><td>{b_}</td><td class="t{t}">{t}</td><td class="t{pred}">{pred} {icon}</td></tr>'
            st.markdown(f'<table class="tt"><tr><th>x₁</th><th>x₂</th><th>Target</th><th>Pred</th></tr>{rows}</table>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown(f"""
        <div class="mrow">
          <div class="mc"><div class="mv" style="color:var(--accent3)">{correct}/4</div><div class="ml">Correct</div></div>
          <div class="mc"><div class="mv">{p.w[0]:.3f}</div><div class="ml">Weight w₁</div></div>
          <div class="mc"><div class="mv">{p.w[1]:.3f}</div><div class="ml">Weight w₂</div></div>
          <div class="mc"><div class="mv">{p.b:.3f}</div><div class="ml">Bias b</div></div>
        </div>""", unsafe_allow_html=True)

        # Decision boundary
        st.markdown('<div class="panel"><div class="ptitle">Decision Boundary</div>', unsafe_allow_html=True)
        xr = np.linspace(-0.4, 1.4, 200)
        yb = -(p.w[0]*xr + p.b)/p.w[1] if abs(p.w[1]) > 1e-6 else np.zeros_like(xr)
        colors = ['#00e5ff' if t == 1 else '#ef4444' for t in targets]
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=xr, y=yb, mode='lines',
            line=dict(color='rgba(124,58,237,.8)', width=2, dash='dash'), name='Decision Boundary'))
        for (a,bv), t, c in zip(inputs, targets, colors):
            fig2.add_trace(go.Scatter(x=[a], y=[bv], mode='markers',
                marker=dict(color=c, size=14, line=dict(color='white', width=2)),
                name=f'({a},{bv})={t}'))
        fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#5a7a9a', family='JetBrains Mono'),
            xaxis=dict(range=[-.4,1.4], gridcolor='#1a3550', color='#5a7a9a', title='x₁'),
            yaxis=dict(range=[-.4,1.4], gridcolor='#1a3550', color='#5a7a9a', title='x₂'),
            margin=dict(l=10,r=10,t=10,b=10), height=280)
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: FORWARD PROP
# ═══════════════════════════════════════════════════════════════════════════════
elif "Forward" in page:
    from modules.forward_prop import ForwardPropNetwork
    import plotly.graph_objects as go

    st.markdown("""
    <div class="shead">
      <div class="sicon" style="background:rgba(124,58,237,.1);border:1px solid rgba(124,58,237,.3);font-size:20px">➡️</div>
      <div><h2 style="color:#a78bfa">Forward Propagation</h2>
           <p>Step through a fully-connected network and inspect every activation</p></div>
    </div>""", unsafe_allow_html=True)

    if show_formulas:
        st.markdown('<div class="formula">a⁽ˡ⁾ = σ(W⁽ˡ⁾ · a⁽ˡ⁻¹⁾ + b⁽ˡ⁾) &nbsp;|&nbsp; σ(z) ∈ {ReLU, Sigmoid, Tanh, Linear}</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown('<div class="panel"><div class="ptitle">Architecture</div>', unsafe_allow_html=True)
        n_in  = st.slider("Input neurons",   1, 8, 3)
        n_h1  = st.slider("Hidden layer 1",  1, 8, 4)
        n_h2  = st.slider("Hidden layer 2",  0, 8, 2)
        n_out = st.slider("Output neurons",  1, 4, 1)
        act   = st.selectbox("Activation", ["ReLU","Sigmoid","Tanh","Linear"])
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<div class="panel"><div class="ptitle">Input Values</div>', unsafe_allow_html=True)
        xv = [st.number_input(f"x{i+1}", value=round(np.random.uniform(0,1),2),
              min_value=-5.0, max_value=5.0, step=0.01, key=f"fp{i}") for i in range(n_in)]
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        net = ForwardPropNetwork(n_in, n_h1, n_h2, n_out, act)
        activations = net.forward(np.array(xv))
        lnames = ["Input"]
        if n_h1 > 0: lnames.append("Hidden 1")
        if n_h2 > 0: lnames.append("Hidden 2")
        lnames.append("Output")

        st.markdown('<div class="panel"><div class="ptitle">Layer Activations</div>', unsafe_allow_html=True)
        for lname, a in zip(lnames, activations):
            cL, cV = st.columns([1,3])
            with cL:
                st.markdown(f'<div style="font-family:Space Mono,monospace;font-size:.68rem;color:var(--muted);padding:6px 0">{lname}</div>', unsafe_allow_html=True)
            with cV:
                vals = a.flatten()
                fig = go.Figure(go.Bar(x=list(range(len(vals))), y=vals,
                    marker=dict(color=vals,
                        colorscale=[[0,'#1a3550'],[0.5,'#7c3aed'],[1,'#00e5ff']],
                        line=dict(width=0))))
                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=0,r=0,t=0,b=0), height=75,
                    xaxis=dict(showgrid=False, visible=False),
                    yaxis=dict(showgrid=False, color='#5a7a9a', tickfont=dict(size=8)),
                    showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        final = activations[-1].flatten()
        cells = "".join(f'<div class="mc"><div class="mv" style="color:var(--accent3);font-size:1.2rem">{v:.4f}</div><div class="ml">Output {i+1}</div></div>' for i,v in enumerate(final))
        st.markdown(f'<div class="mrow">{cells}</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: BACKPROP
# ═══════════════════════════════════════════════════════════════════════════════
elif "Backprop" in page:
    from modules.backward_prop import BackpropNetwork
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    st.markdown("""
    <div class="shead">
      <div class="sicon" style="background:rgba(16,185,129,.1);border:1px solid rgba(16,185,129,.3);font-size:20px">⬅️</div>
      <div><h2 style="color:var(--accent3)">Backpropagation</h2>
           <p>Train a multi-layer network and visualize gradient flow + decision boundaries</p></div>
    </div>""", unsafe_allow_html=True)

    if show_formulas:
        st.markdown('<div class="formula">δ⁽ˡ⁾ = (W⁽ˡ⁺¹⁾ᵀδ⁽ˡ⁺¹⁾)⊙σ\'(z⁽ˡ⁾) &nbsp;|&nbsp; ∂L/∂W⁽ˡ⁾ = δ⁽ˡ⁾(a⁽ˡ⁻¹⁾)ᵀ &nbsp;|&nbsp; L = ½‖y−ŷ‖²</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown('<div class="panel"><div class="ptitle">Hyperparameters</div>', unsafe_allow_html=True)
        problem  = st.selectbox("Problem", ["XOR","Circle","Spiral","Moons"])
        lr_bp    = st.slider("Learning Rate", 0.001, 1.0, 0.05, 0.001, format="%.3f")
        ep_bp    = st.slider("Epochs", 100, 5000, 1000, 100)
        h1_bp    = st.slider("Hidden Layer 1", 2, 16, 8)
        h2_bp    = st.slider("Hidden Layer 2", 0, 16, 4)
        st.markdown('</div>', unsafe_allow_html=True)
        train_bp = st.button("⚡ Train Network")

    with col2:
        if train_bp:
            net_bp = BackpropNetwork(lr=lr_bp, h1=h1_bp, h2=h2_bp)
            losses_bp, grad_norms = net_bp.train(problem=problem, epochs=ep_bp)

            fig = make_subplots(rows=1, cols=2, subplot_titles=["Training Loss","Gradient Norm"])
            fig.add_trace(go.Scatter(y=losses_bp[::max(1,len(losses_bp)//100)],
                mode='lines', line=dict(color='#10b981', width=2),
                fill='tozeroy', fillcolor='rgba(16,185,129,.07)'), row=1, col=1)
            fig.add_trace(go.Scatter(y=grad_norms[::max(1,len(grad_norms)//100)],
                mode='lines', line=dict(color='#f59e0b', width=2),
                fill='tozeroy', fillcolor='rgba(245,158,11,.07)'), row=1, col=2)
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#5a7a9a', family='JetBrains Mono', size=10),
                height=260, showlegend=False, margin=dict(l=10,r=10,t=30,b=10))
            fig.update_xaxes(gridcolor='#1a3550', color='#5a7a9a')
            fig.update_yaxes(gridcolor='#1a3550', color='#5a7a9a')
            st.plotly_chart(fig, use_container_width=True)

            st.markdown('<div class="panel"><div class="ptitle">Decision Boundary</div>', unsafe_allow_html=True)
            db_fig = net_bp.plot_decision_boundary(problem)
            st.plotly_chart(db_fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            acc = net_bp.accuracy(problem)
            st.markdown(f"""
            <div class="mrow">
              <div class="mc"><div class="mv" style="color:var(--accent3)">{acc:.1f}%</div><div class="ml">Accuracy</div></div>
              <div class="mc"><div class="mv">{losses_bp[-1]:.4f}</div><div class="ml">Final Loss</div></div>
              <div class="mc"><div class="mv">{ep_bp}</div><div class="ml">Epochs</div></div>
              <div class="mc"><div class="mv">{lr_bp}</div><div class="ml">Learn Rate</div></div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="text-align:center;padding:4rem 2rem;color:var(--muted)">
              <div style="font-size:3rem;margin-bottom:1rem">⬅️</div>
              <div style="font-family:'Space Mono',monospace;font-size:.78rem;letter-spacing:2px">Configure & click Train to begin</div>
            </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: VISION HUB
# ═══════════════════════════════════════════════════════════════════════════════
elif "Vision" in page:
    from modules.opencv_hub import VisionHub

    st.markdown("""
    <div class="shead">
      <div class="sicon" style="background:rgba(245,158,11,.1);border:1px solid rgba(245,158,11,.3);font-size:20px">👁️</div>
      <div><h2 style="color:var(--accent4)">OpenCV Vision Hub</h2>
           <p>YOLOv8 object detection · MediaPipe pose & face landmarks</p></div>
    </div>""", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🎯 Object Detection", "🦴 Pose Estimation", "😊 Face Mesh"])

    with tab1:
        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown('<div class="panel"><div class="ptitle">Settings</div>', unsafe_allow_html=True)
            model_s = st.selectbox("YOLOv8 Model", ["yolov8n","yolov8s","yolov8m"])
            conf_t  = st.slider("Confidence", 0.1, 1.0, 0.5, 0.05)
            st.markdown('</div>', unsafe_allow_html=True)
            uploaded = st.file_uploader("Upload Image", type=["jpg","jpeg","png","webp"])
        with c2:
            if uploaded:
                from PIL import Image
                img = Image.open(uploaded)
                hub = VisionHub()
                result_img, detections = hub.detect_objects(img, conf=conf_t)
                st.image(result_img, use_column_width=True)
                if detections:
                    st.markdown('<div class="panel"><div class="ptitle">Detections</div>', unsafe_allow_html=True)
                    for d in detections:
                        cc = "cg" if d['confidence'] > 0.7 else "ca"
                        st.markdown(f'<span class="chip {cc}">{d["label"]} · {d["confidence"]:.2f}</span> ', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="text-align:center;padding:4rem;color:var(--muted);border:1px dashed var(--border);border-radius:16px">
                  <div style="font-size:2.5rem;margin-bottom:1rem">📸</div>
                  <div style="font-family:'Space Mono',monospace;font-size:.76rem">Upload an image to run detection</div>
                </div>""", unsafe_allow_html=True)

    with tab2:
        up2 = st.file_uploader("Upload Image for Pose", type=["jpg","jpeg","png"], key="pose")
        if up2:
            from PIL import Image
            hub = VisionHub()
            st.image(hub.estimate_pose(Image.open(up2)), use_column_width=True)
        else:
            st.info("Upload an image to run MediaPipe pose estimation.")

    with tab3:
        up3 = st.file_uploader("Upload Image for Face Mesh", type=["jpg","jpeg","png"], key="face")
        if up3:
            from PIL import Image
            hub = VisionHub()
            st.image(hub.face_mesh(Image.open(up3)), use_column_width=True)
        else:
            st.info("Upload an image to run MediaPipe face mesh detection.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: SENTIMENT RNN
# ═══════════════════════════════════════════════════════════════════════════════
elif "Sentiment" in page:
    from modules.sentiment_rnn import SentimentRNN
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    st.markdown("""
    <div class="shead">
      <div class="sicon" style="background:rgba(239,68,68,.1);border:1px solid rgba(239,68,68,.3);font-size:20px">📝</div>
      <div><h2 style="color:#f87171">Sentiment RNN (LSTM)</h2>
           <p>LSTM-based text classifier · train on custom examples · predict sentiment</p></div>
    </div>""", unsafe_allow_html=True)

    if show_formulas:
        st.markdown('<div class="formula">fₜ=σ(Wf·[hₜ₋₁,xₜ]+bf) &nbsp; iₜ=σ(Wi·[hₜ₋₁,xₜ]+bi) &nbsp; c̃ₜ=tanh(Wc·[hₜ₋₁,xₜ]+bc) &nbsp; cₜ=fₜ⊙cₜ₋₁+iₜ⊙c̃ₜ &nbsp; hₜ=oₜ⊙tanh(cₜ)</div>', unsafe_allow_html=True)

    t1, t2 = st.tabs(["🏋️ Train", "🔮 Predict"])

    with t1:
        c1, c2 = st.columns([1, 1.5])
        with c1:
            st.markdown('<div class="panel"><div class="ptitle">Hyperparameters</div>', unsafe_allow_html=True)
            hsz   = st.slider("Hidden Size",   16, 256, 64, 16)
            nlay  = st.slider("LSTM Layers",    1,   4,  2)
            lr_r  = st.slider("Learning Rate", 0.0001, 0.01, 0.001, 0.0001, format="%.4f")
            ep_r  = st.slider("Epochs",         5, 100, 20)
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('<div class="panel"><div class="ptitle">Training Data (text TAB label)</div>', unsafe_allow_html=True)
            sample = "I love this product, amazing!\tpositive\nAbsolutely terrible experience.\tnegative\nGreat quality and fast shipping.\tpositive\nWaste of money, very disappointed.\tnegative\nWorks perfectly, highly recommend!\tpositive\nBroken on arrival, do not buy.\tnegative"
            tdata = st.text_area("", sample, height=175)
            st.markdown('</div>', unsafe_allow_html=True)
            tr = st.button("🏋️ Train LSTM")

        with c2:
            if tr:
                rnn = SentimentRNN(hidden_size=hsz, num_layers=nlay, lr=lr_r)
                losses_r, accs = rnn.train(tdata, epochs=ep_r)
                st.session_state['rnn'] = rnn

                fig = make_subplots(rows=1, cols=2, subplot_titles=["Loss","Accuracy"])
                fig.add_trace(go.Scatter(y=losses_r, mode='lines+markers',
                    line=dict(color='#f87171', width=2), marker=dict(size=4),
                    fill='tozeroy', fillcolor='rgba(239,68,68,.07)'), row=1, col=1)
                fig.add_trace(go.Scatter(y=accs, mode='lines+markers',
                    line=dict(color='#10b981', width=2), marker=dict(size=4),
                    fill='tozeroy', fillcolor='rgba(16,185,129,.07)'), row=1, col=2)
                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#5a7a9a', family='JetBrains Mono', size=10),
                    height=260, showlegend=False, margin=dict(l=10,r=10,t=30,b=10))
                fig.update_xaxes(gridcolor='#1a3550', color='#5a7a9a')
                fig.update_yaxes(gridcolor='#1a3550', color='#5a7a9a')
                st.plotly_chart(fig, use_container_width=True)
                st.success(f"✅ Training complete · Final accuracy: {accs[-1]:.1f}%")
            else:
                st.markdown("""
                <div style="text-align:center;padding:3.5rem;color:var(--muted)">
                  <div style="font-size:2.5rem;margin-bottom:1rem">📝</div>
                  <div style="font-family:'Space Mono',monospace;font-size:.76rem">Configure & click Train LSTM</div>
                </div>""", unsafe_allow_html=True)

    with t2:
        ti = st.text_input("Enter text to classify", "This product is absolutely incredible!")
        pb = st.button("🔮 Predict Sentiment")
        if pb:
            if 'rnn' in st.session_state:
                rnn = st.session_state['rnn']
                label, prob = rnn.predict(ti)
                color = "var(--accent3)" if label == "positive" else "var(--danger)"
                emoji = "😊" if label == "positive" else "😞"
                st.markdown(f"""
                <div class="panel" style="text-align:center;padding:2.5rem">
                  <div style="font-size:3rem;margin-bottom:.8rem">{emoji}</div>
                  <div style="font-size:2rem;font-weight:700;color:{color};font-family:'Outfit',sans-serif;margin-bottom:.5rem">{label.upper()}</div>
                  <div style="font-family:'JetBrains Mono',monospace;color:var(--muted)">Confidence: {prob:.1%}</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.warning("Train the model first on the Train tab.")


# FOOTER
st.markdown("""
<div class="footer">
  ⚡ NEUROLAB NEURAL NETWORK TOOLBOX &nbsp;·&nbsp; STREAMLIT + PYTORCH &nbsp;·&nbsp; v1.0.0
</div>""", unsafe_allow_html=True)
