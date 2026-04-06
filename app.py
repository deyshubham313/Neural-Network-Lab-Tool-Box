import streamlit as st
import numpy as np
import pandas as pd
import cv2
from PIL import Image, ImageOps
from collections import Counter
import time

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NeuroLab · Neural Network Laboratory",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Session State ─────────────────────────────────────────────────────────────
if 'page' not in st.session_state:
    st.session_state.page = 'overview'
if 'p_results' not in st.session_state:
    st.session_state.p_results = None
if 'f_results' not in st.session_state:
    st.session_state.f_results = None
if 'b_results' not in st.session_state:
    st.session_state.b_results = None
if 's_results' not in st.session_state:
    st.session_state.s_results = None
if 'vision_frame' not in st.session_state:
    st.session_state.vision_frame = None

# ── CSS with 3D Animations & Overflow Fixes ───────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&family=JetBrains+Mono:wght@400;700&display=swap');

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

html, body {
    background: #050508;
    color: #ffffff;
    font-family: 'Inter', sans-serif;
    overflow-x: hidden !important;
    max-width: 100vw !important;
}

/* Hide Streamlit Chrome */
#MainMenu, footer, header, [data-testid="stSidebar"], [data-testid="stSidebarNav"] {
    display: none !important;
}

.block-container {
    padding: 0 !important;
    max-width: 100% !important;
    width: 100% !important;
}

/* Main Container with 3D Perspective */
.main-container {
    perspective: 2000px;
    width: 100%;
    min-height: 100vh;
    overflow-x: hidden;
    position: relative;
}

/* Animated Background Grid */
.bg-grid {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background-image: 
        linear-gradient(rgba(0, 240, 255, 0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0, 240, 255, 0.03) 1px, transparent 1px);
    background-size: 60px 60px;
    transform: perspective(1000px) rotateX(60deg) translateY(-100px) scale(1.5);
    transform-origin: center top;
    pointer-events: none;
    z-index: 0;
    animation: gridMove 20s linear infinite;
}

@keyframes gridMove {
    0% { background-position: 0 0; }
    100% { background-position: 60px 60px; }
}

/* Floating Orbs */
.orb {
    position: fixed;
    border-radius: 50%;
    filter: blur(100px);
    opacity: 0.4;
    z-index: 0;
    pointer-events: none;
}
.orb-1 { width: 400px; height: 400px; background: rgba(0, 240, 255, 0.3); top: -10%; left: -5%; animation: float 15s ease-in-out infinite; }
.orb-2 { width: 300px; height: 300px; background: rgba(184, 41, 221, 0.3); top: 40%; right: -5%; animation: float 18s ease-in-out infinite reverse; }
.orb-3 { width: 350px; height: 350px; background: rgba(0, 255, 136, 0.2); bottom: -10%; left: 30%; animation: float 20s ease-in-out infinite; }

@keyframes float {
    0%, 100% { transform: translate(0, 0) scale(1); }
    33% { transform: translate(30px, -30px) scale(1.1); }
    66% { transform: translate(-20px, 20px) scale(0.9); }
}

/* Navigation - Floating Glass */
.nav-float {
    position: fixed;
    top: 20px;
    left: 50%;
    transform: translateX(-50%) translateZ(50px);
    z-index: 1000;
    background: rgba(10, 10, 15, 0.8);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 100px;
    padding: 8px;
    display: flex;
    gap: 4px;
    box-shadow: 0 20px 40px rgba(0,0,0,0.5), 0 0 0 1px rgba(0,240,255,0.1);
    max-width: 95vw;
    overflow-x: auto;
    scrollbar-width: none;
}
.nav-float::-webkit-scrollbar { display: none; }

.nav-btn {
    background: transparent;
    border: none;
    color: rgba(255,255,255,0.6);
    padding: 10px 20px;
    border-radius: 100px;
    cursor: pointer;
    font-family: 'Inter', sans-serif;
    font-size: 0.85rem;
    font-weight: 500;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    white-space: nowrap;
    position: relative;
    overflow: hidden;
}

.nav-btn:hover {
    color: white;
    background: rgba(255,255,255,0.1);
    transform: translateY(-2px);
}

.nav-btn.active {
    background: linear-gradient(135deg, #00f0ff, #b829dd);
    color: white;
    box-shadow: 0 4px 20px rgba(0, 240, 255, 0.4);
}

/* 3D Cards with Tilt Effect */
.card-3d {
    background: linear-gradient(145deg, #111118, #0a0a0f);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 24px;
    padding: 24px;
    position: relative;
    transform-style: preserve-3d;
    transition: all 0.5s cubic-bezier(0.34, 1.56, 0.64, 1);
    box-shadow: 
        0 10px 30px rgba(0,0,0,0.5),
        inset 0 1px 0 rgba(255,255,255,0.05);
    overflow: hidden;
    width: 100%;
    max-width: 100%;
}

.card-3d::before {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(135deg, rgba(0,240,255,0.1), transparent 50%, rgba(184,41,221,0.1));
    opacity: 0;
    transition: opacity 0.3s;
    pointer-events: none;
    z-index: 0;
}

.card-3d:hover {
    transform: translateY(-10px) rotateX(5deg) rotateY(5deg) scale(1.02);
    border-color: rgba(0, 240, 255, 0.3);
    box-shadow: 
        0 30px 60px rgba(0,0,0,0.6),
        0 0 40px rgba(0, 240, 255, 0.1),
        inset 0 1px 0 rgba(255,255,255,0.1);
}

.card-3d:hover::before {
    opacity: 1;
}

.card-content {
    position: relative;
    z-index: 1;
}

/* Page Transition Container */
.page-wrapper {
    padding: 100px 40px 40px;
    max-width: 1400px;
    margin: 0 auto;
    position: relative;
    z-index: 1;
    animation: pageEnter 0.8s cubic-bezier(0.4, 0, 0.2, 1);
    width: 100%;
    box-sizing: border-box;
}

@keyframes pageEnter {
    from {
        opacity: 0;
        transform: translateY(40px) rotateX(10deg);
        filter: blur(10px);
    }
    to {
        opacity: 1;
        transform: translateY(0) rotateX(0);
        filter: blur(0);
    }
}

/* Neon Text Effects */
.neon-cyan {
    color: #00f0ff;
    text-shadow: 0 0 20px rgba(0, 240, 255, 0.5);
}
.neon-purple {
    color: #b829dd;
    text-shadow: 0 0 20px rgba(184, 41, 221, 0.5);
}
.neon-green {
    color: #00ff88;
    text-shadow: 0 0 20px rgba(0, 255, 136, 0.5);
}

/* Custom Scrollbar */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}
::-webkit-scrollbar-track {
    background: #050508;
}
::-webkit-scrollbar-thumb {
    background: #1e1e2e;
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover {
    background: #00f0ff;
}

/* Neural Network Canvas */
.neural-canvas-container {
    width: 100%;
    height: 400px;
    background: radial-gradient(circle at center, rgba(0,240,255,0.05), transparent 70%);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 24px;
    position: relative;
    overflow: hidden;
    margin: 20px 0;
}

/* Vision Camera Container - FIXED NO MIRROR */
.camera-container {
    border-radius: 20px;
    overflow: hidden;
    border: 2px solid rgba(0, 240, 255, 0.3);
    box-shadow: 0 0 40px rgba(0, 240, 255, 0.1);
    background: #000;
}

/* Metrics Grid */
.metrics-3d {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 16px;
    margin: 20px 0;
}

.metric-box {
    background: linear-gradient(145deg, #111118, #0a0a0f);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px;
    padding: 20px;
    text-align: center;
    transform-style: preserve-3d;
    transition: transform 0.3s;
}
.metric-box:hover {
    transform: translateZ(20px) scale(1.05);
    border-color: rgba(0, 240, 255, 0.3);
}

/* Responsive Fixes */
@media (max-width: 768px) {
    .page-wrapper { padding: 100px 20px 20px; }
    .nav-float { padding: 6px; }
    .nav-btn { padding: 8px 14px; font-size: 0.8rem; }
    .card-3d { padding: 16px; }
}

/* Streamlit Overrides */
.stButton>button {
    background: linear-gradient(135deg, #00f0ff, #b829dd) !important;
    color: black !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.75rem 1.5rem !important;
    transition: all 0.3s !important;
    width: 100%;
    position: relative;
    overflow: hidden;
}
.stButton>button:hover {
    transform: translateY(-3px) scale(1.02) !important;
    box-shadow: 0 10px 30px rgba(0, 240, 255, 0.4) !important;
}

.stSlider>div>div>div {
    background: rgba(0, 240, 255, 0.2) !important;
}
.stSlider>div>div>div>div {
    background: #00f0ff !important;
}

[data-testid="stDataFrame"] {
    background: rgba(10,10,15,0.8) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 12px !important;
    width: 100% !important;
}

/* Camera Input Fix */
[data-testid="stCameraInput"] {
    border-radius: 16px;
    overflow: hidden;
}
[data-testid="stCameraInput"] > div {
    border: none !important;
    background: rgba(0,0,0,0.5) !important;
}
[data-testid="stCameraInput"] video {
    transform: scaleX(-1); /* Fix mirror in preview */
}
</style>

<div class="bg-grid"></div>
<div class="orb orb-1"></div>
<div class="orb orb-2"></div>
<div class="orb orb-3"></div>
""", unsafe_allow_html=True)

# ── Math Helpers ──────────────────────────────────────────────────────────────
def sigmoid(x): 
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
def relu(x): return np.maximum(0, x)
def tanh(x): return np.tanh(x)

# ── 3D Neural Animation ───────────────────────────────────────────────────────
def get_neural_html():
    return """
    <div class="neural-canvas-container">
        <canvas id="netCanvas" style="width:100%; height:100%;"></canvas>
        <script>
            const canvas = document.getElementById('netCanvas');
            const ctx = canvas.getContext('2d');
            let width = canvas.offsetWidth;
            let height = canvas.offsetHeight;
            canvas.width = width;
            canvas.height = height;
            
            let time = 0;
            const layers = [5, 8, 8, 5];
            const nodes = [];
            
            layers.forEach((count, l) => {
                const x = (width / (layers.length + 1)) * (l + 1);
                for(let i = 0; i < count; i++) {
                    const y = (height / (count + 1)) * (i + 1);
                    nodes.push({x, y, layer: l, phase: Math.random() * Math.PI * 2});
                }
            });
            
            function draw() {
                ctx.fillStyle = 'rgba(5, 5, 8, 0.05)';
                ctx.fillRect(0, 0, width, height);
                time += 0.016;
                
                // Draw connections
                nodes.forEach((a, i) => {
                    nodes.forEach((b, j) => {
                        if(b.layer === a.layer + 1) {
                            const pulse = (Math.sin(time * 2 + a.phase) + 1) / 2;
                            ctx.beginPath();
                            ctx.moveTo(a.x, a.y);
                            ctx.lineTo(b.x, b.y);
                            ctx.strokeStyle = `rgba(0, 240, 255, ${0.05 + pulse * 0.15})`;
                            ctx.lineWidth = 1;
                            ctx.stroke();
                        }
                    });
                });
                
                // Draw nodes
                nodes.forEach(node => {
                    const pulse = Math.sin(time * 3 + node.phase);
                    const radius = 5 + pulse * 2;
                    
                    // Glow
                    const grad = ctx.createRadialGradient(node.x, node.y, 0, node.x, node.y, radius * 4);
                    grad.addColorStop(0, 'rgba(0, 240, 255, 0.3)');
                    grad.addColorStop(1, 'rgba(0, 240, 255, 0)');
                    ctx.fillStyle = grad;
                    ctx.beginPath();
                    ctx.arc(node.x, node.y, radius * 4, 0, Math.PI * 2);
                    ctx.fill();
                    
                    // Core
                    ctx.fillStyle = '#00f0ff';
                    ctx.beginPath();
                    ctx.arc(node.x, node.y, radius, 0, Math.PI * 2);
                    ctx.fill();
                    
                    // Highlight
                    ctx.fillStyle = 'rgba(255,255,255,0.8)';
                    ctx.beginPath();
                    ctx.arc(node.x - 2, node.y - 2, radius * 0.4, 0, Math.PI * 2);
                    ctx.fill();
                });
                
                requestAnimationFrame(draw);
            }
            draw();
            
            window.addEventListener('resize', () => {
                width = canvas.offsetWidth;
                height = canvas.offsetHeight;
                canvas.width = width;
                canvas.height = height;
            });
        </script>
    </div>
    """

# ── MediaPipe Models (FIXED IMPORTS FOR ALL VERSIONS) ─────────────────────────
@st.cache_resource
def get_mediapipe_models():
    try:
        import mediapipe as mp
        
        # Standard MediaPipe 0.9.x and 0.10.x API
        return {
            'face_detection': mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.6),
            'face_mesh': mp.solutions.face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            ),
            'pose': mp.solutions.pose.Pose(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            ),
            'hands': mp.solutions.hands.Hands(
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            ),
            'mp_drawing': mp.solutions.drawing_utils,
            'mp_drawing_styles': mp.solutions.drawing_styles,
            'mp_face_mesh': mp.solutions.face_mesh,
            'mp_pose': mp.solutions.pose,
            'mp_hands': mp.solutions.hands
        }
    except Exception as e:
        st.error(f"MediaPipe initialization error: {e}")
        return None

# ── Navigation ────────────────────────────────────────────────────────────────
pages = ["overview", "perceptron", "forward", "backprop", "vision", "sentiment"]
labels = ["🏠 Home", "⚡ Perceptron", "➡️ Forward", "🔄 Backprop", "👁️ Vision", "💭 Sentiment"]

st.markdown('<div class="nav-float">', unsafe_allow_html=True)
cols = st.columns(len(pages))
for i, (page, label) in enumerate(zip(pages, labels)):
    with cols[i]:
        if st.button(label, key=f"btn_{page}", 
                    type="primary" if st.session_state.page == page else "secondary",
                    use_container_width=True):
            st.session_state.page = page
            st.rerun()
st.markdown('</div>', unsafe_allow_html=True)

# ── Page Content ──────────────────────────────────────────────────────────────
page = st.session_state.page

st.markdown(f'<div class="page-wrapper">', unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# OVERVIEW PAGE
# ═════════════════════════════════════════════════════════════════════════════
if page == "overview":
    st.markdown("""
    <div style="text-align: center; padding: 40px 0;">
        <h1 style="font-size: clamp(2.5rem, 6vw, 4rem); font-weight: 800; margin-bottom: 1rem; letter-spacing: -2px;">
            <span class="neon-cyan">Neuro</span><span style="color: white;">Lab</span>
        </h1>
        <p style="font-size: 1.25rem; color: rgba(255,255,255,0.6); max-width: 600px; margin: 0 auto;">
            Interactive 3D Neural Network Laboratory
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.components.v1.html(get_neural_html(), height=400)
    
    # Module Grid
    cols = st.columns(3)
    modules = [
        ("perceptron", "⚡", "Perceptron", "Train logic gates with single-layer networks", "cyan"),
        ("forward", "➡️", "Forward Pass", "Visualize data flow through MLP layers", "purple"),
        ("backprop", "🔄", "Backpropagation", "Train XOR with gradient descent visualization", "green"),
        ("vision", "👁️", "Vision Lab", "Real-time computer vision with MediaPipe", "cyan"),
        ("sentiment", "💭", "Sentiment LSTM", "Text classification with RNNs", "purple"),
        ("", "📊", "Coming Soon", "More modules under development", "muted")
    ]
    
    for i, (key, icon, title, desc, color) in enumerate(modules):
        with cols[i % 3]:
            color_class = f"neon-{color}" if color != "muted" else ""
            if key:  # Clickable modules
                with st.container():
                    st.markdown(f"""
                    <div class="card-3d" style="cursor: pointer; height: 250px; display: flex; flex-direction: column; justify-content: center; align-items: center; text-align: center;" 
                         onclick="document.getElementById('go_{key}').click()">
                        <div style="font-size: 3rem; margin-bottom: 1rem;">{icon}</div>
                        <h3 class="{color_class}" style="margin-bottom: 0.5rem; font-size: 1.25rem;">{title}</h3>
                        <p style="color: rgba(255,255,255,0.6); font-size: 0.9rem; line-height: 1.5;">{desc}</p>
                        <div style="margin-top: auto; padding-top: 1rem;">
                            <span style="font-size: 0.75rem; opacity: 0.5; font-family: monospace;">Click to explore →</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    if st.button("Go", key=f"go_{key}", use_container_width=True):
                        st.session_state.page = key
                        st.rerun()
            else:  # Coming soon
                st.markdown(f"""
                <div class="card-3d" style="opacity: 0.5; height: 250px; display: flex; flex-direction: column; justify-content: center; align-items: center; text-align: center;">
                    <div style="font-size: 3rem; margin-bottom: 1rem;">{icon}</div>
                    <h3 style="margin-bottom: 0.5rem; font-size: 1.25rem; color: rgba(255,255,255,0.4);">{title}</h3>
                    <p style="color: rgba(255,255,255,0.4); font-size: 0.9rem;">{desc}</p>
                </div>
                """, unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# PERCEPTRON PAGE
# ═════════════════════════════════════════════════════════════════════════════
elif page == "perceptron":
    col1, col2 = st.columns([1, 1], gap="medium")
    
    with col1:
        st.markdown("""
        <div style="margin-bottom: 2rem;">
            <h2 class="neon-cyan" style="font-size: 2rem; margin-bottom: 0.5rem;">Perceptron Lab</h2>
            <p style="color: rgba(255,255,255,0.6);">Binary classification with the perceptron learning rule</p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.container():
            st.markdown('<div class="card-3d">', unsafe_allow_html=True)
            gate = st.selectbox("Logic Gate", ["AND", "OR", "NAND", "NOR", "XOR"])
            lr = st.slider("Learning Rate", 0.01, 1.0, 0.1, 0.01)
            epochs = st.slider("Epochs", 10, 500, 100, 10)
            
            if gate == "XOR":
                st.warning("⚠️ XOR is not linearly separable")
            
            if st.button("🚀 Train Perceptron", use_container_width=True):
                X = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=float)
                y = {"AND":[0,0,0,1], "OR":[0,1,1,1], "NAND":[1,1,1,0], "NOR":[1,0,0,0], "XOR":[0,1,1,0]}[gate]
                y = np.array(y, dtype=float)
                
                w = np.random.randn(2) * 0.1
                b = 0.0
                losses = []
                
                prog = st.progress(0)
                for ep in range(epochs):
                    err = 0
                    for xi, yi in zip(X, y):
                        pred = 1 if np.dot(w, xi) + b >= 0 else 0
                        error = yi - pred
                        w += lr * error * xi
                        b += lr * error
                        err += abs(error)
                    losses.append(err)
                    prog.progress((ep+1)/epochs)
                    if err == 0 and gate != "XOR":
                        losses.extend([0] * (epochs - ep - 1))
                        break
                prog.empty()
                
                st.session_state.p_results = {'w':w, 'b':b, 'losses':losses, 'gate':gate, 'X':X, 'y':y}
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        if st.session_state.p_results:
            res = st.session_state.p_results
            st.markdown('<div class="card-3d">', unsafe_allow_html=True)
            
            status = "✅ Converged" if res['losses'][-1] == 0 else "⚠️ Not Converged"
            color = "neon-green" if res['losses'][-1] == 0 else "neon-purple"
            st.markdown(f"<h3 class='{color}'>{status}</h3>", unsafe_allow_html=True)
            
            cols = st.columns(3)
            with cols[0]: st.metric("Error", f"{int(res['losses'][-1])}/4")
            with cols[1]: st.metric("Epochs", len([l for l in res['losses'] if l != 0]) or len(res['losses']))
            with cols[2]: st.metric("Weights", f"{res['w'][0]:.2f}, {res['w'][1]:.2f}")
            
            st.line_chart(res['losses'], height=200)
            
            # Truth Table
            table = []
            for xi, yi in zip(res['X'], res['y']):
                pred = 1 if np.dot(res['w'], xi) + res['b'] >= 0 else 0
                table.append({
                    'A': int(xi[0]), 'B': int(xi[1]), 
                    'Target': int(yi), 'Output': pred,
                    'Match': '✅' if pred == yi else '❌'
                })
            st.dataframe(pd.DataFrame(table), use_container_width=True, hide_index=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="card-3d" style="height: 400px; display: flex; align-items: center; justify-content: center; color: rgba(255,255,255,0.3);">
                <div style="text-align: center;">
                    <div style="font-size: 3rem; margin-bottom: 1rem;">🧮</div>
                    <p>Train to see results</p>
                </div>
            </div>
            """)

# ═════════════════════════════════════════════════════════════════════════════
# FORWARD PASS PAGE
# ═════════════════════════════════════════════════════════════════════════════
elif page == "forward":
    col1, col2 = st.columns([1, 1], gap="medium")
    
    with col1:
        st.markdown("""
        <div style="margin-bottom: 2rem;">
            <h2 class="neon-purple" style="font-size: 2rem; margin-bottom: 0.5rem;">Forward Pass</h2>
            <p style="color: rgba(255,255,255,0.6);">Visualize activation flow through network layers</p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.container():
            st.markdown('<div class="card-3d">', unsafe_allow_html=True)
            n_in = st.slider("Input Neurons", 2, 8, 3)
            n_hid = st.slider("Hidden Neurons", 2, 12, 5)
            n_out = st.slider("Output Neurons", 1, 4, 2)
            activation = st.selectbox("Activation", ["ReLU", "Sigmoid", "Tanh"])
            
            st.markdown("### Input Values")
            inputs = []
            for i in range(n_in):
                val = st.slider(f"Input {i+1}", -2.0, 2.0, 0.5, 0.1, key=f"in_{i}")
                inputs.append(val)
            
            if st.button("▶️ Run Forward", use_container_width=True):
                np.random.seed(42)
                x = np.array(inputs)
                W1 = np.random.randn(n_hid, n_in) * 0.5
                b1 = np.zeros(n_hid)
                W2 = np.random.randn(n_out, n_hid) * 0.5
                b2 = np.zeros(n_out)
                
                z1 = W1 @ x + b1
                a1 = relu(z1) if activation == "ReLU" else sigmoid(z1) if activation == "Sigmoid" else tanh(z1)
                z2 = W2 @ a1 + b2
                a2 = sigmoid(z2)
                
                st.session_state.f_results = {
                    'x': x, 'a1': a1, 'a2': a2,
                    'arch': f"{n_in}-{n_hid}-{n_out}", 'act': activation
                }
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        if st.session_state.f_results:
            r = st.session_state.f_results
            st.markdown('<div class="card-3d">', unsafe_allow_html=True)
            st.markdown(f"<h4 style='text-align: center; color: #b829dd;'>{r['arch']}</h4>", unsafe_allow_html=True)
            
            # Visualization
            cols = st.columns(3)
            with cols[0]:
                st.markdown("<p style='text-align: center; opacity: 0.6; font-size: 0.8rem;'>INPUT</p>", unsafe_allow_html=True)
                for v in r['x']:
                    st.markdown(f'<div style="padding: 8px; background: rgba(0,240,255,0.1); border: 1px solid rgba(0,240,255,0.3); border-radius: 8px; margin: 4px 0; text-align: center; font-family: monospace;">{v:.3f}</div>', unsafe_allow_html=True)
            
            with cols[1]:
                st.markdown(f"<p style='text-align: center; opacity: 0.6; font-size: 0.8rem;'>HIDDEN ({r['act']})</p>", unsafe_allow_html=True)
                for v in r['a1'][:6]:
                    intensity = min(abs(v), 1.0)
                    bg = f"rgba(0, 240, 255, {0.1 + intensity * 0.3})"
                    st.markdown(f'<div style="padding: 8px; background: {bg}; border: 1px solid rgba(0,240,255,0.3); border-radius: 8px; margin: 4px 0; text-align: center; font-family: monospace;">{v:.3f}</div>', unsafe_allow_html=True)
            
            with cols[2]:
                st.markdown("<p style='text-align: center; opacity: 0.6; font-size: 0.8rem;'>OUTPUT</p>", unsafe_allow_html=True)
                for v in r['a2']:
                    prob = v * 100
                    st.markdown(f'<div style="padding: 12px; background: linear-gradient(135deg, rgba(184,41,221,0.2), rgba(0,240,255,0.2)); border: 1px solid rgba(184,41,221,0.5); border-radius: 8px; margin: 4px 0; text-align: center;"><div style="font-family: monospace; font-weight: bold;">{v:.4f}</div><div style="font-size: 0.75rem; color: #b829dd;">{prob:.1f}%</div></div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="card-3d" style="height: 400px; display: flex; align-items: center; justify-content: center; color: rgba(255,255,255,0.3);">
                <div style="text-align: center;">
                    <div style="font-size: 3rem; margin-bottom: 1rem;">➡️</div>
                    <p>Run forward pass to visualize</p>
                </div>
            </div>
            """)

# ═════════════════════════════════════════════════════════════════════════════
# BACKPROP PAGE
# ═════════════════════════════════════════════════════════════════════════════
elif page == "backprop":
    col1, col2 = st.columns([1, 1], gap="medium")
    
    with col1:
        st.markdown("""
        <div style="margin-bottom: 2rem;">
            <h2 class="neon-green" style="font-size: 2rem; margin-bottom: 0.5rem;">Backpropagation</h2>
            <p style="color: rgba(255,255,255,0.6);">Train 2-layer MLP on XOR dataset</p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.container():
            st.markdown('<div class="card-3d">', unsafe_allow_html=True)
            hidden = st.slider("Hidden Size", 2, 16, 8)
            lr = st.slider("Learning Rate", 0.001, 0.5, 0.05, 0.001, format="%.3f")
            epochs = st.slider("Epochs", 100, 3000, 1000, 100)
            
            if st.button("🎯 Train XOR", use_container_width=True):
                X = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=float)
                Y = np.array([[0], [1], [1], [0]], dtype=float)
                
                np.random.seed(1)
                W1 = np.random.randn(2, hidden) * np.sqrt(2.0/2)
                b1 = np.zeros((1, hidden))
                W2 = np.random.randn(hidden, 1) * np.sqrt(2.0/hidden)
                b2 = np.zeros((1, 1))
                
                losses = []
                bar = st.progress(0)
                
                for ep in range(epochs):
                    z1 = X @ W1 + b1
                    a1 = np.tanh(z1)
                    z2 = a1 @ W2 + b2
                    a2 = sigmoid(z2)
                    
                    loss = -np.mean(Y * np.log(a2 + 1e-8) + (1-Y) * np.log(1-a2 + 1e-8))
                    losses.append(float(loss))
                    
                    dz2 = a2 - Y
                    dW2 = a1.T @ dz2 / 4
                    db2 = np.sum(dz2, axis=0, keepdims=True) / 4
                    da1 = dz2 @ W2.T
                    dz1 = da1 * (1 - a1**2)
                    dW1 = X.T @ dz1 / 4
                    db1 = np.sum(dz1, axis=0, keepdims=True) / 4
                    
                    W2 -= lr * dW2
                    b2 -= lr * db2
                    W1 -= lr * dW1
                    b1 -= lr * db1
                    
                    if ep % max(1, epochs//50) == 0:
                        bar.progress((ep+1)/epochs)
                bar.empty()
                
                st.session_state.b_results = {
                    'losses': losses, 'W1': W1, 'W2': W2, 'b1': b1, 'b2': b2
                }
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        if st.session_state.b_results:
            r = st.session_state.b_results
            st.markdown('<div class="card-3d">', unsafe_allow_html=True)
            st.markdown(f"<h3 style='color: #00ff88; margin-bottom: 1rem;'>Final Loss: {r['losses'][-1]:.6f}</h3>", unsafe_allow_html=True)
            
            st.line_chart(r['losses'], height=250)
            
            X = np.array([[0,0], [0,1], [1,0], [1,1]])
            z1 = X @ r['W1'] + r['b1']
            a1 = np.tanh(z1)
            z2 = a1 @ r['W2'] + r['b2']
            preds = sigmoid(z2)
            
            table = []
            for i, (xi, p) in enumerate(zip(X, preds)):
                target = 1 if xi.sum() == 1 else 0
                table.append({
                    'Input': str(list(map(int, xi))),
                    'Target': target,
                    'Output': f"{p[0]:.4f}",
                    'Pred': int(p > 0.5),
                    '✓': '✅' if (p > 0.5) == target else '❌'
                })
            st.dataframe(pd.DataFrame(table), use_container_width=True, hide_index=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="card-3d" style="height: 400px; display: flex; align-items: center; justify-content: center; color: rgba(255,255,255,0.3);">
                <div style="text-align: center;">
                    <div style="font-size: 3rem; margin-bottom: 1rem;">🔄</div>
                    <p>Train the network</p>
                </div>
            </div>
            """)

# ═════════════════════════════════════════════════════════════════════════════
# VISION LAB PAGE - FIXED WITH ALL MODES & NO MIRROR
# ═════════════════════════════════════════════════════════════════════════════
elif page == "vision":
    st.markdown("""
    <div style="margin-bottom: 2rem;">
        <h2 class="neon-cyan" style="font-size: 2rem; margin-bottom: 0.5rem;">Vision Laboratory</h2>
        <p style="color: rgba(255,255,255,0.6);">Real-time computer vision (Camera fixed - no mirror effect)</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2], gap="medium")
    
    with col1:
        with st.container():
            st.markdown('<div class="card-3d">', unsafe_allow_html=True)
            mode = st.selectbox("Vision Mode", [
                "Face Detection", 
                "Face Mesh (468 landmarks)", 
                "Pose Estimation", 
                "Hand Tracking",
                "Edge Detection"
            ])
            
            st.info("📸 Take a photo to process")
            camera_input = st.camera_input("Capture", label_visibility="collapsed")
            
            if mode == "Edge Detection":
                low = st.slider("Low Threshold", 0, 255, 50)
                high = st.slider("High Threshold", 0, 255, 150)
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        if camera_input is not None:
            # Load image
            bytes_data = camera_input.getvalue()
            img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            
            # IMPORTANT: Flip horizontally to fix mirror effect
            img = cv2.flip(img, 1)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            models = get_mediapipe_models()
            
            if models is None:
                st.error("Failed to load MediaPipe models. Please check your installation.")
            else:
                h, w = img.shape[:2]
                processed_img = img_rgb.copy()
                stats = {}
                
                # Process based on mode
                if mode == "Face Detection":
                    results = models['face_detection'].process(img_rgb)
                    stats['Faces'] = 0
                    if results.detections:
                        stats['Faces'] = len(results.detections)
                        for detection in results.detections:
                            bboxC = detection.location_data.relative_bounding_box
                            x = int(bboxC.xmin * w)
                            y = int(bboxC.ymin * h)
                            width = int(bboxC.width * w)
                            height = int(bboxC.height * h)
                            
                            cv2.rectangle(processed_img, (x, y), (x+width, y+height), (0, 240, 255), 3)
                            cv2.putText(processed_img, f"{int(detection.score[0]*100)}%", (x, y-10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 240, 255), 2)
                
                elif mode == "Face Mesh (468 landmarks)":
                    results = models['face_mesh'].process(img_rgb)
                    stats['Faces'] = 0
                    stats['Landmarks'] = 0
                    if results.multi_face_landmarks:
                        stats['Faces'] = len(results.multi_face_landmarks)
                        for face_landmarks in results.multi_face_landmarks:
                            stats['Landmarks'] = len(face_landmarks.landmark)
                            models['mp_drawing'].draw_landmarks(
                                image=processed_img,
                                landmark_list=face_landmarks,
                                connections=models['mp_face_mesh'].FACEMESH_TESSELATION,
                                landmark_drawing_spec=None,
                                connection_drawing_spec=models['mp_drawing_styles'].get_default_face_mesh_tesselation_style()
                            )
                            models['mp_drawing'].draw_landmarks(
                                image=processed_img,
                                landmark_list=face_landmarks,
                                connections=models['mp_face_mesh'].FACEMESH_CONTOURS,
                                landmark_drawing_spec=None,
                                connection_drawing_spec=models['mp_drawing_styles'].get_default_face_mesh_contours_style()
                            )
                
                elif mode == "Pose Estimation":
                    results = models['pose'].process(img_rgb)
                    stats['Pose'] = "Not Detected"
                    if results.pose_landmarks:
                        stats['Pose'] = "Detected"
                        visible = sum(1 for lm in results.pose_landmarks.landmark if lm.visibility > 0.5)
                        stats['Keypoints'] = f"{visible}/33"
                        
                        models['mp_drawing'].draw_landmarks(
                            processed_img,
                            results.pose_landmarks,
                            models['mp_pose'].POSE_CONNECTIONS,
                            landmark_drawing_spec=models['mp_drawing_styles'].get_default_pose_landmarks_style()
                        )
                
                elif mode == "Hand Tracking":
                    results = models['hands'].process(img_rgb)
                    stats['Hands'] = 0
                    if results.multi_hand_landmarks:
                        stats['Hands'] = len(results.multi_hand_landmarks)
                        for hand_landmarks in results.multi_hand_landmarks:
                            models['mp_drawing'].draw_landmarks(
                                processed_img,
                                hand_landmarks,
                                models['mp_hands'].HAND_CONNECTIONS,
                                models['mp_drawing_styles'].get_default_hand_landmarks_style(),
                                models['mp_drawing_styles'].get_default_hand_connections_style()
                            )
                
                elif mode == "Edge Detection":
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                    edges = cv2.Canny(blurred, low, high)
                    processed_img = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
                    stats['Edge Density'] = f"{(np.sum(edges > 0) / (h*w))*100:.2f}%"
                
                # Display processed image (no mirror now - already flipped)
                st.markdown('<div class="camera-container">', unsafe_allow_html=True)
                st.image(processed_img, channels="RGB", use_column_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Stats
                if stats:
                    st.markdown('<div class="metrics-3d">', unsafe_allow_html=True)
                    stat_cols = st.columns(len(stats))
                    for col, (key, val) in zip(stat_cols, stats.items()):
                        with col:
                            st.markdown(f"""
                            <div class="metric-box">
                                <div style="font-size: 1.5rem; font-weight: bold; color: #00f0ff; font-family: monospace;">{val}</div>
                                <div style="font-size: 0.75rem; color: rgba(255,255,255,0.6);">{key}</div>
                            </div>
                            """, unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="card-3d" style="height: 400px; display: flex; align-items: center; justify-content: center; color: rgba(255,255,255,0.3);">
                <div style="text-align: center;">
                    <div style="font-size: 3rem; margin-bottom: 1rem;">👁️</div>
                    <p>Capture an image to analyze</p>
                </div>
            </div>
            """)

# ═════════════════════════════════════════════════════════════════════════════
# SENTIMENT PAGE
# ═════════════════════════════════════════════════════════════════════════════
elif page == "sentiment":
    col1, col2 = st.columns([1, 1], gap="medium")
    
    with col1:
        st.markdown("""
        <div style="margin-bottom: 2rem;">
            <h2 class="neon-purple" style="font-size: 2rem; margin-bottom: 0.5rem;">Sentiment Analysis</h2>
            <p style="color: rgba(255,255,255,0.6);">Bag-of-Words text classification</p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.container():
            st.markdown('<div class="card-3d">', unsafe_allow_html=True)
            data = st.text_area("Training Data (text[tab]label)", 
                "I love this product\t1\nAmazing experience\t1\nGreat quality\t1\nTerrible\t0\nBad product\t0\nWorst ever\t0", 
                height=200)
            lr = st.number_input("Learning Rate", 0.001, 1.0, 0.01, 0.001)
            epochs = st.number_input("Epochs", 10, 500, 100, 10)
            
            if st.button("🏋️ Train Model", use_container_width=True):
                lines = [l.strip() for l in data.split('\n') if '\t' in l]
                texts = [l.split('\t')[0].lower().split() for l in lines]
                labels = [int(l.split('\t')[1]) for l in lines]
                
                vocab = {w:i for i,w in enumerate(set([w for t in texts for w in t]))}
                V = len(vocab)
                
                X = np.zeros((len(texts), V))
                for i, text in enumerate(texts):
                    for w in text:
                        if w in vocab: X[i, vocab[w]] += 1
                X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
                y = np.array(labels)
                
                w = np.zeros(V)
                b = 0.0
                losses = []
                
                bar = st.progress(0)
                for ep in range(epochs):
                    preds = sigmoid(X @ w + b)
                    loss = -np.mean(y * np.log(preds + 1e-8) + (1-y) * np.log(1-preds + 1e-8))
                    losses.append(loss)
                    
                    dw = X.T @ (preds - y) / len(y)
                    db = np.mean(preds - y)
                    w -= lr * dw
                    b -= lr * db
                    bar.progress((ep+1)/epochs)
                bar.empty()
                
                st.session_state.s_results = {'w':w, 'b':b, 'vocab':vocab, 'losses':losses}
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        if st.session_state.s_results:
            r = st.session_state.s_results
            st.markdown('<div class="card-3d">', unsafe_allow_html=True)
            
            test = st.text_input("Test Text", "This is amazing!")
            if st.button("Analyze Sentiment", use_container_width=True):
                words = test.lower().split()
                x = np.zeros(len(r['vocab']))
                for w in words:
                    if w in r['vocab']: x[r['vocab'][w]] += 1
                if np.sum(x) > 0: x = x / np.linalg.norm(x)
                
                prob = sigmoid(x @ r['w'] + r['b'])
                pred = "Positive 😊" if prob > 0.5 else "Negative 😞"
                conf = prob if prob > 0.5 else 1-prob
                
                color = "#00ff88" if prob > 0.5 else "#ff3366"
                emoji = "😊" if prob > 0.5 else "😞"
                
                st.markdown(f"""
                <div style="text-align: center; padding: 2rem; border: 2px solid {color}; border-radius: 16px; margin: 1rem 0; background: rgba({0 if prob > 0.5 else 255}, {255 if prob > 0.5 else 51}, {136 if prob > 0.5 else 102}, 0.1);">
                    <div style="font-size: 3rem; margin-bottom: 0.5rem;">{emoji}</div>
                    <div style="font-size: 1.5rem; font-weight: bold; color: {color};">{pred}</div>
                    <div style="font-family: monospace; margin-top: 0.5rem; color: rgba(255,255,255,0.8);">{conf*100:.1f}% confidence</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.line_chart(r['losses'], height=200)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="card-3d" style="height: 400px; display: flex; align-items: center; justify-content: center; color: rgba(255,255,255,0.3);">
                <div style="text-align: center;">
                    <div style="font-size: 3rem; margin-bottom: 1rem;">💭</div>
                    <p>Train a model to classify</p>
                </div>
            </div>
            """)

st.markdown('</div>', unsafe_allow_html=True)  # Close page-wrapper