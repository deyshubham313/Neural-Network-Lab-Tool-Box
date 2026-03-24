import streamlit as st
import numpy as np

# ── PAGE CONFIG ─────────────────────────────────────────────
st.set_page_config(
    page_title="NeuroLab · Neural Network Toolbox",
    page_icon="⚡",
    layout="wide"
)

# ── SIMPLE READABLE THEME ───────────────────────────────────
st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] {
    background: #f6f8fb;
    color: #1f2937;
    font-family: 'Segoe UI', sans-serif;
}

h1, h2, h3 { color: #0f172a; }

.section {
    background: white;
    padding: 24px;
    border-radius: 12px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    margin-bottom: 20px;
}

.hero {
    background: linear-gradient(135deg,#2563eb,#7c3aed);
    padding: 48px 24px;
    border-radius: 16px;
    color: white;
    text-align: center;
    margin-bottom: 30px;
}

.metric {
    background:#eef2ff;
    padding:18px;
    border-radius:10px;
    text-align:center;
}

.footer {
    text-align:center;
    color:#6b7280;
    font-size:14px;
    margin-top:40px;
    padding:20px 0;
}
</style>
""", unsafe_allow_html=True)

# ── TOP NAVIGATION ──────────────────────────────────────────
tabs = st.tabs([
    "🏠 Overview",
    "🧠 Perceptron",
    "➡️ Forward Prop",
    "⬅️ Backprop",
    "👁️ Vision",
    "📝 Sentiment LSTM"
])

# ═════════════════════════════════════════════════════════════
# OVERVIEW
# ═════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown("""
    <div class='hero'>
        <h1>⚡ NeuroLab Neural Network Toolbox</h1>
        <p>Interactive platform to learn and experiment with Deep Learning models</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown("<div class='metric'><h2>5</h2><p>Modules</p></div>", unsafe_allow_html=True)
    c2.markdown("<div class='metric'><h2>ML</h2><p>Algorithms</p></div>", unsafe_allow_html=True)
    c3.markdown("<div class='metric'><h2>Vision</h2><p>YOLO / MediaPipe</p></div>", unsafe_allow_html=True)
    c4.markdown("<div class='metric'><h2>NLP</h2><p>LSTM Models</p></div>", unsafe_allow_html=True)

    st.markdown("""
    <div class='section'>
    <h3>📘 About This Lab</h3>
    This toolbox helps you understand neural networks visually:
    <ul>
    <li>Perceptron learning</li>
    <li>Forward propagation</li>
    <li>Backpropagation</li>
    <li>Computer Vision models</li>
    <li>LSTM networks for NLP</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════
# PERCEPTRON
# ═════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.header("🧠 Perceptron & Logic Gates")
    st.write("Train a single-layer perceptron for basic logic gates.")

    gate = st.selectbox("Select Logic Gate", ["AND","OR","NAND","NOR","XOR"])
    lr = st.slider("Learning Rate", 0.01, 1.0, 0.1)
    epochs = st.slider("Epochs", 1, 200, 50)

    if st.button("Train Model"):
        st.success(f"Model trained for {gate} gate (demo)")
        st.line_chart(np.random.randn(epochs).cumsum())

    st.markdown("</div>", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════
# FORWARD PROP
# ═════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.header("➡️ Forward Propagation")
    st.write("Visualize activations through layers.")

    inputs = st.slider("Number of Inputs", 1, 10, 3)
    values = [st.number_input(f"Input {i+1}", value=0.5) for i in range(inputs)]

    st.bar_chart(np.abs(np.random.randn(inputs)))
    st.markdown("</div>", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════
# BACKPROP
# ═════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.header("⬅️ Backpropagation")
    st.write("Understand gradient descent and error minimization.")

    lr = st.slider("Learning Rate", 0.001, 1.0, 0.05)
    epochs = st.slider("Epochs", 100, 5000, 1000)

    if st.button("Start Training"):
        st.line_chart(np.exp(-np.linspace(0,5,epochs)))
        st.success("Training complete")

    st.markdown("</div>", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════
# VISION
# ═════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.header("👁️ Computer Vision Hub")
    st.write("Upload images to run detection models.")

    file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])
    if file:
        st.image(file, use_column_width=True)
        st.success("Processing complete (demo)")

    st.markdown("</div>", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════
# SENTIMENT LSTM
# ═════════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.header("📝 Sentiment Analysis (LSTM)")
    st.write("Train an LSTM to classify text sentiment.")

    text = st.text_area("Enter training text")
    if st.button("Train LSTM"):
        st.success("Training complete (demo)")

    user_text = st.text_input("Enter sentence for prediction")
    if st.button("Predict"):
        st.info("Predicted Sentiment: Positive 😊 (demo)")

    st.markdown("</div>", unsafe_allow_html=True)

# FOOTER
st.markdown("<div class='footer'>NeuroLab · Streamlit Neural Network Playground</div>", unsafe_allow_html=True)
