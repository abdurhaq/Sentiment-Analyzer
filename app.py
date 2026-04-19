import streamlit as st
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch.nn.functional as F
import time
import os

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SentimentLens",
    page_icon="◎",
    layout="centered",
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=Syne+Mono&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background: #0a0a0a !important;
    color: #f0ede6 !important;
    font-family: 'Syne', sans-serif !important;
}

[data-testid="stAppViewContainer"] {
    background: #0a0a0a !important;
}

[data-testid="stHeader"] { background: transparent !important; }

.block-container {
    max-width: 760px !important;
    padding: 3rem 2rem 4rem !important;
}

/* Hide default streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }

/* ── Hero ── */
.hero {
    text-align: center;
    padding: 3rem 0 2.5rem;
    border-bottom: 1px solid #222;
    margin-bottom: 2.5rem;
}
.hero-eyebrow {
    font-family: 'Syne Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.25em;
    color: #555;
    text-transform: uppercase;
    margin-bottom: 1rem;
}
.hero-title {
    font-size: clamp(3rem, 8vw, 5.5rem);
    font-weight: 800;
    line-height: 0.95;
    letter-spacing: -0.03em;
    color: #f0ede6;
    margin-bottom: 1.2rem;
}
.hero-title span {
    color: transparent;
    -webkit-text-stroke: 1px #f0ede6;
}
.hero-sub {
    font-size: 0.9rem;
    color: #555;
    letter-spacing: 0.05em;
}

/* ── Textarea ── */
textarea {
    background: #111 !important;
    border: 1px solid #2a2a2a !important;
    border-radius: 4px !important;
    color: #f0ede6 !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 1rem !important;
    resize: none !important;
    transition: border-color 0.2s !important;
}
textarea:focus {
    border-color: #f0ede6 !important;
    box-shadow: none !important;
    outline: none !important;
}

/* ── Button ── */
.stButton > button {
    background: #f0ede6 !important;
    color: #0a0a0a !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: 2px !important;
    padding: 0.75rem 2.5rem !important;
    cursor: pointer !important;
    width: 100% !important;
    transition: opacity 0.15s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

/* ── Result card ── */
.result-card {
    margin-top: 2.5rem;
    border: 1px solid #222;
    border-radius: 4px;
    overflow: hidden;
    animation: fadeUp 0.4s ease both;
}
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}
.result-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1.25rem 1.5rem;
}
.result-label {
    font-size: 0.65rem;
    font-family: 'Syne Mono', monospace;
    letter-spacing: 0.2em;
    color: #555;
    text-transform: uppercase;
    margin-bottom: 0.35rem;
}
.result-verdict {
    font-size: 2.8rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    line-height: 1;
}
.result-verdict.positive { color: #b8f5b0; }
.result-verdict.negative { color: #f5b0b0; }
.score-box {
    text-align: right;
}
.score-num {
    font-size: 2.8rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    line-height: 1;
    color: #f0ede6;
}
.score-label {
    font-size: 0.65rem;
    font-family: 'Syne Mono', monospace;
    letter-spacing: 0.2em;
    color: #555;
    text-transform: uppercase;
    margin-top: 0.35rem;
}

/* Bar */
.bar-wrap {
    height: 4px;
    background: #1a1a1a;
}
.bar-fill {
    height: 100%;
    transition: width 0.6s cubic-bezier(0.4,0,0.2,1);
}
.bar-fill.positive { background: #b8f5b0; }
.bar-fill.negative { background: #f5b0b0; }

/* Probabilities row */
.prob-row {
    display: flex;
    justify-content: space-between;
    padding: 1rem 1.5rem;
    border-top: 1px solid #1a1a1a;
    background: #0d0d0d;
}
.prob-item { text-align: center; }
.prob-val {
    font-size: 1.1rem;
    font-weight: 700;
    color: #f0ede6;
}
.prob-tag {
    font-size: 0.6rem;
    font-family: 'Syne Mono', monospace;
    letter-spacing: 0.15em;
    color: #444;
    text-transform: uppercase;
    margin-top: 0.2rem;
}

/* ── History ── */
.history-title {
    font-size: 0.65rem;
    font-family: 'Syne Mono', monospace;
    letter-spacing: 0.2em;
    color: #333;
    text-transform: uppercase;
    margin: 2.5rem 0 1rem;
    border-top: 1px solid #1a1a1a;
    padding-top: 2rem;
}
.history-item {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.75rem 0;
    border-bottom: 1px solid #111;
    gap: 1rem;
}
.history-text {
    font-size: 0.85rem;
    color: #666;
    flex: 1;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.history-badge {
    font-size: 0.65rem;
    font-family: 'Syne Mono', monospace;
    letter-spacing: 0.1em;
    padding: 0.2rem 0.6rem;
    border-radius: 2px;
    white-space: nowrap;
}
.history-badge.positive { background: #0f2b0f; color: #b8f5b0; }
.history-badge.negative { background: #2b0f0f; color: #f5b0b0; }

/* ── Examples ── */
.example-label {
    font-size: 0.65rem;
    font-family: 'Syne Mono', monospace;
    letter-spacing: 0.2em;
    color: #333;
    text-transform: uppercase;
    margin-bottom: 0.75rem;
}

/* spinner override */
[data-testid="stSpinner"] > div { border-top-color: #f0ede6 !important; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_model():
    tokenizer = DistilBertTokenizerFast.from_pretrained("abdurhaq/sentiment-lens")
    model = DistilBertForSequenceClassification.from_pretrained("abdurhaq/sentiment-lens")
    model.eval()
    return tokenizer, model

def predict(text: str, tokenizer, model):
    inputs = tokenizer(
        text, return_tensors="pt",
        truncation=True, padding=True, max_length=128
    )
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = F.softmax(logits, dim=-1)[0]
    neg, pos = probs[0].item(), probs[1].item()
    label = "POSITIVE" if pos > neg else "NEGATIVE"
    confidence = max(pos, neg)
    return label, confidence, pos, neg

# ── Session state ──────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-eyebrow">◎ &nbsp; Deep Learning · NLP · DistilBERT</div>
  <div class="hero-title">Sentiment<br><span>Lens</span></div>
  <div class="hero-sub">Fine-tuned on Stanford SST-2 · 91%+ accuracy</div>
</div>
""", unsafe_allow_html=True)

# ── Load model ─────────────────────────────────────────────────────────────────
with st.spinner("Loading model..."):
    tokenizer, model = load_model()

# ── Examples ───────────────────────────────────────────────────────────────────
st.markdown('<div class="example-label">Try an example</div>', unsafe_allow_html=True)

examples = [
    "This film was an absolute masterpiece.",
    "Worst experience I've ever had.",
    "The product is okay, nothing special.",
    "I can't stop recommending this to everyone!",
]

cols = st.columns(len(examples))
selected_example = None
for i, (col, ex) in enumerate(zip(cols, examples)):
    with col:
        if st.button(f"#{i+1}", key=f"ex_{i}", help=ex):
            selected_example = ex

# ── Input ──────────────────────────────────────────────────────────────────────
default_text = selected_example if selected_example else ""
text_input = st.text_area(
    "",
    value=default_text,
    placeholder="Type or paste any text — a review, tweet, headline...",
    height=130,
    label_visibility="collapsed",
)

run = st.button("Analyze →")

# ── Prediction ─────────────────────────────────────────────────────────────────
if run and text_input.strip():
    with st.spinner(""):
        time.sleep(0.3)  # tiny pause for UX feel
        label, confidence, pos, neg = predict(text_input.strip(), tokenizer, model)

    css_class = "positive" if label == "POSITIVE" else "negative"
    conf_pct   = f"{confidence * 100:.1f}%"
    pos_pct    = f"{pos * 100:.1f}%"
    neg_pct    = f"{neg * 100:.1f}%"
    bar_width  = f"{confidence * 100:.1f}%"

    st.markdown(f"""
    <div class="result-card">
      <div class="result-header">
        <div>
          <div class="result-label">Verdict</div>
          <div class="result-verdict {css_class}">{label}</div>
        </div>
        <div class="score-box">
          <div class="score-num">{conf_pct}</div>
          <div class="score-label">Confidence</div>
        </div>
      </div>
      <div class="bar-wrap">
        <div class="bar-fill {css_class}" style="width:{bar_width}"></div>
      </div>
      <div class="prob-row">
        <div class="prob-item">
          <div class="prob-val">{pos_pct}</div>
          <div class="prob-tag">Positive prob.</div>
        </div>
        <div class="prob-item">
          <div class="prob-val">{neg_pct}</div>
          <div class="prob-tag">Negative prob.</div>
        </div>
        <div class="prob-item">
          <div class="prob-val">{len(text_input.split())}</div>
          <div class="prob-tag">Words</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # save to history
    st.session_state.history.insert(0, {
        "text": text_input.strip(),
        "label": label,
        "conf": conf_pct,
        "class": css_class,
    })
    st.session_state.history = st.session_state.history[:8]

elif run and not text_input.strip():
    st.warning("Please enter some text first.")

# ── History ────────────────────────────────────────────────────────────────────
if st.session_state.history:
    st.markdown('<div class="history-title">Recent analyses</div>', unsafe_allow_html=True)
    for item in st.session_state.history:
        preview = item["text"][:72] + "..." if len(item["text"]) > 72 else item["text"]
        st.markdown(f"""
        <div class="history-item">
          <span class="history-text">{preview}</span>
          <span class="history-badge {item['class']}">{item['label']} · {item['conf']}</span>
        </div>
        """, unsafe_allow_html=True)
