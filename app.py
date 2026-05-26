import streamlit as st
import os, re, torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

# ── Page config ────────────────────────────────────────────────────────
st.set_page_config(page_title="MindScan", page_icon="🧠", layout="centered")

# ── Inject Custom CSS ──────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');

* { font-family: 'Outfit', sans-serif !important; }

.stApp {
    background: linear-gradient(135deg, #0a0a1a 0%, #12122a 50%, #0f0a1e 100%);
    min-height: 100vh;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem !important; max-width: 720px !important; }

/* Hide Streamlit's built-in running indicator — bleeds into expander titles */
[data-testid="stStatusWidget"] { display: none !important; }

.stApp::before {
    content: '';
    position: fixed;
    width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(124,58,237,0.15) 0%, transparent 70%);
    top: -100px; left: -100px;
    border-radius: 50%;
    pointer-events: none;
    z-index: 0;
}
.stApp::after {
    content: '';
    position: fixed;
    width: 350px; height: 350px;
    background: radial-gradient(circle, rgba(6,182,212,0.12) 0%, transparent 70%);
    bottom: -80px; right: -80px;
    border-radius: 50%;
    pointer-events: none;
    z-index: 0;
}

.hero-title {
    text-align: center;
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(90deg, #7c3aed, #06b6d4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}
.hero-sub {
    text-align: center;
    color: #94a3b8;
    font-size: 1rem;
    margin-bottom: 2rem;
}

div[data-testid="stVerticalBlock"] > div:has(> div[data-testid="stTextArea"]) {
    background: rgba(255,255,255,0.04);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 20px;
    padding: 1.5rem 2rem 1rem 2rem;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4), 0 0 60px rgba(124,58,237,0.08);
    margin-bottom: 0.5rem;
}

.stTextArea textarea {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(124,58,237,0.3) !important;
    border-radius: 12px !important;
    color: #f1f5f9 !important;
    font-size: 1rem !important;
    padding: 1rem !important;
    transition: border-color 0.3s, box-shadow 0.3s !important;
}
.stTextArea textarea:focus {
    border-color: #7c3aed !important;
    box-shadow: 0 0 20px rgba(124,58,237,0.3) !important;
}
.stTextArea label {
    color: #e2e8f0 !important;
    font-size: 1.1rem !important;
    font-weight: 600 !important;
}
.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #06b6d4) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.75rem 2.5rem !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    width: 100% !important;
    transition: all 0.3s !important;
    box-shadow: 0 4px 20px rgba(124,58,237,0.4) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(124,58,237,0.6) !important;
}
.result-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1.5rem 2rem;
    animation: fadeIn 0.5s ease;
    margin-top: 1rem;
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}
.badge {
    display: inline-block;
    padding: 0.4rem 1.2rem;
    border-radius: 999px;
    font-weight: 700;
    font-size: 1rem;
    letter-spacing: 0.05em;
    margin-bottom: 1rem;
}
.badge-depression           { background: rgba(239,68,68,0.2);   color: #fca5a5; border: 1px solid rgba(239,68,68,0.4); }
.badge-anxiety              { background: rgba(245,158,11,0.2);  color: #fcd34d; border: 1px solid rgba(245,158,11,0.4); }
.badge-normal               { background: rgba(34,197,94,0.2);   color: #86efac; border: 1px solid rgba(34,197,94,0.4); }
.badge-stress               { background: rgba(249,115,22,0.2);  color: #fdba74; border: 1px solid rgba(249,115,22,0.4); }
.badge-bipolar              { background: rgba(6,182,212,0.2);   color: #67e8f9; border: 1px solid rgba(6,182,212,0.4); }
.badge-suicidal             { background: rgba(220,38,38,0.3);   color: #ff8080; border: 1px solid rgba(220,38,38,0.6); }
.badge-personality-disorder { background: rgba(168,85,247,0.2);  color: #d8b4fe; border: 1px solid rgba(168,85,247,0.4); }

.conf-label { color: #94a3b8; font-size: 0.85rem; margin-bottom: 0.4rem; }
.conf-bar-bg {
    background: rgba(255,255,255,0.08);
    border-radius: 999px;
    height: 8px;
    overflow: hidden;
    margin-bottom: 1rem;
}
.conf-bar-fill {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #7c3aed, #06b6d4);
    transition: width 1s ease;
}
.empathy-msg {
    color: #cbd5e1;
    font-size: 0.95rem;
    line-height: 1.6;
    margin-bottom: 0.8rem;
}
.resource-box {
    background: rgba(124,58,237,0.08);
    border: 1px solid rgba(124,58,237,0.25);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    color: #c4b5fd;
    font-size: 0.88rem;
    line-height: 1.7;
    margin-bottom: 0.8rem;
}
.disclaimer {
    color: #475569;
    font-size: 0.75rem;
    border-top: 1px solid rgba(255,255,255,0.06);
    padding-top: 0.8rem;
    margin-top: 0.5rem;
}
.footer {
    text-align: center;
    color: #334155;
    font-size: 0.8rem;
    margin-top: 3rem;
    padding-bottom: 2rem;
}
</style>
""", unsafe_allow_html=True)


# ── Load model (cached so it only loads once) ──────────────────────────
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = DistilBertTokenizer.from_pretrained("models/distilbert_final_tokenizer")
    model = DistilBertForSequenceClassification.from_pretrained("models/distilbert_final")
    model = model.to(device)
    model.eval()
    return model, tokenizer, device

MODEL_READY = False
try:
    model, tokenizer, device = load_model()
    MODEL_READY = True
except Exception as e:
    st.error(f"Model failed to load: {e}")


# ── Labels ─────────────────────────────────────────────────────────────
id2label = {
    0: 'Anxiety',
    1: 'Bipolar',
    2: 'Depression',
    3: 'Normal',
    4: 'Personality Disorder',
    5: 'Stress',
    6: 'Suicidal'
}


# ── Predict function ───────────────────────────────────────────────────
def predict(text):
    inputs = tokenizer(
        text,
        return_tensors='pt',
        max_length=128,
        truncation=True,
        padding='max_length'
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    pred_id = torch.argmax(outputs.logits, dim=-1).item()
    confidence = torch.softmax(outputs.logits, dim=-1).max().item() * 100
    return id2label[pred_id], round(confidence, 1)


# ── LIME explainability ────────────────────────────────────────────────
labels_list = ['Anxiety', 'Bipolar', 'Depression', 'Normal', 'Personality Disorder', 'Stress', 'Suicidal']

def predict_proba(texts):
    inputs = tokenizer(
        texts,
        return_tensors='pt',
        max_length=128,
        truncation=True,
        padding='max_length'
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    return torch.softmax(outputs.logits, dim=-1).cpu().numpy()


# ── Empathy messages + resources ──────────────────────────────────────
EMPATHY = {
    'Depression': (
        "💙",
        "It sounds like you may be going through a really tough time. Please know that you're not alone, and support is available.",
        "You're not alone. Talk to someone you trust, or contact a helpline. Try to get sunlight, movement, and rest today.",
        "🇵🇰 Umang helpline: 0317-4288665"
    ),
    'Suicidal': (
        "🔴",
        "Your feelings are valid and you deserve support right now. Please reach out — help is available immediately.",
        "Please reach out immediately. You matter and help is available right now.",
        "🇵🇰 Umang helpline: 0317-4288665 | Rozan Counselling: 051-2890505"
    ),
    'Anxiety': (
        "💛",
        "It seems like anxiety might be present in your words. Taking deep breaths and talking to someone you trust can help.",
        "Try box breathing: inhale 4 counts, hold 4, exhale 4, hold 4. Repeat.",
        "🇵🇰 Umang helpline: 0317-4288665"
    ),
    'Bipolar': (
        "🩵",
        "Emotional fluctuations can be really challenging. A mental health professional can offer valuable support.",
        "Track your mood daily and maintain a consistent sleep schedule. Speak to a psychiatrist if you haven't.",
        "🇵🇰 Umang helpline: 0317-4288665"
    ),
    'Stress': (
        "🧡",
        "You seem to be under a lot of pressure right now. Remember to take breaks and be kind to yourself.",
        "Take a break. Even 10 minutes of walking or journaling can help reset your mind.",
        ""
    ),
    'Personality Disorder': (
        "💜",
        "Your experience is real and your feelings matter. Therapy can make a big difference.",
        "Therapy, especially DBT, can be very effective. Consider reaching out to a mental health professional.",
        "🇵🇰 Umang helpline: 0317-4288665"
    ),
    'Normal': (
        "💚",
        "You seem to be in a relatively stable headspace. Keep nurturing your mental wellness!",
        "You're doing great! Keep maintaining healthy habits.",
        ""
    ),
}


# ── UI ─────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">🧠 MindScan</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">AI-powered mental health sentiment analysis</div>', unsafe_allow_html=True)

if not MODEL_READY:
    st.markdown('<div class="demo-banner">⚡ Model not loaded — check that models/ folder has distilbert_final inside</div>', unsafe_allow_html=True)

user_input = st.text_area(
    "How are you feeling today?",
    height=160,
    placeholder="Type freely… this space is safe."
)
analyze = st.button("Analyze ✦")

if analyze:
    if not user_input.strip():
        st.warning("Please enter some text first.")
    elif not MODEL_READY:
        st.error("Model is not loaded. Check your models/ folder.")
    else:
        prediction, confidence = predict(user_input)

        emoji, empathy_msg, resource_tip, helpline = EMPATHY.get(
            prediction,
            ("💚", "Thank you for sharing.", "", "")
        )

        badge_class = f"badge-{prediction.lower().replace(' ', '-')}"
        confidence_display = round(confidence, 1)

        # ── Result card ──
        st.markdown(f"""
        <div class="result-card">
            <div style="color:#94a3b8; font-size:0.85rem; margin-bottom:0.5rem;">DETECTED SENTIMENT</div>
            <span class="badge {badge_class}">{emoji} {prediction.upper()}</span>
            <div class="conf-label">Model Confidence — {confidence_display}%</div>
            <div class="conf-bar-bg">
                <div class="conf-bar-fill" style="width:{confidence_display}%"></div>
            </div>
            <div class="empathy-msg">{empathy_msg}</div>
        """, unsafe_allow_html=True)

        if resource_tip or helpline:
            st.markdown(f"""
            <div class="resource-box">
                <strong>💡 What can help:</strong><br>
                {resource_tip}<br>
                {"<br><strong>📞 " + helpline + "</strong>" if helpline else ""}
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
            <div class="disclaimer">⚠️ This is not a medical diagnosis. If you're struggling, please reach out to a licensed mental health professional.</div>
        </div>
        """, unsafe_allow_html=True)


st.markdown(
    '<div class="footer">Built with Python · HuggingFace Transformers · Streamlit &nbsp;|&nbsp; 🧠 MindScan</div>',
    unsafe_allow_html=True
)