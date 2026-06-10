
# MindScan — Complete Project Documentation
### AI-Powered Mental Health Sentiment Analyzer
**Course Project | Intro to AI | 4th Semester**

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Team Structure](#2-team-structure)
3. [Project Roadmap](#3-project-roadmap)
4. [Dataset](#4-dataset)
5. [Models Built](#5-models-built)
6. [Web Application](#6-web-application)
7. [File Structure](#7-file-structure)
8. [Tech Stack](#8-tech-stack)
9. [Features](#9-features)
10. [UI Design Decisions](#10-ui-design-decisions)
11. [Bugs Fixed During Development](#11-bugs-fixed-during-development)
12. [How to Run Locally](#13-how-to-run-locally)

---

## 1. Project Overview

**MindScan** is an NLP-based mental health sentiment classifier that analyzes free-form text and predicts which of 7 mental health categories the text reflects. The user types how they are feeling, the model predicts a category, shows a confidence score, gives an empathy message, and provides Pakistani mental health helpline resources.

**Why this project:**
- Covers every major topic in an Intro AI course: dataset, EDA, preprocessing, multiple models, evaluation, and a web app
- Uses a real-world dataset (Reddit mental health corpus from Kaggle)

**The 7 prediction labels:**
| Label | Description |
|---|---|
| Anxiety | Worry, panic, nervousness |
| Bipolar | Mood swings, emotional fluctuation |
| Depression | Sadness, hopelessness, emptiness |
| Normal | Stable, healthy mental state |
| Personality Disorder | Identity, relationship, emotional regulation issues |
| Stress | Pressure, overwhelm, burnout |
| Suicidal | Crisis-level ideation |

---

## 2. Team Structure

**3-person team, 4-week timeline**

| Person | Role | Deliverable |
|---|---|---|
| Fatima Anjum | Data + EDA + Preprocessing | Clean dataset + EDA Jupyter notebook |
| Anosha Shams | Model Evaluation | Trained models + results charts |
| Fatima Zahra | Web App + Model Training + | GitHub repo |

---

## 3. Project Roadmap

```
Phase 1 — Week 1: Data + EDA
├── Load dataset (Pandas)
├── EDA + charts (word clouds, class distribution)
└── NLP preprocessing (clean, tokenize, stem)

Phase 2 — Week 2: Model Building
├── Baseline: Logistic Regression with TF-IDF (76.29% accuracy)
├── Better: Random Forest / SVM
└── Best: Fine-tuned DistilBERT (83% accuracy) ✓ Used in app

Phase 3 — Week 3: Evaluation + Visuals
├── Accuracy, F1, Confusion Matrix
├── Model comparison bar chart
└── SHAP / feature importance

Phase 4 — Week 4: Web App 
├── Streamlit app (type text → get prediction)
└── GitHub repo + LinkedIn post
```

---

## 4. Dataset

- **Source:** Kaggle — Reddit Mental Health NLP Corpus
- **File:** `Combined Data.csv`
- **Size:** ~11MB, approximately 50,000 labeled posts
- **Labels:** 7 categories (Anxiety, Bipolar, Depression, Normal, Personality Disorder, Stress, Suicidal)
- **Origin:** Reddit posts from mental health subreddits, scraped and labeled
- **Shared via:** Google Drive (AIProject shared folder)

---

## 5. Models Built

All model training was done by Person 2 (Anosha) in Google Colab.

### Model 1 — Logistic Regression (Baseline)
- **Vectorizer:** TF-IDF
- **Accuracy:** 76.29%
- **Files:** `lr_model.pkl`, `tfidf_vectorizer.pkl`
- **Status:** Not used in final app (DistilBERT is better)

### Model 2 — DistilBERT (Final Model)
- **Type:** Fine-tuned `distilbert-base-uncased` from HuggingFace
- **Accuracy:** 83%
- **Training:** Fine-tuned on the Reddit dataset, 7 output classes
- **Files:** `distilbert_final/` folder, `distilbert_final_tokenizer/` folder
- **Note:** An earlier version (`distilbert_weights.pt`) had a bug where accuracy dropped on reload. Anosha fixed this by saving the full model folder instead of just weights.
- **Size:** ~255MB (model weights)
- **Status:**  Used in the final app

### Model Loading Code (Final Version)
```python
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = DistilBertTokenizer.from_pretrained("models/distilbert_final_tokenizer")
model = DistilBertForSequenceClassification.from_pretrained("models/distilbert_final")
model = model.to(device)
model.eval()
```

### Label Mapping
```python
id2label = {
    0: 'Anxiety',
    1: 'Bipolar',
    2: 'Depression',
    3: 'Normal',
    4: 'Personality Disorder',
    5: 'Stress',
    6: 'Suicidal'
}
```

---

## 6. Web Application

Built with **Streamlit** in Python. The app takes user text input, runs it through the DistilBERT model, and returns a prediction with supporting information.

### How the prediction works
1. User types text in the textarea
2. User clicks "Analyze ✦"
3. Text is tokenized using DistilBert tokenizer (max 128 tokens, padded)
4. Model runs a forward pass with `torch.no_grad()`
5. `argmax` on logits gives the predicted class ID
6. `softmax` on logits gives the confidence percentage
7. Result is mapped via `id2label` to the label string
8. UI renders the badge, confidence bar, empathy message, and resources

### Prediction Function
```python
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
```

### Mental Health Resources by Anosha
Each predicted label shows a tailored message and Pakistani helpline:

| Label | Resource |
|---|---|
| Depression | Umang helpline: 0317-4288665 |
| Suicidal | Umang: 0317-4288665 \| Rozan Counselling: 051-2890505 |
| Anxiety | Umang helpline: 0317-4288665 |
| Bipolar | Umang helpline: 0317-4288665 |
| Personality Disorder | Umang helpline: 0317-4288665 |
| Stress | General wellness tip (no helpline) |
| Normal | Positive reinforcement (no helpline) |

---

## 7. File Structure

```
mental-health-nlp/
├── MindScan/                          ← main project (also HF Space root)
│   ├── models/
│   │   ├── distilbert_final/          ← final DistilBERT model (255MB, gitignored)
│   │   │   ├── config.json
│   │   │   ├── model.safetensors
│   │   │   └── ...
│   │   └── distilbert_final_tokenizer/ ← tokenizer files (gitignored)
│   │       ├── tokenizer_config.json
│   │       ├── vocab.txt
│   │       └── ...
│   ├── notebooks/
│   │   └── Untitled0.ipynb           
│   ├
─ app.py                     ← main Streamlit application
│   ├── .streamlit/
│   │   └── config.toml                ← disables file watcher (kills terminal spam)
│   ├── .gitattributes                 ← Git LFS tracking rules
│   ├── .gitignore                     ← excludes models/, .venv/, *.csv
│   ├── Dockerfile                     ← for HF Spaces container
│   ├── README.md                      ← HF Space config + project description
│   └── requirements.txt              ← Python dependencies
└── .venv/                             ← virtual environment (gitignored)
```

### .gitignore contents
```
models/
.venv/
__pycache__/
*.pyc
notebooks/*.csv
```

### requirements.txt
```
streamlit
torch
transformers
lime
sentencepiece
safetensors
pandas
numpy

```

### .streamlit/config.toml
```toml
[server]
fileWatcherType = "none"
```
*This silences the hundreds of `ModuleNotFoundError: No module named 'torchvision'` warnings that Streamlit's file watcher produces when scanning the transformers library.*

---

## 8. Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Language | Python 3.13 | Everything |
| ML Framework | PyTorch | Model inference |
| NLP Library | HuggingFace Transformers | DistilBERT model + tokenizer |
| Web Framework | Streamlit | UI and app serving |
| Explainability | LIME (removed from final) | Word-level feature importance |
| Version Control | Git + GitHub | Code repository |
| Deployment | Hugging Face Spaces | Live app hosting |
| Training Environment | Google Colab | Model fine-tuning (Person 2) |
| File Sharing | Google Drive (AIProject folder) | Team file exchange |

---

## 9. Features

### Core Features
- **Text input** — Free-form textarea with placeholder text
- **Prediction badge** — Color-coded label pill (7 different colors per category)
- **Confidence bar** — Animated gradient progress bar showing model confidence %
- **Empathy message** — Warm, supportive message tailored to the predicted label
- **Resource box** — "What can help" tip + Pakistani helpline number (where applicable)
- **Medical disclaimer** — Shown on every result

### Design Features
- Dark purple/cyan gradient background
- Glass morphism card effect on input area
- Outfit font (Google Fonts)
- Floating orb background decorations
- Fade-in animation on result card
- Responsive centered layout (max 720px)

---

## 10. UI Design Decisions

### Why custom CSS instead of Streamlit themes
Streamlit's default theming is very limited. All styling was injected via `st.markdown("""<style>...</style>""", unsafe_allow_html=True)`.

### The glass-card problem
Originally the textarea was wrapped in a custom HTML div to apply glass card styling:
```python
# THIS DOES NOT WORK IN STREAMLIT
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
user_input = st.text_area(...)  # renders OUTSIDE the div
st.markdown('</div>', unsafe_allow_html=True)
```
Streamlit renders each widget in its own isolated container — custom HTML divs cannot wrap native widgets across separate `st.markdown` calls. The div opens as an empty grey box and the widget renders outside it.

**Fix:** Target Streamlit's own container using the `data-testid` attribute:
```css
div[data-testid="stVerticalBlock"] > div:has(> div[data-testid="stTextArea"]) {
    background: rgba(255,255,255,0.04);
    border-radius: 20px;
    /* etc */
}
```

### The spinner-in-expander-title problem
Streamlit's built-in running indicator (`stStatusWidget`) was visually bleeding into the expander label, showing as `arr🔄Se which words...` instead of `🔍 See which words...`.

**Root cause:** Streamlit positions its global running spinner near whatever element is currently computing. The expander was the nearest element.

**Fix:** Hide the status widget entirely via CSS:
```css
[data-testid="stStatusWidget"] { display: none !important; }
```

### Badge CSS class generation
The model returns strings like `'Personality Disorder'` with a capital and a space. CSS class names cannot have spaces.

**Fix:**
```python
badge_class = f"badge-{prediction.lower().replace(' ', '-')}"
# 'Personality Disorder' → 'badge-personality-disorder'
```

### LIME bar width normalization
LIME returns weights as tiny decimal values like `0.004`, `0.003`. The original formula `abs(weight) * 300` gave bars of ~1px width (invisible).

**Fix:** Normalize relative to the largest weight so the top word always gets 120px:
```python
max_weight = max(abs(w) for _, w in word_weights)
bar_width = int((abs(weight) / max_weight) * 120)
```

---

## 11. Bugs Fixed During Development

| Bug | Cause | Fix |
|---|---|---|
| App shows "Model not loaded" | Old code looked for `sentiment_model.pkl` (doesn't exist) | Replaced with DistilBERT loading |
| EMPATHY dict never matched | Dict keys were lowercase (`'depression'`) but model returns `'Depression'` | Changed all dict keys to match model output casing |
| Badge CSS never applied | Badge class was `badge-neutral` but label is `Normal` | Added `badge-normal` CSS class |
| Grey box above textarea | `glass-card` div cannot wrap Streamlit widgets | Used CSS `data-testid` selector instead |
| Spinner bleeding into expander title | `st.spinner()` inside expander caused visual overlap | Removed all `st.spinner()` calls; hid `stStatusWidget` via CSS |
| LIME bars invisible | `abs(weight) * 300` = ~1px because weights are tiny decimals | Normalized to max weight: `(abs(w)/max_w) * 120` |
| Hundreds of torchvision errors in terminal | Streamlit's file watcher scans all of transformers library | Added `.streamlit/config.toml` with `fileWatcherType = "none"` |
| Model accuracy dropped on reload | Bug in original `distilbert_weights.pt` saving method | Anosha fixed by saving full model folder (`distilbert_final/`) |
| `inputs.to(device)` error in predict_proba | Batch tokenizer output can't call `.to()` directly | Changed to `{k: v.to(device) for k, v in inputs.items()}` |

---


## GitHub
- Repository created at: `github.com/fazasfz/mental-health-nlp` (or similar)
- `models/` folder added to `.gitignore` — model files are too large (255MB) for GitHub's 100MB limit
- Code, requirements, and structure pushed successfully



### What remains to complete deployment
1. Remove CSV from git history completely (use `git filter-branch` or `git-filter-repo`)
2. Force push clean history to HF Space
3. Upload model files manually via HF Spaces Files tab (since they're gitignored)
4. Confirm `app_file: src/app.py` is in README.md YAML header
5. Verify app builds and runs

---

## 13. How to Run Locally

### Prerequisites
- Python 3.10+
- The `models/` folder with `distilbert_final/` and `distilbert_final_tokenizer/` downloaded from the shared Google Drive (AIProject folder)

### Steps

```bash
# 1. Clone the repo
git clone https://github.com/YOURUSERNAME/mental-health-nlp.git
cd mental-health-nlp/MindScan

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download model files from Google Drive
# Place distilbert_final/ and distilbert_final_tokenizer/ inside models/

# 5. Run the app
streamlit run app.py
```

App opens at: `http://localhost:8501`


---

*Documentation written by Fatima
*Project: MindScan | Team of 3 | 4th Semester AI Course*
