"""
Sentira — Text Sentiment Signal Analyzer
==========================================
A single-file Flask app that serves a pre-trained MultinomialNB + TF-IDF
sentiment classifier through a modern, instrument-panel styled UI.

FOLDER LAYOUT (only 2 files you create/manage):
    app.py
    requirements.txt

Plus, in the SAME folder as app.py, place your two pickle files:
    model_multinominalnb.pkl
    vector_tfidfvectorizer.pkl

Run locally:
    pip install -r requirements.txt
    python app.py
    -> open http://localhost:5000

Deploy on Render:
    New Web Service -> connect repo
    Build Command : pip install -r requirements.txt
    Start Command : gunicorn app:app
"""

import os
import re
import pickle
import logging
from datetime import datetime

from flask import Flask, render_template_string, request, jsonify

# --------------------------------------------------------------------------
# App setup
# --------------------------------------------------------------------------
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("sentira")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model_multinominalnb.pkl")
VECTOR_PATH = os.path.join(BASE_DIR, "vector_tfidfvectorizer.pkl")

model = None
vectorizer = None
MODEL_READY = False

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(VECTOR_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    MODEL_READY = True
    log.info("Model and vectorizer loaded successfully.")
except Exception as exc:  # noqa: BLE001
    log.error("Failed to load model/vectorizer: %s", exc)


# --------------------------------------------------------------------------
# Preprocessing — mirrors typical IMDB-style review cleanup used to train
# TF-IDF sentiment models (strip HTML tags, non-letters, extra whitespace).
# --------------------------------------------------------------------------
TAG_RE = re.compile(r"<[^>]+>")
NON_ALPHA_RE = re.compile(r"[^a-zA-Z\s]")
MULTI_SPACE_RE = re.compile(r"\s+")


def clean_text(raw_text: str) -> str:
    text = TAG_RE.sub(" ", raw_text)
    text = NON_ALPHA_RE.sub(" ", text)
    text = text.lower()
    text = MULTI_SPACE_RE.sub(" ", text).strip()
    return text


def get_top_signal_words(cleaned_text: str, limit: int = 8):
    """Return vocabulary words from the input that carry the most TF-IDF
    weight, used to drive the animated waveform readout."""
    try:
        vec = vectorizer.transform([cleaned_text])
        row = vec.tocoo()
        pairs = list(zip(row.col, row.data))
        pairs.sort(key=lambda p: p[1], reverse=True)
        feature_names = vectorizer.get_feature_names_out()
        words = [feature_names[i] for i, _ in pairs[:limit]]
        return words
    except Exception:  # noqa: BLE001
        return []


# --------------------------------------------------------------------------
# UI — embedded template (single file, no /templates folder needed)
# --------------------------------------------------------------------------
INDEX_HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Sentira — Text Sentiment Signal Analyzer</title>
<meta name="description" content="Decode the sentiment inside any text with a TF-IDF + Naive Bayes signal analysis engine.">

<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">

<style>
  :root{
    --bg:            #0b0f14;
    --panel:         #121820;
    --panel-raised:  #161e28;
    --stroke:        #232d3a;
    --text:          #e8edf2;
    --text-muted:    #7c8a99;
    --text-faint:    #4d5966;

    --signal-pos:    #00e5a0;   /* teal — positive */
    --signal-pos-dim:#00e5a022;
    --signal-neg:    #ff4d6d;   /* coral — negative */
    --signal-neg-dim:#ff4d6d22;
    --signal-amber:  #ffb020;   /* amber — highlight / focus */

    --font-display: 'Space Grotesk', sans-serif;
    --font-body:    'Inter', sans-serif;
    --font-mono:    'JetBrains Mono', monospace;

    --radius: 14px;
    --radius-sm: 8px;
  }

  *{ box-sizing:border-box; margin:0; padding:0; }

  html{ scroll-behavior:smooth; }

  body{
    background:
      radial-gradient(circle at 15% 0%, #0f151d 0%, transparent 45%),
      radial-gradient(circle at 85% 20%, #0e1a17 0%, transparent 40%),
      var(--bg);
    color:var(--text);
    font-family:var(--font-body);
    min-height:100vh;
    line-height:1.5;
  }

  body::before{
    content:"";
    position:fixed; inset:0;
    background-image:repeating-linear-gradient(
      to bottom, rgba(255,255,255,0.012) 0px, rgba(255,255,255,0.012) 1px,
      transparent 1px, transparent 3px
    );
    pointer-events:none;
    z-index:0;
  }

  a{ color:inherit; }
  :focus-visible{
    outline:2px solid var(--signal-amber);
    outline-offset:3px;
    border-radius:4px;
  }

  .wrap{ max-width:1180px; margin:0 auto; padding:0 28px; position:relative; z-index:1; }

  header{
    padding:26px 0 10px;
    display:flex; align-items:center; justify-content:space-between;
    border-bottom:1px solid var(--stroke);
  }
  .brand{ display:flex; align-items:center; gap:12px; }
  .brand-mark{
    width:38px; height:38px; border-radius:10px;
    background:linear-gradient(145deg, var(--signal-pos), #00a37a);
    display:flex; align-items:center; justify-content:center;
    box-shadow:0 0 22px var(--signal-pos-dim);
    flex-shrink:0;
  }
  .brand-mark svg{ width:20px; height:20px; }
  .brand-name{ font-family:var(--font-display); font-size:20px; font-weight:700; letter-spacing:0.3px; }
  .brand-tag{
    font-family:var(--font-mono); font-size:10.5px; letter-spacing:1.5px;
    color:var(--text-muted); text-transform:uppercase; margin-top:1px;
  }
  .status-chip{
    display:flex; align-items:center; gap:8px;
    font-family:var(--font-mono); font-size:11.5px; letter-spacing:0.5px;
    color:var(--text-muted); border:1px solid var(--stroke);
    padding:7px 13px; border-radius:100px; text-transform:uppercase;
  }
  .status-dot{
    width:7px; height:7px; border-radius:50%;
    background:var(--signal-pos); box-shadow:0 0 8px var(--signal-pos);
    animation:pulse 2.4s ease-in-out infinite;
  }
  .status-dot.down{ background:var(--signal-neg); box-shadow:0 0 8px var(--signal-neg); animation:none; }
  @keyframes pulse{ 0%,100%{opacity:1;} 50%{opacity:0.35;} }

  .hero{
    display:grid; grid-template-columns: 1.05fr 0.95fr; gap:56px;
    padding:64px 0 44px; align-items:center;
  }
  .eyebrow{
    font-family:var(--font-mono); font-size:11.5px; letter-spacing:2px;
    color:var(--signal-pos); text-transform:uppercase; margin-bottom:18px;
    display:flex; align-items:center; gap:10px;
  }
  .eyebrow::before{ content:""; width:22px; height:1px; background:var(--signal-pos); display:inline-block; }
  .hero h1{
    font-family:var(--font-display); font-weight:700;
    font-size:clamp(34px, 4.4vw, 54px); line-height:1.06; letter-spacing:-0.5px;
    margin-bottom:20px;
  }
  .hero h1 span{ color:var(--signal-pos); }
  .hero p{ color:var(--text-muted); font-size:16px; max-width:480px; margin-bottom:32px; }

  .spec-row{ display:flex; gap:30px; flex-wrap:wrap; }
  .spec{ font-family:var(--font-mono); }
  .spec .val{ font-size:22px; font-weight:700; color:var(--text); }
  .spec .lbl{ font-size:10.5px; color:var(--text-faint); text-transform:uppercase; letter-spacing:1px; margin-top:2px; }

  .hero-visual{
    background:var(--panel);
    border:1px solid var(--stroke);
    border-radius:var(--radius);
    padding:26px;
    position:relative;
    overflow:hidden;
  }
  .hero-visual::after{
    content:""; position:absolute; inset:0;
    background:linear-gradient(180deg, transparent 60%, #0b0f1499 100%);
    pointer-events:none;
  }
  .hero-visual-head{
    display:flex; justify-content:space-between; align-items:center;
    font-family:var(--font-mono); font-size:11px; color:var(--text-faint);
    text-transform:uppercase; letter-spacing:1px; margin-bottom:18px;
  }
  .rec-dot{ color:var(--signal-neg); }
  #idleWave{ display:flex; align-items:flex-end; gap:3px; height:150px; }
  #idleWave .bar{
    flex:1; background:linear-gradient(180deg, var(--signal-pos), #0d3d31);
    border-radius:2px 2px 0 0; opacity:0.55;
  }

  .panel-title{
    font-family:var(--font-mono); font-size:11px; letter-spacing:1.5px;
    text-transform:uppercase; color:var(--text-faint); margin-bottom:14px;
    display:flex; align-items:center; gap:8px;
  }
  .panel-title .n{ color:var(--signal-amber); }

  .analyzer{
    background:var(--panel);
    border:1px solid var(--stroke);
    border-radius:var(--radius);
    padding:30px;
    margin-bottom:26px;
  }
  .analyzer-top{ display:flex; justify-content:space-between; align-items:baseline; margin-bottom:16px; }
  .analyzer-top h2{ font-family:var(--font-display); font-size:20px; font-weight:600; }
  #charCount{ font-family:var(--font-mono); font-size:12px; color:var(--text-faint); }
  #charCount.warn{ color:var(--signal-neg); }

  textarea{
    width:100%; min-height:150px; resize:vertical;
    background:var(--bg); border:1px solid var(--stroke); border-radius:var(--radius-sm);
    color:var(--text); font-family:var(--font-body); font-size:15px; line-height:1.6;
    padding:16px; transition:border-color .15s ease;
  }
  textarea:focus{ outline:none; border-color:var(--signal-pos); }
  textarea::placeholder{ color:var(--text-faint); }

  .chips{ display:flex; gap:9px; flex-wrap:wrap; margin-top:16px; }
  .chip{
    font-family:var(--font-mono); font-size:11.5px;
    border:1px solid var(--stroke); background:var(--panel-raised);
    color:var(--text-muted); padding:7px 13px; border-radius:100px;
    cursor:pointer; transition:all .15s ease;
  }
  .chip:hover{ border-color:var(--signal-pos); color:var(--text); }

  .analyzer-actions{
    display:flex; align-items:center; gap:16px; margin-top:22px;
  }
  .btn-analyze{
    font-family:var(--font-display); font-weight:600; font-size:15px;
    background:var(--signal-pos); color:#04140f; border:none;
    padding:14px 28px; border-radius:var(--radius-sm);
    cursor:pointer; transition:transform .12s ease, box-shadow .12s ease;
    display:flex; align-items:center; gap:10px;
  }
  .btn-analyze:hover{ transform:translateY(-1px); box-shadow:0 8px 24px var(--signal-pos-dim); }
  .btn-analyze:disabled{ opacity:0.5; cursor:not-allowed; transform:none; box-shadow:none; }
  .btn-clear{
    font-family:var(--font-mono); font-size:12.5px; color:var(--text-faint);
    background:none; border:none; cursor:pointer; text-decoration:underline;
    text-underline-offset:3px;
  }
  .btn-clear:hover{ color:var(--text-muted); }

  #errorBox{
    display:none; margin-top:16px; padding:12px 16px;
    background:var(--signal-neg-dim); border:1px solid #ff4d6d55;
    border-radius:var(--radius-sm); color:var(--signal-neg);
    font-family:var(--font-mono); font-size:13px;
  }

  #readout{
    display:none;
    background:var(--panel);
    border:1px solid var(--stroke);
    border-radius:var(--radius);
    padding:30px;
    margin-bottom:26px;
    animation:rise .35s ease;
  }
  @keyframes rise{ from{ opacity:0; transform:translateY(10px);} to{opacity:1; transform:translateY(0);} }

  .readout-grid{ display:grid; grid-template-columns: 1fr 1fr; gap:30px; }
  @media (max-width:760px){ .readout-grid{ grid-template-columns:1fr; } }

  .verdict{ display:flex; align-items:center; gap:18px; margin-bottom:24px; }
  .verdict-badge{
    font-family:var(--font-display); font-weight:700; font-size:26px;
    padding:12px 22px; border-radius:var(--radius-sm);
    letter-spacing:0.5px;
  }
  .verdict-badge.positive{ background:var(--signal-pos-dim); color:var(--signal-pos); border:1px solid #00e5a055; }
  .verdict-badge.negative{ background:var(--signal-neg-dim); color:var(--signal-neg); border:1px solid #ff4d6d55; }
  .verdict-meta{ font-family:var(--font-mono); font-size:12px; color:var(--text-faint); }
  .verdict-meta div{ margin-top:2px; }

  .meter-row{ margin-bottom:16px; }
  .meter-label{
    display:flex; justify-content:space-between;
    font-family:var(--font-mono); font-size:12px; color:var(--text-muted);
    margin-bottom:6px; text-transform:uppercase; letter-spacing:0.5px;
  }
  .meter-track{ height:10px; background:var(--panel-raised); border-radius:100px; overflow:hidden; border:1px solid var(--stroke); }
  .meter-fill{ height:100%; border-radius:100px; width:0%; transition:width .7s cubic-bezier(.2,.8,.2,1); }
  .meter-fill.positive{ background:linear-gradient(90deg, #00a37a, var(--signal-pos)); }
  .meter-fill.negative{ background:linear-gradient(90deg, #b12e46, var(--signal-neg)); }

  #liveWave{ display:flex; align-items:flex-end; gap:3px; height:120px; margin-top:6px; }
  #liveWave .bar{ flex:1; border-radius:2px 2px 0 0; transition:height .5s ease, background .3s ease; }

  .signal-words{ margin-top:18px; }
  .signal-words .words{ display:flex; flex-wrap:wrap; gap:8px; margin-top:10px; }
  .word-tag{
    font-family:var(--font-mono); font-size:11.5px;
    background:var(--panel-raised); border:1px solid var(--stroke);
    padding:5px 11px; border-radius:6px; color:var(--text-muted);
  }

  #history{ margin-bottom:60px; }
  #historyList{ display:flex; flex-direction:column; gap:10px; }
  .empty-state{
    font-family:var(--font-mono); font-size:12.5px; color:var(--text-faint);
    border:1px dashed var(--stroke); border-radius:var(--radius-sm);
    padding:22px; text-align:center;
  }
  .hist-item{
    display:flex; align-items:center; gap:14px;
    background:var(--panel); border:1px solid var(--stroke);
    border-radius:var(--radius-sm); padding:13px 16px;
  }
  .hist-dot{ width:9px; height:9px; border-radius:50%; flex-shrink:0; }
  .hist-dot.positive{ background:var(--signal-pos); box-shadow:0 0 8px var(--signal-pos); }
  .hist-dot.negative{ background:var(--signal-neg); box-shadow:0 0 8px var(--signal-neg); }
  .hist-text{ flex:1; font-size:13.5px; color:var(--text-muted); overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  .hist-conf{ font-family:var(--font-mono); font-size:11.5px; color:var(--text-faint); flex-shrink:0; }
  .hist-time{ font-family:var(--font-mono); font-size:11px; color:var(--text-faint); flex-shrink:0; }

  .stack{
    display:flex; align-items:center; justify-content:center; gap:12px;
    flex-wrap:wrap; padding:26px 0; border-top:1px solid var(--stroke);
    border-bottom:1px solid var(--stroke); margin-bottom:40px;
  }
  .stack-item{
    font-family:var(--font-mono); font-size:11px; letter-spacing:0.5px;
    color:var(--text-faint); border:1px solid var(--stroke);
    padding:7px 14px; border-radius:100px; text-transform:uppercase;
  }

  footer{ padding:30px 0 50px; text-align:center; }
  footer p{ font-family:var(--font-mono); font-size:11.5px; color:var(--text-faint); }

  @media (max-width:900px){
    .hero{ grid-template-columns:1fr; padding-top:40px; }
    .hero p{ max-width:none; }
  }

  @media (prefers-reduced-motion: reduce){
    *{ animation-duration:0.001ms !important; transition-duration:0.001ms !important; }
  }
</style>
</head>
<body>

<div class="wrap">

  <header>
    <div class="brand">
      <div class="brand-mark">
        <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
          <path d="M3 12h3l2-7 4 14 2-7h7" stroke="#04140f" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
      </div>
      <div>
        <div class="brand-name">Sentira</div>
        <div class="brand-tag">Text Signal Analysis Engine</div>
      </div>
    </div>
    <div class="status-chip">
      <span class="status-dot {{ 'down' if not model_ready else '' }}"></span>
      {{ 'model online' if model_ready else 'model unavailable' }}
    </div>
  </header>

  <section class="hero">
    <div>
      <div class="eyebrow">Naive Bayes · TF-IDF Vectorization</div>
      <h1>Decode the sentiment<br>hiding inside <span>any text.</span></h1>
      <p>Sentira reads a sentence like an instrument reads a signal — breaking it down into weighted terms and reporting back polarity and confidence in real time.</p>
      <div class="spec-row">
        <div class="spec"><div class="val">5,000</div><div class="lbl">Vocabulary Terms</div></div>
        <div class="spec"><div class="val">TF-IDF</div><div class="lbl">Feature Extraction</div></div>
        <div class="spec"><div class="val">MultinomialNB</div><div class="lbl">Classifier</div></div>
        <div class="spec"><div class="val">2-Class</div><div class="lbl">Positive / Negative</div></div>
      </div>
    </div>

    <div class="hero-visual">
      <div class="hero-visual-head">
        <span><span class="rec-dot">●</span> idle signal</span>
        <span>awaiting input</span>
      </div>
      <div id="idleWave"></div>
    </div>
  </section>

  <div class="stack">
    <span class="stack-item">Flask</span>
    <span class="stack-item">scikit-learn</span>
    <span class="stack-item">TF-IDF Vectorizer</span>
    <span class="stack-item">Multinomial Naive Bayes</span>
    <span class="stack-item">Gunicorn</span>
  </div>

  <section class="analyzer">
    <div class="analyzer-top">
      <h2>Run an analysis</h2>
      <span id="charCount">0 / 5000</span>
    </div>
    <textarea id="textInput" maxlength="5000" placeholder="Paste a review, comment, or any block of text here..."></textarea>

    <div class="chips">
      <button class="chip" data-sample="This movie completely blew me away — brilliant acting, a gripping story, and a soundtrack I still can't stop thinking about.">Sample: rave review</button>
      <button class="chip" data-sample="What a waste of time. The plot made no sense, the pacing dragged endlessly, and I regretted every minute of it.">Sample: harsh review</button>
      <button class="chip" data-sample="The product arrived on time and works exactly as described, though the packaging could have been sturdier.">Sample: mixed feedback</button>
    </div>

    <div class="analyzer-actions">
      <button class="btn-analyze" id="analyzeBtn" {{ 'disabled' if not model_ready else '' }}>
        <span id="analyzeLabel">Analyze Text</span>
      </button>
      <button class="btn-clear" id="clearBtn">Clear</button>
    </div>

    <div id="errorBox"></div>
  </section>

  <section id="readout">
    <div class="panel-title"><span class="n">//</span> Signal Readout</div>

    <div class="readout-grid">
      <div>
        <div class="verdict">
          <div class="verdict-badge" id="verdictBadge">—</div>
          <div class="verdict-meta">
            <div>Analyzed at <span id="metaTime">—</span></div>
            <div id="metaWords">— words parsed</div>
          </div>
        </div>

        <div class="meter-row">
          <div class="meter-label"><span>Positive</span><span id="posVal">0%</span></div>
          <div class="meter-track"><div class="meter-fill positive" id="posFill"></div></div>
        </div>
        <div class="meter-row">
          <div class="meter-label"><span>Negative</span><span id="negVal">0%</span></div>
          <div class="meter-track"><div class="meter-fill negative" id="negFill"></div></div>
        </div>

        <div class="signal-words">
          <div class="panel-title" style="margin-bottom:0;">Dominant Terms</div>
          <div class="words" id="signalWords"></div>
        </div>
      </div>

      <div>
        <div class="panel-title">Waveform</div>
        <div id="liveWave"></div>
      </div>
    </div>
  </section>

  <section id="history">
    <div class="panel-title"><span class="n">//</span> Recent Scans</div>
    <div id="historyList">
      <div class="empty-state">No scans yet — analyzed text will appear here.</div>
    </div>
  </section>

  <footer>
    <p>SENTIRA · TF-IDF + MULTINOMIAL NAIVE BAYES SENTIMENT ENGINE · BUILT WITH FLASK</p>
  </footer>

</div>

<script>
  // ---------- idle waveform ----------
  const idleWave = document.getElementById('idleWave');
  const IDLE_BARS = 48;
  for (let i = 0; i < IDLE_BARS; i++){
    const bar = document.createElement('div');
    bar.className = 'bar';
    bar.style.height = (8 + Math.random() * 30) + '%';
    idleWave.appendChild(bar);
  }
  function animateIdle(){
    idleWave.querySelectorAll('.bar').forEach(bar => {
      bar.style.transition = 'height 1.4s ease';
      bar.style.height = (8 + Math.random() * 55) + '%';
    });
  }
  animateIdle();
  setInterval(animateIdle, 1400);

  // ---------- elements ----------
  const textInput   = document.getElementById('textInput');
  const charCount    = document.getElementById('charCount');
  const analyzeBtn   = document.getElementById('analyzeBtn');
  const analyzeLabel = document.getElementById('analyzeLabel');
  const clearBtn     = document.getElementById('clearBtn');
  const errorBox     = document.getElementById('errorBox');
  const readout      = document.getElementById('readout');
  const historyList  = document.getElementById('historyList');

  let history = [];

  textInput.addEventListener('input', () => {
    const len = textInput.value.length;
    charCount.textContent = `${len} / 5000`;
    charCount.classList.toggle('warn', len > 4700);
  });

  document.querySelectorAll('.chip').forEach(chip => {
    chip.addEventListener('click', () => {
      textInput.value = chip.dataset.sample;
      textInput.dispatchEvent(new Event('input'));
      textInput.focus();
    });
  });

  clearBtn.addEventListener('click', () => {
    textInput.value = '';
    textInput.dispatchEvent(new Event('input'));
    readout.style.display = 'none';
    errorBox.style.display = 'none';
  });

  function showError(msg){
    errorBox.textContent = msg;
    errorBox.style.display = 'block';
  }

  function renderLiveWave(sentiment, confidence, words){
    const wave = document.getElementById('liveWave');
    wave.innerHTML = '';
    const color = sentiment === 'positive' ? '#00e5a0' : '#ff4d6d';
    const dim   = sentiment === 'positive' ? '#0d3d31' : '#3d0d18';
    const BARS = 40;
    const peak = Math.max(20, confidence);
    for (let i = 0; i < BARS; i++){
      const bar = document.createElement('div');
      bar.className = 'bar';
      const wobble = Math.sin(i * 0.5) * 18 + Math.random() * 22;
      const height = Math.max(6, Math.min(100, (peak * 0.5) + wobble));
      bar.style.background = `linear-gradient(180deg, ${color}, ${dim})`;
      bar.style.height = '4%';
      wave.appendChild(bar);
      setTimeout(() => { bar.style.height = height + '%'; }, i * 12);
    }
  }

  function addHistory(text, sentiment, confidence, time){
    history.unshift({ text, sentiment, confidence, time });
    history = history.slice(0, 6);
    historyList.innerHTML = '';
    if (history.length === 0){
      historyList.innerHTML = '<div class="empty-state">No scans yet — analyzed text will appear here.</div>';
      return;
    }
    history.forEach(item => {
      const row = document.createElement('div');
      row.className = 'hist-item';
      row.innerHTML = `
        <span class="hist-dot ${item.sentiment}"></span>
        <span class="hist-text">${item.text.replace(/</g,'&lt;')}</span>
        <span class="hist-conf">${item.confidence.toFixed(1)}%</span>
        <span class="hist-time">${item.time}</span>
      `;
      historyList.appendChild(row);
    });
  }

  async function analyze(){
    const text = textInput.value.trim();
    if (!text){ showError('Please enter some text to analyze.'); return; }

    errorBox.style.display = 'none';
    analyzeBtn.disabled = true;
    analyzeLabel.textContent = 'Analyzing…';

    try {
      const res = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text })
      });
      const data = await res.json();

      if (!res.ok){
        showError(data.error || 'Something went wrong. Please try again.');
        return;
      }

      readout.style.display = 'block';
      readout.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

      const badge = document.getElementById('verdictBadge');
      badge.textContent = data.sentiment.toUpperCase();
      badge.className = 'verdict-badge ' + data.sentiment;

      document.getElementById('metaTime').textContent = data.timestamp;
      document.getElementById('metaWords').textContent = `${data.word_count} words parsed`;

      document.getElementById('posVal').textContent = data.positive.toFixed(1) + '%';
      document.getElementById('negVal').textContent = data.negative.toFixed(1) + '%';
      document.getElementById('posFill').style.width = data.positive + '%';
      document.getElementById('negFill').style.width = data.negative + '%';

      const wordsBox = document.getElementById('signalWords');
      wordsBox.innerHTML = '';
      if (data.signal_words && data.signal_words.length){
        data.signal_words.forEach(w => {
          const tag = document.createElement('span');
          tag.className = 'word-tag';
          tag.textContent = w;
          wordsBox.appendChild(tag);
        });
      } else {
        wordsBox.innerHTML = '<span class="word-tag">no strong terms detected</span>';
      }

      renderLiveWave(data.sentiment, data.confidence, data.signal_words);
      addHistory(text, data.sentiment, data.confidence, data.timestamp);

    } catch (err) {
      showError('Could not reach the analysis server. Please try again.');
    } finally {
      analyzeBtn.disabled = false;
      analyzeLabel.textContent = 'Analyze Text';
    }
  }

  analyzeBtn.addEventListener('click', analyze);
  textInput.addEventListener('keydown', (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') analyze();
  });
</script>

</body>
</html>
"""


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@app.route("/")
def home():
    return render_template_string(INDEX_HTML, model_ready=MODEL_READY)


@app.route("/health")
def health():
    return jsonify(status="ok" if MODEL_READY else "degraded", model_ready=MODEL_READY)


@app.route("/predict", methods=["POST"])
def predict():
    if not MODEL_READY:
        return jsonify(error="Model is not loaded on the server."), 503

    payload = request.get_json(silent=True) or {}
    raw_text = (payload.get("text") or "").strip()

    if not raw_text:
        return jsonify(error="Please enter some text to analyze."), 400

    if len(raw_text) > 5000:
        return jsonify(error="Text is too long. Keep it under 5000 characters."), 400

    cleaned = clean_text(raw_text)
    if not cleaned:
        return jsonify(error="Text has no analyzable words after cleanup."), 400

    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]

    proba_map = {}
    try:
        probabilities = model.predict_proba(vectorized)[0]
        proba_map = {
            cls: float(round(p * 100, 2))
            for cls, p in zip(model.classes_, probabilities)
        }
    except Exception:  # noqa: BLE001
        proba_map = {str(prediction): 100.0}

    positive_score = proba_map.get("positive", 0.0)
    negative_score = proba_map.get("negative", 0.0)
    confidence = max(proba_map.values()) if proba_map else 100.0

    signal_words = get_top_signal_words(cleaned)

    return jsonify(
        sentiment=str(prediction),
        confidence=confidence,
        positive=positive_score,
        negative=negative_score,
        word_count=len(cleaned.split()),
        signal_words=signal_words,
        timestamp=datetime.utcnow().strftime("%H:%M:%S"),
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
