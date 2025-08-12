# app.py - Spamlyser Pro - Ensemble Edition (Enhanced with Drag & Drop Bulk Uploader)
# Note: This file is built by merging your provided app.py code and adding
# a polished Bulk Upload / Drag-and-Drop CSV/XLSX uploader tab.
# Make sure all supporting modules (ensemble_classifier_method.py) are available.

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

import time
import re
from datetime import datetime
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import logging
import json
from dataclasses import dataclass
from collections import defaultdict
import torch

# Import the ensemble classifier classes (must exist in your project)
from ensemble_classifier_method import EnsembleSpamClassifier, ModelPerformanceTracker, PredictionResult

# ----- Config & Constants -----
MODEL_OPTIONS = {
    "DistilBERT": {
        "id": "mreccentric/distilbert-base-uncased-spamlyser",
        "description": "Lightweight & Fast",
        "icon": "⚡",
        "color": "#ff6b6b"
    },
    "BERT": {
        "id": "mreccentric/bert-base-uncased-spamlyser",
        "description": "Balanced Performance",
        "icon": "🎯",
        "color": "#4ecdc4"
    },
    "RoBERTa": {
        "id": "mreccentric/roberta-base-spamlyser",
        "description": "Robust & Accurate",
        "icon": "🚀",
        "color": "#45b7d1"
    },
    "ALBERT": {
        "id": "mreccentric/albert-base-v2-spamlyser",
        "description": "Parameter Efficient",
        "icon": "🧠",
        "color": "#96ceb4"
    }
}

ENSEMBLE_METHODS = {
    "majority_voting": {"name": "Majority Voting", "icon": "🗳️"},
    "weighted_average": {"name": "Weighted Average", "icon": "⚖️"},
    "confidence_weighted": {"name": "Confidence Weighted", "icon": "🎯"},
    "adaptive_threshold": {"name": "Adaptive Threshold", "icon": "🔧"},
    "meta_ensemble": {"name": "Meta Ensemble", "icon": "🧠"}
}

# Paths
logo_path = str(Path(__file__).resolve().parent / "SpamlyserLogo.png")

# ----- Streamlit Page Config -----
st.set_page_config(
    page_title="Spamlyser Pro - Ensemble Edition",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----- Session State Initialization -----
if 'classification_history' not in st.session_state:
    st.session_state.classification_history = []
if 'model_stats' not in st.session_state:
    st.session_state.model_stats = {model: {'spam': 0, 'ham': 0, 'total': 0} for model in MODEL_OPTIONS}
if 'ensemble_tracker' not in st.session_state:
    st.session_state.ensemble_tracker = ModelPerformanceTracker()
if 'ensemble_classifier' not in st.session_state:
    st.session_state.ensemble_classifier = EnsembleSpamClassifier(performance_tracker=st.session_state.ensemble_tracker)
if 'ensemble_history' not in st.session_state:
    st.session_state.ensemble_history = []
if 'loaded_models' not in st.session_state:
    st.session_state.loaded_models = {model_name: None for model_name in MODEL_OPTIONS}

# ----- CSS (kept from your original UI) -----
st.markdown("""
<style>
    .main { background: linear-gradient(135deg, #0f0f0f 0%, #1a1a1a 100%); }
    .stApp { background: linear-gradient(135deg, #0f0f0f 0%, #1a1a1a 100%); }
    .prediction-card { background: linear-gradient(145deg, #1a1a1a, #2d2d2d); padding: 25px; border-radius: 20px; border: 1px solid #404040; box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4); text-align: center; margin: 20px 0; }
    .spam-alert { background: linear-gradient(145deg, #2a1a1a, #3d2626); border: 2px solid #ff4444; color: #ff6b6b; }
    .ham-safe { background: linear-gradient(145deg, #1a2a1a, #263d26); border: 2px solid #44ff44; color: #6bff6b; }
    .analysis-header { background: linear-gradient(90deg, #333, #555); padding: 15px; border-radius: 10px; margin: 20px 0; border-left: 4px solid #00d4aa; }
    .feature-card { background: rgba(255, 255, 255, 0.05); backdrop-filter: blur(10px); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 15px; padding: 20px; margin: 10px 0; }
</style>
""", unsafe_allow_html=True)

# ----- Helper: Model Loading & Pipelines -----
@st.cache_resource
def load_tokenizer(model_id):
    try:
        return AutoTokenizer.from_pretrained(model_id)
    except Exception as e:
        st.error(f"Error loading tokenizer for {model_id}: {e}")
        return None

@st.cache_resource
def load_model(model_id):
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = AutoModelForSequenceClassification.from_pretrained(model_id)
        model.to(device)
        return model
    except Exception as e:
        st.error(f"Error loading model {model_id}: {e}")
        return None

@st.cache_resource
def _load_model_cached(model_id):
    try:
        tokenizer = load_tokenizer(model_id)
        model = load_model(model_id)
        if tokenizer is None or model is None:
            return None
        pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)
        return pipe
    except Exception as e:
        st.error(f"Error creating pipeline for {model_id}: {e}")
        return None


def load_model_if_needed(model_name, _progress_callback=None):
    if st.session_state.loaded_models[model_name] is None:
        model_id = MODEL_OPTIONS[model_name]['id']
        status = st.empty()
        try:
            if _progress_callback: _progress_callback(f"Loading {model_name} tokenizer...")
            pipe = _load_model_cached(model_id)
            if pipe is None:
                status.error(f"Failed to load {model_name}")
                return None
            st.session_state.loaded_models[model_name] = pipe
            if _progress_callback: _progress_callback(f"Loaded {model_name}")
        finally:
            status.empty()
    return st.session_state.loaded_models[model_name]


def get_loaded_models(progress_container: Optional[st.delta_generator] = None):
    models = {}
    total = len(MODEL_OPTIONS)
    i = 0
    for name in MODEL_OPTIONS:
        i += 1
        if progress_container:
            progress_container.info(f"Loading {name} ({i}/{total})")
        models[name] = load_model_if_needed(name)
    return models

# ----- Message Feature Analysis & Risk Indicators -----

def analyse_message_features(message):
    features = {
        'length': len(message),
        'word_count': len(message.split()),
        'uppercase_ratio': sum(1 for c in message if c.isupper()) / len(message) if message else 0,
        'digit_ratio': sum(1 for c in message if c.isdigit()) / len(message) if message else 0,
        'special_chars': len(re.findall(r'[!@#$%^&*(),.?":{}|<>]', message)),
        'urls': len(re.findall(r'http[s]?://', message)),
        'phone_numbers': len(re.findall(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', message)),
        'exclamation_marks': message.count('!'),
        'question_marks': message.count('?')
    }
    return features


def get_risk_indicators(message, prediction):
    indicators = []
    spam_keywords = ['free', 'win', 'winner', 'congratulations', 'urgent', 'limited', 'offer', 'click', 'call now']
    found_keywords = [word for word in spam_keywords if word.lower() in message.lower()]
    if found_keywords:
        indicators.append(f"⚠️ Spam keywords detected: {', '.join(found_keywords)}")
    if len(message) > 0:
        uppercase_ratio = sum(1 for c in message if c.isupper()) / len(message)
        if uppercase_ratio > 0.3:
            indicators.append("🔴 Excessive uppercase usage")
    if message.count('!') > 2:
        indicators.append("❗ Multiple exclamation marks")
    if re.search(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', message):
        indicators.append("📞 Phone number detected")
    if re.search(r'http[s]?://', message):
        indicators.append("🔗 URL detected")
    return indicators


def get_ensemble_predictions(message, models):
    predictions = {}
    for model_name, model in models.items():
        if model:
            try:
                result = model(message)[0]
                predictions[model_name] = {'label': result['label'].upper(), 'score': result['score']}
            except Exception as e:
                st.warning(f"Error with {model_name}: {e}")
    return predictions

# ----- UI: Header -----
st.markdown("""
<div style="text-align: center; padding: 20px 0; background: linear-gradient(90deg, #1a1a1a, #2d2d2d); border-radius: 15px; margin-bottom: 30px; border: 1px solid #404040;">
    <h1 style="color: #00d4aa; font-size: 2.4rem; margin: 0;">🛡️ Spamlyser Pro - Ensemble Edition</h1>
    <p style="color: #888; margin: 5px 0 0 0;">Advanced Multi-Model SMS Threat Detection & Bulk Analytics</p>
</div>
""", unsafe_allow_html=True)

# ----- Sidebar Controls -----
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 20px; background: linear-gradient(145deg, #1e1e1e, #2a2a2a); border-radius: 10px; margin-bottom: 10px;">
        <h4 style="color:#00d4aa; margin:0;">Analysis Mode</h4>
    </div>
    """, unsafe_allow_html=True)

    analysis_mode = st.radio("Choose Analysis Mode", ["Single Model", "Ensemble Analysis"], index=0)

    if analysis_mode == "Single Model":
        selected_model_name = st.selectbox("Choose AI Model", list(MODEL_OPTIONS.keys()), format_func=lambda x: f"{MODEL_OPTIONS[x]['icon']} {x} - {MODEL_OPTIONS[x]['description']}")
    else:
        selected_ensemble_method = st.selectbox("Choose Ensemble Method", list(ENSEMBLE_METHODS.keys()), format_func=lambda x: f"{ENSEMBLE_METHODS[x]['icon']} {ENSEMBLE_METHODS[x]['name']}")

    st.markdown("---")
    st.markdown("### 📊 Session Stats")
    if analysis_mode == "Single Model":
        stats = st.session_state.model_stats[selected_model_name]
        total = stats['total']
        if total > 0:
            spam_count = stats['spam']; ham_count = stats['ham']
            spam_rate = (spam_count/total)*100
            st.metric("Total Analysed", total)
            st.metric("Spam Detected", spam_count)
            st.metric("Ham (Safe)", ham_count)
            st.metric("Spam Rate", f"{spam_rate:.1f}%")
        else:
            st.info("No classifications yet")
    else:
        total_ens = len(st.session_state.ensemble_history)
        if total_ens > 0:
            spam_count = sum(1 for h in st.session_state.ensemble_history if h['prediction'] == 'SPAM')
            st.metric("Ensemble Analyses", total_ens)
            st.metric("Spam Detected", spam_count)
        else:
            st.info("No ensemble analyses yet")

# ----- Main Layout with Tabs: Analyse & Bulk Uploader -----
col1, col2 = st.columns([2, 1])

with col1:
    tabs = st.tabs(["🔍 Analyse Message", "📂 Bulk Uploader"])

    # ---------- TAB 1: Single/Ensemble Message Analysis ----------
    with tabs[0]:
        st.markdown('<div class="analysis-header"><h3 style="color:#00d4aa;margin:0;">Message Analysis</h3></div>', unsafe_allow_html=True)
        user_sms = st.text_area("Enter SMS message to analyse", height=120, placeholder="Type or paste your SMS message here...")
        c1, c2 = st.columns([1,1])
        with c1:
            analyse_btn = st.button("🔍 Analyse Message")
        with c2:
            if st.button("🗑️ Clear"):
                st.experimental_rerun()

        if analyse_btn and user_sms.strip():
            if analysis_mode == "Single Model":
                classifier = load_model_if_needed(selected_model_name)
                if classifier:
                    with st.spinner(f"Analysing with {selected_model_name}..."):
                        time.sleep(0.3)
                        res = classifier(user_sms)[0]
                        label = res['label'].upper(); confidence = res['score']
                        st.session_state.model_stats[selected_model_name][label.lower()] += 1
                        st.session_state.model_stats[selected_model_name]['total'] += 1
                        st.session_state.classification_history.append({'timestamp': datetime.now(), 'message': user_sms[:50]+('...' if len(user_sms)>50 else ''), 'prediction': label, 'confidence': confidence, 'model': selected_model_name})

                        features = analyse_message_features(user_sms)
                        risk_indicators = get_risk_indicators(user_sms, label)

                        card_class = 'spam-alert' if label == 'SPAM' else 'ham-safe'
                        icon = '🚨' if label == 'SPAM' else '✅'
                        st.markdown(f"<div class='prediction-card {card_class}'><h2 style='margin:0 0 10px 0;'>{icon} {label}</h2><h4>Confidence: {confidence:.2%}</h4><p style='opacity:0.8;'>Model: {selected_model_name} • {datetime.now().strftime('%H:%M:%S')}</p></div>", unsafe_allow_html=True)

            else:
                models = {}
                with st.spinner("Loading ensemble models..."):
                    for mn in MODEL_OPTIONS:
                        models[mn] = load_model_if_needed(mn)
                if any(models.values()):
                    with st.spinner("Running ensemble..."):
                        predictions = get_ensemble_predictions(user_sms, models)
                        ensemble_result = st.session_state.ensemble_classifier.get_ensemble_prediction(predictions, selected_ensemble_method)
                        st.session_state.ensemble_history.append({'timestamp': datetime.now(), 'message': user_sms[:50]+('...' if len(user_sms)>50 else ''), 'prediction': ensemble_result['label'], 'confidence': ensemble_result['confidence'], 'method': selected_ensemble_method, 'spam_probability': ensemble_result['spam_probability']})

                        card_class = 'spam-alert' if ensemble_result['label'] == 'SPAM' else 'ham-safe'
                        icon = '🚨' if ensemble_result['label'] == 'SPAM' else '✅'
                        st.markdown(f"<div class='prediction-card {card_class}'><h2 style='margin:0 0 10px 0;'>{icon} {ensemble_result['label']}</h2><h4>Confidence: {ensemble_result['confidence']:.2%}</h4><h5>Spam Prob: {ensemble_result['spam_probability']:.2%}</h5></div>", unsafe_allow_html=True)

        # Show features if available
        if 'features' in locals():
            d1, d2 = st.columns(2)
            with d1:
                st.markdown('#### 📋 Message Features')
                st.markdown(f"<div class='feature-card'><strong>Length:</strong> {features['length']}<br><strong>Words:</strong> {features['word_count']}<br><strong>Uppercase:</strong> {features['uppercase_ratio']:.1%}<br><strong>Numbers:</strong> {features['digit_ratio']:.1%}</div>", unsafe_allow_html=True)
            with d2:
                st.markdown('#### ⚠️ Risk Indicators')
                if risk_indicators:
                    for ind in risk_indicators:
                        st.markdown(f"- {ind}")
                else:
                    st.markdown("✅ No significant risk indicators detected")

    # ---------- TAB 2: Bulk Uploader ----------
    with tabs[1]:
        st.markdown('<div class="analysis-header"><h3 style="color:#00d4aa;margin:0;">Bulk Uploader</h3></div>', unsafe_allow_html=True)
        st.markdown("Upload a CSV or Excel file with a `message` column. The app will classify all messages and show instant analytics.")

        uploaded_file = st.file_uploader("Drag & Drop your CSV/Excel file here", type=["csv", "xlsx"] , key='bulk_uploader')
        use_ensemble_for_bulk = st.checkbox("Use Ensemble for bulk classification (slower)", value=False)
        bulk_selected_model = None
        if not use_ensemble_for_bulk:
            # select single model for speed
            bulk_selected_model = st.selectbox("Select model for bulk classification (faster)", list(MODEL_OPTIONS.keys()))

        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df_bulk = pd.read_csv(uploaded_file)
                else:
                    df_bulk = pd.read_excel(uploaded_file)
            except Exception as e:
                st.error(f"Failed to read the uploaded file: {e}")
                df_bulk = None

            if df_bulk is not None:
                if 'message' not in df_bulk.columns:
                    st.error("CSV/XLSX must contain a 'message' column")
                else:
                    st.success(f"Loaded {len(df_bulk)} messages")

                    # Option: preview
                    if st.checkbox("Preview first 10 rows"):
                        st.dataframe(df_bulk.head(10))

                    if st.button("🚀 Run Bulk Classification"):
                        results = []
                        progress = st.progress(0)
                        status = st.empty()

                        # Decide pipeline(s)
                        if use_ensemble_for_bulk:
                            status.info("Loading ensemble models...")
                            models = get_loaded_models(status)
                        else:
                            status.info(f"Loading {bulk_selected_model} model...")
                            single_pipe = load_model_if_needed(bulk_selected_model)

                        batch_size = 32
                        messages = df_bulk['message'].fillna('').astype(str).tolist()
                        n = len(messages)

                        for i in range(0, n, batch_size):
                            batch = messages[i:i+batch_size]
                            status.info(f"Classifying messages {i+1}-{min(i+batch_size,n)} of {n}")

                            if use_ensemble_for_bulk:
                                # get predictions per model then apply ensemble
                                batch_preds = []
                                # use get_ensemble_predictions on joined messages individually
                                for j, msg in enumerate(batch):
                                    preds = get_ensemble_predictions(msg, models)
                                    ens = st.session_state.ensemble_classifier.get_ensemble_prediction(preds, selected_ensemble_method)
                                    batch_preds.append({'label': ens['label'], 'confidence': ens['confidence'], 'spam_probability': ens['spam_probability']})
                            else:
                                # single model classification via pipeline
                                try:
                                    out = single_pipe(batch)
                                    batch_preds = [{'label': o['label'].upper(), 'confidence': o['score'], 'spam_probability': o['score'] if o['label'].upper()=='SPAM' else 1-o['score']} for o in out]
                                except Exception as e:
                                    st.error(f"Error during classification: {e}")
                                    break

                            for j, pred in enumerate(batch_preds):
                                results.append(pred)

                            progress.progress(min((i+batch_size)/n, 1.0))

                        # attach results to dataframe
                        df_results = df_bulk.copy()
                        df_results['prediction'] = [r['label'] for r in results]
                        df_results['confidence'] = [r['confidence'] for r in results]

                        st.success("Bulk classification completed")
                        status.empty(); progress.empty()

                        # Show summary chart
                        st.markdown('### 📊 Spam/Ham Distribution')
                        fig = px.pie(df_results, names='prediction', title='Spam vs Ham', hole=0.3)
                        st.plotly_chart(fig, use_container_width=True)

                        # Show top risk indicators (simple)
                        st.markdown('### 🔎 Quick Insights')
                        spam_count = len(df_results[df_results['prediction']=='SPAM'])
                        ham_count = len(df_results[df_results['prediction']=='HAM'])
                        st.metric('Total Messages', len(df_results))
                        st.metric('Spam Detected', spam_count)
                        st.metric('Ham (Safe)', ham_count)

                        # Dataframe and download
                        st.markdown('#### 📄 Sample Results')
                        st.dataframe(df_results.head(50), use_container_width=True)

                        csv_out = df_results.to_csv(index=False).encode('utf-8')
                        st.download_button('📥 Download Classified CSV', data=csv_out, file_name=f'spamlyser_bulk_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv', mime='text/csv')

# ----- Right Column: Analytics & Recent -----
with col2:
    st.markdown('<div class="analysis-header"><h3 style="color:#00d4aa;margin:0;">📈 Analytics</h3></div>', unsafe_allow_html=True)

    if analysis_mode == 'Single Model' and st.session_state.classification_history:
        st.markdown('#### 🕒 Recent Classifications')
        recent = st.session_state.classification_history[-5:]
        for item in reversed(recent):
            color = '#ff6b6b' if item['prediction']=='SPAM' else '#4ecdc4'
            st.markdown(f"<div style='background:rgba(255,255,255,0.04);padding:8px;border-radius:8px;margin:5px 0;border-left:3px solid {color};'><strong style='color:{color};'>{item['prediction']}</strong> ({item['confidence']:.1%})<br><small style='color:#888;'>{item['message']}</small><br><small style='color:#666;'>{item['model']} • {item['timestamp'].strftime('%H:%M')}</small></div>", unsafe_allow_html=True)

    elif analysis_mode == 'Ensemble Analysis' and st.session_state.ensemble_history:
        st.markdown('#### 🕒 Recent Ensemble Results')
        recent = st.session_state.ensemble_history[-5:]
        for item in reversed(recent):
            color = '#ff6b6b' if item['prediction']=='SPAM' else '#4ecdc4'
            st.markdown(f"<div style='background:rgba(255,255,255,0.04);padding:8px;border-radius:8px;margin:5px 0;border-left:3px solid {color};'><strong style='color:{color};'>{item['prediction']}</strong> ({item['confidence']:.1%})<br><small style='color:#888;'>{item['message']}</small><br><small style='color:#666;'>{ENSEMBLE_METHODS[item['method']]['name']} • {item['timestamp'].strftime('%H:%M')}</small></div>", unsafe_allow_html=True)

    else:
        st.info('📝 No classifications yet. Use the Analyse or Bulk Uploader tab to start!')

# ----- Footer -----
st.markdown('---')
st.markdown("""
<div style="text-align: center; padding: 20px; background: rgba(255,255,255,0.02); border-radius: 10px; margin-top: 30px;">
    <p style="color: #888; margin: 0;">🛡️ <strong>Spamlyser Pro - Ensemble Edition</strong> | Bulk Analytics • Developed by MrEccentric</p>
</div>
""", unsafe_allow_html=True)

# End of file
