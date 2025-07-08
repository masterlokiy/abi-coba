import streamlit as st
import pandas as pd
import numpy as np
import urllib.parse
import joblib
from tensorflow.keras.models import load_model

# Muat model dari folder "model"
@st.cache_resource
def load_model_bundle():
    rf_model = joblib.load('./model/model_rf_30.pkl')
    lstm_model = load_model('./model/model_lstm_30.h5')
    scaler = joblib.load('./model/scaler_30.pkl')
    selected_features = joblib.load('./model/selected_features_30.pkl')
    return rf_model, lstm_model, scaler, selected_features

rf_model_30, lstm_model_30, scaler_30, selected_features_30 = load_model_bundle()

# Fungsi untuk mengekstrak fitur dari URL berdasarkan selected_features_30
def extract_features(url):
    parsed_url = urllib.parse.urlparse(url if url.startswith(('http://', 'https://')) else 'http://' + url)
    
    # Inisialisasi fitur dengan nilai default 0 berdasarkan selected_features_30
    features = {feat: 0 for feat in selected_features_30}
    
    # Ekstraksi fitur yang umum digunakan
    features['length_url'] = len(url)
    if parsed_url.hostname:
        features['length_hostname'] = len(parsed_url.hostname)
        features['ip'] = 1 if all(part.isdigit() or part == '.' for part in parsed_url.hostname.split('.')) else 0
        features['ratio_digits_host'] = sum(c.isdigit() for c in parsed_url.hostname) / len(parsed_url.hostname) if parsed_url.hostname else 0
    features['nb_dots'] = url.count('.')
    features['nb_hyphens'] = url.count('-')
    features['nb_at'] = url.count('@')
    features['nb_qm'] = url.count('?')
    features['nb_and'] = url.count('&')
    features['nb_or'] = url.count('|')
    features['nb_eq'] = url.count('=')
    features['nb_underscore'] = url.count('_')
    features['nb_tilde'] = url.count('~')
    features['nb_percent'] = url.count('%')
    features['nb_slash'] = url.count('/')
    features['nb_star'] = url.count('*')
    features['nb_colon'] = url.count(':')
    features['nb_comma'] = url.count(',')
    features['nb_semicolumn'] = url.count(';')
    features['nb_dollar'] = url.count('$')
    features['nb_space'] = url.count(' ')
    features['nb_www'] = 1 if 'www' in url.lower() else 0
    features['nb_com'] = 1 if '.com' in url.lower() else 0
    features['nb_dslash'] = url.count('//')
    features['http_in_path'] = 1 if 'http' in parsed_url.path.lower() else 0
    features['https_token'] = 1 if url.lower().startswith('https') else 0
    features['ratio_digits_url'] = sum(c.isdigit() for c in url) / len(url) if url else 0
    features['punycode'] = 1 if 'xn--' in url.lower() else 0
    features['port'] = 1 if parsed_url.port else 0
    features['tld_in_path'] = 1 if any(tld in parsed_url.path.lower() for tld in ['.com', '.org', '.net']) else 0

    return features

# Fungsi untuk memprediksi URL
def predict_url(url):
    try:
        features = extract_features(url)
        new_data = pd.DataFrame([features])[selected_features_30]
        X_new_scaled = scaler_30.transform(new_data)

        # Prediksi dengan Random Forest
        rf_pred = rf_model_30.predict(new_data)[0]
        rf_proba = rf_model_30.predict_proba(new_data)[0][1]

        # Prediksi dengan LSTM
        X_new_lstm = X_new_scaled.reshape((1, 1, X_new_scaled.shape[1]))
        lstm_pred_proba = lstm_model_30.predict(X_new_lstm, verbose=0)[0][0]
        lstm_pred = 1 if lstm_pred_proba > 0.5 else 0

        # Ensemble Voting
        final_pred = 1 if rf_pred + lstm_pred > 1 else 0

        return {
            'url': url,
            'final_prediction': final_pred,
            'rf_proba': rf_proba,
            'lstm_proba': lstm_pred_proba
        }
    except Exception as e:
        return {'url': url, 'error': str(e)}

# UI Streamlit
st.title("Phishing Website Detector")
st.write("Masukkan URL untuk memeriksa apakah itu phishing atau aman.")

url = st.text_input("URL", "https://www.google.com")

if st.button("Cek URL"):
    with st.spinner("Menganalisis URL..."):
        result = predict_url(url)

    if 'error' in result:
        st.error(f"Error: {result['error']}")
    else:
        st.success("Analisis Selesai!")
        st.write(f"URL: {result['url']}")
        st.write(f"Hasil Analisis: **{'Phishing' if result['final_prediction'] == 1 else 'Aman'}**")
        st.write(f"Probabilitas RF: {result['rf_proba']:.2f}")
        st.write(f"Probabilitas LSTM: {result['lstm_proba']:.2f}")

st.write("---")
st.write("Dibuat pada: 10:27 AM WIB, Kamis, 19 Juni 2025")
st.write("Model menggunakan 30 fitur terpilih dari dataset awal.")
