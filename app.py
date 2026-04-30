import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ===================== LOAD MODEL & ASSETS =====================
@st.cache_resource
def load_ml_assets():
    try:
        # Mengacu pada struktur folder yang established
        model = joblib.load("model/model_delivery.pkl")
        scaler = joblib.load("model/scaler.pkl")
        feature_names = joblib.load("model/features.pkl")
        return model, scaler, feature_names
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None, None, None

model, scaler, feature_names = load_ml_assets()

# ===================== CONFIG =====================
st.set_page_config(
    page_title="Food Delivery Time Prediction",
    layout="wide",
    page_icon="🛵",
    initial_sidebar_state="collapsed"
)

# ===================== SESSION STATE =====================
if 'pred' not in st.session_state:
    st.session_state.pred = 0
if 'dist' not in st.session_state:
    st.session_state.dist = 5.0
if 'prep' not in st.session_state:
    st.session_state.prep = 15
if 'exp' not in st.session_state:
    st.session_state.exp = 2
if 'traffic' not in st.session_state:
    st.session_state.traffic = "Low"
if 'weather' not in st.session_state:
    st.session_state.weather = "Sunny"
if 'vehicle' not in st.session_state:
    st.session_state.vehicle = "Scooter"
if 'time_day' not in st.session_state:
    st.session_state.time_day = "Morning"

# ===================== PREPROCESS FUNCTION (SESUAI SNIPPET) =====================
def preprocess_and_predict(dist, prep, exp, traffic, weather, vehicle, time_day):
    if model is None:
        return 0

    import pandas as pd

    # ================= INPUT =================
    df = pd.DataFrame([{
        "Distance_km": dist,
        "Preparation_Time_min": prep,
        "Courier_Experience_yrs": exp,
        "Traffic_Level": traffic,
        "Weather": weather,
        "Time_of_Day": time_day,
        "Vehicle_Type": vehicle
    }])

    # ================= FEATURE ENGINEERING =================
    traffic_map = {'Low': 0, 'Medium': 1, 'High': 2}
    df['Traffic_Score'] = df['Traffic_Level'].map(traffic_map)

    df['Is_Difficult_Scenario'] = (
        (df['Traffic_Level'] == 'High') &
        (df['Weather'].isin(['Rainy', 'Snowy', 'Foggy']))
    ).astype(int)

    df['Distance_Traffic'] = df['Distance_km'] * df['Traffic_Score']
    df['Prep_Distance'] = df['Preparation_Time_min'] * df['Distance_km']

    # Drop kolom lama
    df.drop(columns=['Traffic_Level'], inplace=True)

    # ================= ENCODING =================
    df = pd.get_dummies(
        df,
        columns=['Weather', 'Vehicle_Type', 'Time_of_Day'],
        drop_first=True   # WAJIB sama dengan training
    )

    # ================= SAMAKAN FITUR =================
    df = df.reindex(columns=feature_names, fill_value=0)

    # ================= FIX SCALER MISMATCH =================
    # pastikan semua kolom yang dibutuhkan scaler ada
    for col in scaler.feature_names_in_:
        if col not in df.columns:
            df[col] = 0

    # urutan HARUS sama
    df_scaled_part = df[scaler.feature_names_in_]

    # ================= SCALING =================
    df[scaler.feature_names_in_] = scaler.transform(df_scaled_part)

    # ================= PREDIKSI =================
    prediction = model.predict(df)[0]

    return round(prediction, 2)

# ===================== CSS STYLING =====================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    * { font-family: 'Plus Jakarta Sans', sans-serif; }
    .stApp { background-color: #F8FAFC; }
    
    .main-header {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        padding: 1.8rem 2.2rem;
        border-radius: 1rem;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 1.2rem;
    }

    .main-header h1 {
        color: white !important;
        margin: 0;
        font-size: 2rem;          /* 🔥 INI PAS */
        font-weight: 800;
        letter-spacing: 0.5px;
    }

    .main-header p {
        color: #94A3B8 !important;
        margin: 0;
        font-size: 0.85rem;
    }

    .kpi-card {
        background: white; padding: 1rem; border-radius: 0.75rem;
        border: 1px solid #E2E8F0; display: flex; align-items: center; gap: 1rem;
    }
    .kpi-icon {
        font-size: 1.5rem; width: 45px; height: 45px;
        background: #F1F5F9; border-radius: 0.75rem;
        display: flex; align-items: center; justify-content: center;
    }

    .card-custom, [data-testid="stForm"] {
        background: white !important; border-radius: 0.75rem !important;
        border: 1px solid #E2E8F0 !important; padding: 1.5rem !important;
    }

    .result-circle {
        width: 140px; height: 140px;
        background: linear-gradient(135deg, #FF6B35 0%, #FF4D00 100%);
        border-radius: 50%; margin: 0 auto;
        display: flex; flex-direction: column; align-items: center; justify-content: center;
        box-shadow: 0 10px 20px rgba(255, 107, 53, 0.3);
    }
    .circle-val { font-size: 42px; font-weight: 800; color: white; line-height: 1; }
    .circle-unit { font-size: 14px; color: white; opacity: 0.9; }

    .detail-row {
        display: flex; justify-content: space-between;
        padding: 8px 0; border-bottom: 1px solid #F1F5F9; font-size: 13px;
    }
    .detail-label { color: #64748B; display: flex; align-items: center; gap: 8px; }
    .detail-val { font-weight: 700; color: #1E293B; }

    .progress-wrapper { margin-bottom: 12px; }
    .progress-bar { height: 6px; background: #F1F5F9; border-radius: 10px; overflow: hidden; }
    .progress-fill { height: 100%; background: linear-gradient(90deg, #FF6B35, #FF4D00); border-radius: 10px; }

    .badge { padding: 4px 10px; border-radius: 20px; font-size: 10px; font-weight: 700; }
    .badge-warn { background: #FFFBEB; color: #B45309; border: 1px solid #FDE68A; }
    .badge-success { background: #F0FDF4; color: #15803D; border: 1px solid #DCFCE7; }
    
    [data-testid="stFormSubmitButton"] > button {
        background: linear-gradient(135deg, #FF6B35 0%, #FF4D00 100%) !important;
        color: white !important; border: none !important; width: 100% !important;
        font-weight: 700 !important; padding: 0.75rem !important; border-radius: 0.5rem !important;
    }

    /* HAPUS JARAK ATAS STREAMLIT */
    .block-container {
        padding-top: 0.8rem !important;   /* defaultnya besar */
    }

    /* optional: lebih mepet lagi */
    section.main > div {
        padding-top: 0rem !important;
    }
    
    header {
        visibility: hidden;
    }
</style>
""", unsafe_allow_html=True)

# ===================== UI HEADER =====================
st.markdown("""
<div class="main-header">
    <div style="font-size: 3.0rem;">🚚</div>
    <div>
        <h1>FOOD DELIVERY TIME PREDICTION SYSTEM</h1>
        <p>Accurate delivery time estimation using machine learning</p>
    </div>
</div>
""", unsafe_allow_html=True)

# KPI Row
k1, k2, k3, k4 = st.columns(4)
kpis = [("🕒", "MAE ERROR", "5.37 min"), ("📍", "MAPE", "10.3%"), ("🍳", "RMSE", "8.30 min"), ("🎯", "R2 SCORE", "83.2%")]
for i, col in enumerate([k1, k2, k3, k4]):
    col.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-icon">{kpis[i][0]}</div>
        <div>
            <div style="font-size: 10px; color: #64748B; font-weight: 700;">{kpis[i][1]}</div>
            <div style="font-size: 1.2rem; font-weight: 800; color: #1E293B;">{kpis[i][2]}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# ===================== MAIN CONTENT =====================
col_left, col_right = st.columns([1.2, 1], gap="medium")

with col_left:
    with st.form(key="ml_prediction_form"):
        st.markdown("<h3 style='margin-bottom:1.5rem;'>🎯 Delivery Time Estimator</h3>", unsafe_allow_html=True)
        
        form_dist = st.slider("Jarak Tempuh (km)", 0.5, 30.0, st.session_state.dist, step=0.1)
        
        c1, c2 = st.columns(2)
        with c1:
            form_prep = st.number_input("Waktu Persiapan (menit)", 5, 60, st.session_state.prep)
            form_traffic = st.selectbox("Tingkat Kemacetan", ["Low", "Medium", "High"], 
                                       index=["Low", "Medium", "High"].index(st.session_state.traffic))
            form_weather = st.selectbox("Kondisi Cuaca", ["Sunny", "Rainy", "Foggy", "Snowy", "Windy"], 
                                       index=["Sunny", "Rainy", "Foggy", "Snowy", "Windy"].index(st.session_state.weather))
        with c2:
            form_exp = st.number_input("Pengalaman Kurir (tahun)", 0, 15, st.session_state.exp)
            form_vehicle = st.selectbox("Jenis Kendaraan", ["Scooter", "Bike", "Car"],
                                       index=["Scooter", "Bike", "Car"].index(st.session_state.vehicle))
            form_time_day = st.selectbox("Waktu Pengiriman", ["Morning", "Evening", "Night"], 
                                        index=["Morning", "Evening", "Night"].index(st.session_state.time_day))

        submit = st.form_submit_button("🚀 PREDIKSI SEKARANG")
        
        if submit:
            st.session_state.dist = form_dist
            st.session_state.prep = form_prep
            st.session_state.exp = form_exp
            st.session_state.traffic = form_traffic
            st.session_state.weather = form_weather
            st.session_state.vehicle = form_vehicle
            st.session_state.time_day = form_time_day
            
            st.session_state.pred = preprocess_and_predict(
                form_dist, form_prep, form_exp, form_traffic, form_weather, form_vehicle, form_time_day
            )
            st.rerun()

with col_right:
    risk_class = "badge-success" if st.session_state.pred < 45 else "badge-warn"
    risk_label = "Aman" if st.session_state.pred < 45 else "Potensi Delay"
    
    st.markdown(f"""
    <div class="card-custom">
        <h3 style="margin-top:0;">📊 Estimated Delivery Time</h3>
        <div style="display: flex; align-items: center; justify-content: space-between; margin-top: 25px;">
            <div style="text-align: center; flex: 1;">
                <div class="result-circle">
                    <div class="circle-val">{st.session_state.pred if st.session_state.pred > 0 else '--'}</div>
                    <div class="circle-unit">menit</div>
                </div>
                <div style="margin-top:15px;"><span class="badge {risk_class}">{risk_label}</span></div>
                <div style="font-size: 11px; color: #64748B; margin-top: 5px;">Model: Linear Regression</div>
            </div>
            <div style="flex: 1.2; padding-left: 20px; border-left: 1px solid #F1F5F9;">
                <div style="font-size: 12px; font-weight: 700; margin-bottom: 10px;">📋 Ringkasan Input</div>
                <div class="detail-row"><span class="detail-label">📍 Jarak</span><span class="detail-val">{st.session_state.dist} km</span></div>
                <div class="detail-row"><span class="detail-label">🍳 Persiapan</span><span class="detail-val">{st.session_state.prep} min</span></div>
                <div class="detail-row"><span class="detail-label">👤 Kurir</span><span class="detail-val">{st.session_state.exp} thn</span></div>
                <div class="detail-row"><span class="detail-label">🚥 Traffic</span><span class="detail-val">{st.session_state.traffic}</span></div>
                <div class="detail-row"><span class="detail-label">🛵 Kendaraan</span><span class="detail-val">{st.session_state.vehicle}</span></div>
                <div class="detail-row"><span class="detail-label">⏰ Waktu</span><span class="detail-val">{st.session_state.time_day}</span></div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# ===================== BOTTOM ANALYTICS =====================
b1, b2, b3 = st.columns([1, 1, 1.2])

with b1:
    factors_html = """<div class="card-custom"><h4>📈 Feature Importance</h4>"""
    importance_data = [("Distance (km)", 28.5), ("Prep Time (min)", 12.8), ("Weather", 8.7), ("Traffic", 4.9)]
    max_val = 28.5
    for name, val in importance_data:
        pct = int((val / max_val) * 100)
        factors_html += f"""
        <div class="progress-wrapper">
            <div style="display:flex; justify-content:space-between; font-size:11px; margin-bottom:4px;">
                <span>{name}</span><span>{val}</span>
            </div>
            <div class="progress-bar"><div class="progress-fill" style="width:{pct}%;"></div></div>
        </div>"""
    factors_html += "</div>"
    st.markdown(factors_html, unsafe_allow_html=True)

with b2:
    st.markdown(f"""
    <div class="card-custom" style="height:100%;">
        <h4>💡 Analisis Insights</h4>
        <ul style="font-size: 12px; color: #475569; padding-left: 1rem; line-height: 1.8;">
            <li><b>Distance:</b> Faktor paling berpengaruh terhadap waktu pengiriman.</li>
            <li><b>Prep Time:</b> Waktu persiapan yang lebih lama berdampak pada estimasi waktu pengiriman.</li>
            <li><b>Weather:</b> Kondisi cuaca yang buruk memperlambat proses pengiriman.</li>
            <li><b>Traffic:</b> Tingkat kemacetan mempengaruhi kecepatan pengiriman.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with b3:
    st.markdown("""
    <div class="card-custom" style="height:100%;">
        <h4>❓ Model Overview</h4>
        <div style="background: #F8FAFC; padding: 0.75rem; border-radius: 0.5rem; font-size: 0.8rem; border-left: 4px solid #FF4D00;">
            <b>Cleaning:</b> Missing values handled & outliers capped.<br>
            <b>Scaling:</b> RobustScaler applied.<br>
            <b>Algorithm:</b> Linear Regression.<br>
            <b>Encoding:</b> One-Hot Encoding.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
<style>
.footer {
    position: fixed;
    bottom: 0;
    left: 0;
    width: 100%;
    background: #0F172A;
    color: #CBD5F5;
    text-align: center;
    padding: 10px 0;
    font-size: 13px;
    border-top: 1px solid #1E293B;
    z-index: 999;
}

.footer span {
    color: #FF6B35;
    font-weight: 600;
}
</style>

<div class="footer">
    🚚 Food Delivery App • Built by <span>Ahmad Aldiyanto</span> • 2026
</div>
""", unsafe_allow_html=True)