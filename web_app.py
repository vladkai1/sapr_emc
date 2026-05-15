import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pymysql
import matplotlib.pyplot as plt
import os

# 1. Настройки подключения (ТВОИ ДАННЫЕ)
DB_CONFIG = {
    'host': '92.53.96.132',
    'user': 'ct919001_2345',
    'password': 'ТВОЙ_ПАРОЛЬ_ОТ_БД', 
    'database': 'ct919001_2345',
    'port': 3306
}

# 2. Загрузка модели (те файлы joblib, что мы сделали для препода)
@st.cache_resource # Чтобы не грузить модель каждый раз
def load_nn_model():
    if os.path.exists('trained_model.joblib'):
        model = joblib.load('trained_model.joblib')
        scaler = joblib.load('scaler.joblib')
        return model, scaler
    return None, None

# --- ИНТЕРФЕЙС ---
st.set_page_config(page_title="ЭМС Эксперт", layout="centered")
st.title("🛡️ Мобильный АРМ: ЭМС")

model, scaler = load_nn_model()

if model is None:
    st.error("❌ Файлы модели (.joblib) не найдены в папке!")
else:
    st.success("✅ Нейросеть готова к работе")
    
    # Поля ввода специально для мобилки
    s_val = st.number_input("Площадь контура S (см²)", value=45.0, step=1.0)
    h_val = st.number_input("Магнитное поле H (кА/м)", value=0.7, step=0.1)

    if st.button("РАССЧИТАТЬ", use_container_width=True):
        # Математика
        X_scaled = scaler.transform([[s_val, h_val]])
        v_peak = np.exp(model.predict(X_scaled)[0])

        st.metric("Амплитуда помехи", f"{v_peak:.4f} В")

        # Вердикт по ГОСТ
        if v_peak < 0.4:
            st.success("КРИТЕРИЙ А: Работа в норме")
        elif v_peak < 1.2:
            st.warning("КРИТЕРИЙ B: Возможны сбои")
        else:
            st.error("КРИТЕРИЙ C/D: ОПАСНОСТЬ ПРОБОЯ")

        # График
        t = np.linspace(0, 0.0002, 1000)
        v_t = v_peak * 1.037 * (np.exp(-14000 * t) - np.exp(-2447000 * t))
        
        fig, ax = plt.subplots()
        ax.plot(t*1e6, v_t, color='#1f77b4')
        ax.set_title("Импульс напряжения")
        ax.set_xlabel("мкс")
        ax.set_ylabel("Вольт")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)