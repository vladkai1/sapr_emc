import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pymysql
import hashlib
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import os

# --- 1. НАСТРОЙКИ СТРАНИЦЫ ---
st.set_page_config(page_title="САПР ЭМС (Cloud Edition)", layout="wide")

# --- 2. ПОДКЛЮЧЕНИЕ К БД ---
def get_db_connection():
    return pymysql.connect(
        host='92.53.96.132',
        user='ct919001_2345',
        password=st.secrets["db_password"], 
        database='ct919001_2345',
        port=3306
    )

# --- 3. ИНИЦИАЛИЗАЦИЯ СЕССИИ (Для логина) ---
if "admin_logged_in" not in st.session_state:
    st.session_state.admin_logged_in = False

# --- 4. ЗАГРУЗКА МОДЕЛИ ---
def load_nn_model():
    if os.path.exists('trained_model.joblib') and os.path.exists('scaler.joblib'):
        return joblib.load('trained_model.joblib'), joblib.load('scaler.joblib')
    return None, None

def hash_password(password):
    return hashlib.sha256(password.encode('utf-8')).hexdigest()

# --- ИНТЕРФЕЙС: ВКЛАДКИ ---
st.title("🛡️ Облачная Экспертная Система ЭМС")
tab_engineer, tab_admin = st.tabs(["🚀 АРМ Инженера (Анализ)", "⚙️ Панель Разработчика (Cloud)"])

# ==========================================
#        ВКЛАДКА 1: АРМ ИНЖЕНЕРА
# ==========================================
with tab_engineer:
    model, scaler = load_nn_model()

    if model is None:
        st.warning("⚠️ Нейросеть не обучена! Зайдите в Панель Разработчика и запустите синхронизацию.")
    else:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Параметры контура")
            s_val = st.number_input("Площадь контура S (см²)", value=45.0, step=1.0)
            h_val = st.number_input("Магнитное поле H (кА/м)", value=0.7, step=0.1)
            
            calc_btn = st.button("РАССЧИТАТЬ ПОМЕХУ", use_container_width=True, type="primary")

        if calc_btn:
            # Математика
            X_scaled = scaler.transform([[s_val, h_val]])
            v_peak = np.exp(model.predict(X_scaled)[0])

            with col2:
                st.subheader("Результаты анализа")
                st.metric("Амплитуда наведенной помехи (V)", f"{v_peak:.4f} В")
                
                # Экспертный вердикт
                if v_peak < 0.4:
                    st.success("🟢 КРИТЕРИЙ А: Нормальная работа. Сбоев не ожидается.")
                elif v_peak < 1.2:
                    st.warning("🟡 КРИТЕРИЙ B: Возможны ошибки данных (Soft Error) и искажение битов.")
                elif v_peak < 2.5:
                    st.error("🟠 КРИТЕРИЙ C: Аппаратное зависание. Угроза тиристорного защелкивания!")
                else:
                    st.error("🔴 КРИТЕРИЙ D: ФАТАЛЬНО. Угроза теплового пробоя диэлектрика!")

                # График
                t = np.linspace(0, 0.0002, 1000)
                v_t = v_peak * 1.037 * (np.exp(-14000 * t) - np.exp(-2447000 * t))
                
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(t*1e6, v_t, color='#d35400', linewidth=2)
                ax.set_title("Форма импульса напряжения (1.2/50 мкс)")
                ax.set_xlabel("Время (мкс)")
                ax.set_ylabel("Напряжение (В)")
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)

                # Логирование в БД
                try:
                    conn = get_db_connection()
                    with conn.cursor() as cur:
                        cur.execute("INSERT INTO prediction_log (S_in, H_in, V_out, Expert_Verdict, RequestDate) VALUES (%s, %s, %s, %s, %s)",
                                    (s_val, h_val, v_peak, "Web Prediction", datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                    conn.commit()
                    conn.close()
                except:
                    pass

# ==========================================
#        ВКЛАДКА 2: АДМИНКА
# ==========================================
with tab_admin:
    if not st.session_state.admin_logged_in:
        st.subheader("🔒 Авторизация Разработчика")
        with st.form("login_form"):
            login = st.text_input("Логин")
            pwd = st.text_input("Пароль", type="password")
            submit = st.form_submit_button("Войти")
            
            if submit:
                pwd_hash = hash_password(pwd)
                try:
                    conn = get_db_connection()
                    with conn.cursor() as cur:
                        cur.execute("SELECT * FROM users WHERE login=%s AND pass_hash=%s", (login, pwd_hash))
                        user = cur.fetchone()
                    conn.close()
                    
                    if user:
                        st.session_state.admin_logged_in = True
                        st.rerun() # Перезагрузка страницы
                    else:
                        st.error("Неверный логин или пароль")
                except Exception as e:
                    st.error(f"Ошибка БД: {e}")
    else:
        st.success("✅ Вы авторизованы как Администратор")
        if st.button("Выйти"):
            st.session_state.admin_logged_in = False
            st.rerun()
            
        st.divider()
        st.subheader("☁️ Управление облачной ИНС")
        
        if st.button("🚀 Синхронизировать ИНС с Облаком (Обучить модель)", type="primary"):
            with st.spinner('Скачивание данных и обучение нейросети...'):
                try:
                    conn = get_db_connection()
                    df = pd.read_sql("SELECT S_val, H_val, V_val FROM training_data", conn)
                    conn.close()

                    if len(df) < 10:
                        st.error("В базе слишком мало данных для обучения!")
                    else:
                        X = df[['S_val', 'H_val']]
                        y_log = np.log(df['V_val'].values)

                        scaler_new = StandardScaler()
                        X_scaled = scaler_new.fit_transform(X)

                        model_new = MLPRegressor(hidden_layer_sizes=(10, 5), activation='tanh', solver='lbfgs', max_iter=3000, random_state=42)
                        model_new.fit(X_scaled, y_log)

                        # Сохраняем прямо на сервере Streamlit
                        joblib.dump(model_new, 'trained_model.joblib')
                        joblib.dump(scaler_new, 'scaler.joblib')

                        st.success(f"🎉 Модель успешно обучена на {len(df)} записях и сохранена в облаке!")
                except Exception as e:
                    st.error(f"Ошибка при обучении: {e}")
