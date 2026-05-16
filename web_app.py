import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pymysql
import sqlite3
import hashlib
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import os

# --- 1. НАСТРОЙКИ СТРАНИЦЫ ---
st.set_page_config(page_title="САПР ЭМС (Cloud Edition)", layout="wide")

# --- 2. ИНИЦИАЛИЗАЦИЯ РЕЗЕРВНОЙ БД (SQLite) ---
def init_local_db():
    conn = sqlite3.connect('fallback_local.db')
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS training_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            S_val REAL, H_val REAL, V_val REAL, AddedDate TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS prediction_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            S_in REAL, H_in REAL, V_out REAL, Expert_Verdict TEXT, RequestDate TEXT
        )
    """)
    cur.execute("CREATE TABLE IF NOT EXISTS users (login TEXT, pass_hash TEXT)")
    cur.execute("SELECT * FROM users WHERE login='admin'")
    if not cur.fetchone():
        admin_hash = hashlib.sha256('1234'.encode('utf-8')).hexdigest()
        cur.execute("INSERT INTO users VALUES ('admin', ?)", (admin_hash,))
    conn.commit()
    conn.close()

init_local_db()

# --- 3. ПРОВЕРКА И ПОДКЛЮЧЕНИЕ К БД ---
@st.cache_resource(ttl=30)
def check_db_status():
    try:
        conn = pymysql.connect(
            host='92.53.96.132', user='ct919001_2345',
            password=st.secrets["db_password"], database='ct919001_2345',
            port=3306, connect_timeout=3
        )
        conn.close()
        return True
    except Exception:
        return False

IS_ONLINE = check_db_status()

def get_active_connection():
    if IS_ONLINE:
        return pymysql.connect(
            host='92.53.96.132', user='ct919001_2345',
            password=st.secrets["db_password"], database='ct919001_2345', port=3306
        ), "MySQL 8.0 (Timeweb Cloud)"
    else:
        return sqlite3.connect('fallback_local.db'), "SQLite 3.0 (Автономный режим)"

# --- 4. УПРАВЛЕНИЕ СЕССИЕЙ ---
if "admin_logged_in" not in st.session_state:
    st.session_state.admin_logged_in = False

def load_nn_model():
    if os.path.exists('trained_model.joblib') and os.path.exists('scaler.joblib'):
        return joblib.load('trained_model.joblib'), joblib.load('scaler.joblib')
    return None, None

def hash_password(password):
    return hashlib.sha256(password.encode('utf-8')).hexdigest()

# --- ИНТЕРФЕЙС ---
if IS_ONLINE:
    st.sidebar.success("● Подключено к центральному облаку MySQL")
else:
    st.sidebar.warning("⚡ Автономный режим (Локальное хранилище)")

st.title("🛡️ Облачная Экспертная Система ЭМС")
tab_engineer, tab_admin = st.tabs(["🚀 АРМ Инженера (Анализ)", "⚙️ Панель Разработчика"])

# ==========================================
#        ВКЛАДКА 1: АРМ ИНЖЕНЕРА
# ==========================================
with tab_engineer:
    model, scaler = load_nn_model()

    if model is None:
        st.warning("⚠️ Локальные файлы ИНС (.joblib) не найдены. Зайдите в панель разработчика для генерации.")
    else:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader("Параметры контура")
            s_val = st.number_input("Площадь контура S (см²)", value=45.0, step=1.0)
            h_val = st.number_input("Магнитное поле H (кА/м)", value=0.7, step=0.1)
            calc_btn = st.button("РАССЧИТАТЬ ПОМЕХУ", use_container_width=True, type="primary")

        if calc_btn:
            X_scaled = scaler.transform([[s_val, h_val]])
            v_peak = np.exp(model.predict(X_scaled)[0])

            with col2:
                st.subheader("Результаты анализа")
                st.metric("Амплитуда наведенной помехи (V)", f"{v_peak:.4f} В")
                
                if v_peak < 0.4:
                    st.success("🟢 КРИТЕРИЙ А: Нормальная работа. Сбоев не ожидается.")
                elif v_peak < 1.2:
                    st.warning("🟡 КРИТЕРИЙ B: Возможны ошибки данных (Soft Error).")
                elif v_peak < 2.5:
                    st.error("🟠 КРИТЕРИЙ C: Аппаратное зависание процессора!")
                else:
                    st.error("🔴 КРИТЕРИЙ D: ФАТАЛЬНО. Угроза теплового пробоя!")

                t = np.linspace(0, 0.0002, 1000)
                v_t = v_peak * 1.037 * (np.exp(-14000 * t) - np.exp(-2447000 * t))
                
                fig, ax = plt.subplots(figsize=(8, 3.5))
                ax.plot(t*1e6, v_t, color='#d35400', linewidth=2)
                ax.set_xlabel("Время (мкс)")
                ax.set_ylabel("Напряжение (В)")
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)

                try:
                    db_conn, _ = get_active_connection()
                    cur = db_conn.cursor()
                    placeholder = "%s" if IS_ONLINE else "?"
                    cur.execute(f"INSERT INTO prediction_log (S_in, H_in, V_out, Expert_Verdict, RequestDate) VALUES ({placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder})",
                                (s_val, h_val, v_peak, "Web Prediction", datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                    db_conn.commit()
                    db_conn.close()
                except:
                    pass

# ==========================================
#        ВКЛАДКА 2: АДМИНКА
# ==========================================
with tab_admin:
    db_conn, db_name = get_active_connection()
    st.info(f"Активная база данных: {db_name}")
    db_conn.close()

    if not st.session_state.admin_logged_in:
        st.subheader("🔒 Авторизация Разработчика")
        with st.form("login_form"):
            login = st.text_input("Логин")
            pwd = st.text_input("Пароль", type="password")
            submit = st.form_submit_button("Войти")
            
            if submit:
                pwd_hash = hash_password(pwd)
                if not IS_ONLINE:
                    conn = sqlite3.connect('fallback_local.db')
                    cur = conn.cursor()
                    cur.execute("SELECT * FROM users WHERE login=? AND pass_hash=?", (login, pwd_hash))
                    user = cur.fetchone()
                    conn.close()
                else:
                    conn, _ = get_active_connection()
                    with conn.cursor() as cur:
                        cur.execute("SELECT * FROM users WHERE login=%s AND pass_hash=%s", (login, pwd_hash))
                        user = cur.fetchone()
                    conn.close()
                
                if user:
                    st.session_state.admin_logged_in = True
                    st.rerun()
                else:
                    st.error("Неверный логин или пароль")
    else:
        st.success("✅ Вы авторизованы")
        if st.button("Выйти"):
            st.session_state.admin_logged_in = False
            st.rerun()
            
        st.divider()
        
        # --- НОВАЯ СЕКЦИЯ: ЗАГРУЗКА CSV/EXCEL ЧЕРЕЗ САЙТ ---
        st.subheader("📂 Загрузка новых лабораторных данных")
        uploaded_file = st.file_uploader("Перетащите сюда файл .csv или .xlsx с колонками S, H, V", type=["csv", "xlsx"])
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df_new = pd.read_csv(uploaded_file)
                else:
                    df_new = pd.read_excel(uploaded_file)
                
                st.dataframe(df_new.head(5)) # Показываем превью первых 5 строк
                
                if st.button("📥 Записать этот файл в Облако", type="secondary"):
                    conn, _ = get_active_connection()
                    cur = conn.cursor()
                    placeholder = "%s" if IS_ONLINE else "?"
                    
                    success_count = 0
                    for _, row in df_new.iterrows():
                        cur.execute(f"INSERT INTO training_data (S_val, H_val, V_val, AddedDate) VALUES ({placeholder}, {placeholder}, {placeholder}, {placeholder})",
                                    (float(row['S']), float(row['H']), float(row['V']), datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                        success_count += 1
                        
                    conn.commit()
                    conn.close()
                    st.success(f"Успешно импортировано {success_count} записей в базу данных!")
            except Exception as e:
                st.error(f"Ошибка парсинга файла. Проверьте имена колонок (должны быть S, H, V). Ошибка: {e}")

        st.divider()
        st.subheader("☁️ Синхронизация и переобучение ИНС")
        
        if st.button("🚀 Запустить обучение нейросети на новых данных", type="primary"):
            with st.spinner('Обработка матриц...'):
                try:
                    conn, _ = get_active_connection()
                    if IS_ONLINE:
                        df = pd.read_sql("SELECT S_val, H_val, V_val FROM training_data", conn)
                    else:
                        df = pd.read_sql("SELECT S_val, H_val, V_val FROM training_data", conn)
                        if len(df) < 5: # Если локальная бд пустая, генерируем тест-пакет
                            s_mesh, h_mesh = np.meshgrid(np.linspace(19.6, 78.5, 20), np.linspace(0.175, 1.4, 20))
                            df = pd.DataFrame({'S_val': s_mesh.flatten(), 'H_val': h_mesh.flatten(), 'V_val': np.abs(s_mesh.flatten() * h_mesh.flatten() * 0.015)})
                    conn.close()

                    X = df[['S_val', 'H_val']]
                    y_log = np.log(df['V_val'].values + 1e-5)

                    scaler_new = StandardScaler()
                    X_scaled = scaler_new.fit_transform(X)

                    model_new = MLPRegressor(hidden_layer_sizes=(10, 5), activation='tanh', solver='lbfgs', max_iter=1000, random_state=42)
                    model_new.fit(X_scaled, y_log)

                    joblib.dump(model_new, 'trained_model.joblib')
                    joblib.dump(scaler_new, 'scaler.joblib')

                    st.success(f"🎉 Ядро ИНС успешно переобучено! Использовано строк для обучения: {len(df)}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Критическая ошибка обучения: {e}")
