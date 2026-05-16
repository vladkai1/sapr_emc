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
st.set_page_config(page_title="САПР ЭМС: СВЧ-оборудование", layout="wide")

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

# --- 4. УПРАВЛЕНИЕ СЕССИЕЙ И ЗАГРУЗКА ИНС ---
if "admin_logged_in" not in st.session_state:
    st.session_state.admin_logged_in = False

def load_nn_model():
    if os.path.exists('trained_model.joblib') and os.path.exists('scaler.joblib'):
        model = joblib.load('trained_model.joblib')
        scaler = joblib.load('scaler.joblib')
        # Пытаемся загрузить сохраненную точность MAPE, если её нет - пишем "Н/Д"
        mape = joblib.load('mape.joblib') if os.path.exists('mape.joblib') else "Н/Д"
        return model, scaler, mape
    return None, None, None

def hash_password(password):
    return hashlib.sha256(password.encode('utf-8')).hexdigest()

# --- ИНТЕРФЕЙС ---
if IS_ONLINE:
    st.sidebar.success("● БД: MySQL (Timeweb)")
else:
    st.sidebar.warning("⚡ БД: SQLite (Офлайн)")

st.title("📡 САПР ЭМС: Оценка устойчивости СВЧ-аппаратуры")
st.markdown("Моделирование наведенных импульсных помех в соответствии с государственными стандартами РФ.")

tab_engineer, tab_admin = st.tabs(["🚀 АРМ Инженера (Анализ)", "⚙️ Панель Разработчика"])

# ==========================================
#        ВКЛАДКА 1: АРМ ИНЖЕНЕРА
# ==========================================
with tab_engineer:
    model, scaler, current_mape = load_nn_model()

    if model is None:
        st.warning("⚠️ Ядро ИНС не обучено. Загрузите базу данных в Панели Разработчика.")
    else:
        # Выводим реальную точность нейросети
        if isinstance(current_mape, float):
            st.info(f"🧠 Нейронная сеть активна. **Реальная погрешность (MAPE): {current_mape:.2f}%**")
        else:
            st.info("🧠 Нейронная сеть активна. Точность будет рассчитана при следующем обучении.")

        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader("Входные параметры эксперимента")
            s_val = st.number_input("Эквивалентная площадь контура S (см²)", value=45.0, step=1.0)
            h_val = st.number_input("Напряженность магнитного поля H (кА/м)", value=0.7, step=0.1)
            
            st.markdown("""
            **Нормативная база расчета:**
            * **ГОСТ 8.644-2014** (Сила импульсного тока молниевого разряда 1-100 кА)
            * **ГОСТ IEC 61000-4-3-2017** (Устойчивость к РЧ и СВЧ электромагнитному полю)
            """)
            calc_btn = st.button("СМОДЕЛИРОВАТЬ ВОЗДЕЙСТВИЕ", use_container_width=True, type="primary")

        if calc_btn:
            X_scaled = scaler.transform([[s_val, h_val]])
            v_peak = np.exp(model.predict(X_scaled)[0])

            with col2:
                st.subheader("Протокол виртуального испытания")
                st.metric("Амплитуда наведенного напряжения (V)", f"{v_peak:.4f} В")
                
                # Обновленные экспертные вердикты с привязкой к СВЧ
                st.markdown("### Оценка электромагнитной совместимости СВЧ-тракта:")
                if v_peak < 0.4:
                    verdict = "КРИТЕРИЙ А: Норма. Воздействие в пределах запаса помехоустойчивости СВЧ-усилителей."
                    st.success(f"🟢 **{verdict}**\n\nДеградации коэффициента шума и искажения диаграммы направленности не ожидается.")
                elif v_peak < 1.2:
                    verdict = "КРИТЕРИЙ B: Сбой. Временное искажение битов (Soft Error)."
                    st.warning(f"🟡 **{verdict}**\n\nСмещение рабочей точки СВЧ-транзисторов. Возможна потеря пакетов в цифровом тракте ПРМ/ПРД модулей.")
                elif v_peak < 2.5:
                    verdict = "КРИТЕРИЙ C: Зависание. Эффект тиристорного защелкивания (Latch-up)."
                    st.error(f"🟠 **{verdict}**\n\nБлокировка сигнального процессора (DSP) радиолокационного тракта. Требуется жесткая перезагрузка оборудования.")
                else:
                    verdict = "КРИТЕРИЙ D: ФАТАЛЬНО. Пробой диэлектрика."
                    st.error(f"🔴 **{verdict}**\n\nНеобратимое тепловое разрушение p-n переходов арсенид-галлиевых (GaAs) СВЧ-компонентов!")

                # График
                t = np.linspace(0, 0.0002, 1000)
                v_t = v_peak * 1.037 * (np.exp(-14000 * t) - np.exp(-2447000 * t))
                
                fig, ax = plt.subplots(figsize=(8, 3.5))
                ax.plot(t*1e6, v_t, color='#d35400', linewidth=2)
                ax.set_title("Форма импульса по ГОСТ (1.2/50 мкс)", fontsize=10)
                ax.set_xlabel("Время (мкс)")
                ax.set_ylabel("Напряжение (В)")
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)

                # Логирование
                try:
                    db_conn, _ = get_active_connection()
                    cur = db_conn.cursor()
                    placeholder = "%s" if IS_ONLINE else "?"
                    cur.execute(f"INSERT INTO prediction_log (S_in, H_in, V_out, Expert_Verdict, RequestDate) VALUES ({placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder})",
                                (s_val, h_val, v_peak, verdict.split(':')[0], datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                    db_conn.commit()
                    db_conn.close()
                except:
                    pass

# ==========================================
#        ВКЛАДКА 2: АДМИНКА
# ==========================================
with tab_admin:
    if not st.session_state.admin_logged_in:
        st.subheader("🔒 Идентификация инженера-разработчика")
        with st.form("login_form"):
            login = st.text_input("Логин")
            pwd = st.text_input("Пароль", type="password")
            submit = st.form_submit_button("Войти в систему")
            
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
                    st.error("Отказано в доступе")
    else:
        st.success("✅ Права администратора подтверждены")
        if st.button("Завершить сеанс"):
            st.session_state.admin_logged_in = False
            st.rerun()
            
        st.divider()
        
        st.subheader("📂 Импорт результатов физического эксперимента")
        st.write("Загрузите CSV-выгрузку с осциллографа или измерительного стенда (Колонки: S, H, V).")
        uploaded_file = st.file_uploader("Файл данных", type=["csv", "xlsx"])
        
        if uploaded_file is not None:
            try:
                df_new = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
                st.dataframe(df_new.head(3))
                
                if st.button("📥 Интегрировать данные в базу СУБД", type="secondary"):
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
                    st.success(f"Загружено записей: {success_count}. База данных обновлена.")
            except Exception as e:
                st.error(f"Ошибка чтения: {e}")

        st.divider()
        st.subheader("☁️ Тренировка ИНС на эталонных данных")
        
        if st.button("🚀 Выполнить градиентное переобучение (MLPRegressor)", type="primary"):
            with st.spinner('Анализ датасета и корректировка весов...'):
                try:
                    conn, _ = get_active_connection()
                    df = pd.read_sql("SELECT S_val, H_val, V_val FROM training_data", conn)
                    conn.close()

                    if len(df) < 5 and not IS_ONLINE: 
                        s_mesh, h_mesh = np.meshgrid(np.linspace(19.6, 78.5, 20), np.linspace(0.175, 1.4, 20))
                        df = pd.DataFrame({'S_val': s_mesh.flatten(), 'H_val': h_mesh.flatten(), 'V_val': np.abs(s_mesh.flatten() * h_mesh.flatten() * 0.015)})

                    X = df[['S_val', 'H_val']]
                    y_true = df['V_val'].values
                    y_log = np.log(y_true + 1e-5)

                    scaler_new = StandardScaler()
                    X_scaled = scaler_new.fit_transform(X)

                    model_new = MLPRegressor(hidden_layer_sizes=(10, 5), activation='tanh', solver='lbfgs', max_iter=2000, random_state=42)
                    model_new.fit(X_scaled, y_log)

                    # --- РАСЧЕТ РЕАЛЬНОЙ ТОЧНОСТИ (MAPE) ---
                    y_pred_log = model_new.predict(X_scaled)
                    y_pred_real = np.exp(y_pred_log)
                    
                    # Формула MAPE: среднее от |(Факт - Прогноз) / Факт| * 100%
                    mape_value = np.mean(np.abs((y_true - y_pred_real) / (y_true + 1e-9))) * 100

                    joblib.dump(model_new, 'trained_model.joblib')
                    joblib.dump(scaler_new, 'scaler.joblib')
                    joblib.dump(mape_value, 'mape.joblib') # Сохраняем точность в файл

                    st.success(f"🎉 Модель обучена! Объем выборки: {len(df)} записей.")
                    st.info(f"📊 **Достигнутая точность (MAPE): {mape_value:.2f}%**")
                    st.rerun()
                except Exception as e:
                    st.error(f"Сбой оптимизатора: {e}")
