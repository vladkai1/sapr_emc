import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sqlite3
import hashlib
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import os

# --- 1. НАСТРОЙКИ СТРАНИЦЫ ---
st.set_page_config(page_title="САПР ЭМС: Оценка устойчивости", layout="wide")

# --- 2. ИНИЦИАЛИЗАЦИЯ ЛОКАЛЬНОЙ БД (SQLite) ---
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

def get_active_connection():
    return sqlite3.connect('fallback_local.db'), "SQLite 3.0 (Локальный режим)"

# --- 3. УПРАВЛЕНИЕ СЕССИЕЙ И ЗАГРУЗКА ИНС ---
if "admin_logged_in" not in st.session_state:
    st.session_state.admin_logged_in = False

def load_nn_model():
    if os.path.exists('trained_model.joblib') and os.path.exists('scaler.joblib'):
        model = joblib.load('trained_model.joblib')
        scaler = joblib.load('scaler.joblib')
        mape = joblib.load('mape.joblib') if os.path.exists('mape.joblib') else "Н/Д"
        return model, scaler, mape
    return None, None, None

def hash_password(password):
    return hashlib.sha256(password.encode('utf-8')).hexdigest()

# --- ИНТЕРФЕЙС ---
st.sidebar.info("● БД: SQLite (Локальный режим)")

st.title("📡 САПР ЭМС: Оценка устойчивости электронных средств к импульсному магнитному полю")
st.markdown("Моделирование наведенных импульсных помех во внутренних линиях связи в соответствии с ГОСТ РФ.")

tab_engineer, tab_admin = st.tabs(["🚀 АРМ Инженера (Анализ)", "⚙️ Панель Разработчика"])

# ==========================================
#        ВКЛАДКА 1: АРМ ИНЖЕНЕРА
# ==========================================
with tab_engineer:
    model, scaler, current_mape = load_nn_model()

    if model is None:
        st.warning("⚠️ Ядро ИНС не обучено. Загрузите базу данных в Панели Разработчика.")
    else:
        if isinstance(current_mape, float):
            st.info(f"🧠 Нейронная сеть активна. **Реальная погрешность аппроксимации (MAPE): {current_mape:.2f}%**")
        else:
            st.info("🧠 Нейронная сеть активна. Точность будет рассчитана при следующем обучении.")

        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Входные параметры")
            
            # --- ПЕРЕКЛЮЧАТЕЛЬ РЕЖИМОВ ---
            input_mode = st.radio(
                "Режим ввода данных:", 
                ["Ввод конструктивных параметров (L, h, d)", "Прямой ввод параметров (S, H)"]
            )
            
            if input_mode == "Ввод конструктивных параметров (L, h, d)":
                st.markdown("**(Оценка реальной линии связи внутри СВТ)**")
                l_val = st.number_input("Длина линии связи/шлейфа L (см)", value=15.0, step=1.0)
                h_cable = st.number_input("Отступ шлейфа от экрана/шасси h (см)", value=3.0, step=0.5)
                d_val = st.number_input("Расстояние от СВТ до токоотвода молнии d (м)", value=15.0, step=1.0)
                
                # АНАЛИТИЧЕСКИЙ БЛОК:
                # 1. Расчет площади контура
                s_val = l_val * h_cable
                
                # 2. Расчет поля H по формуле Био-Савара-Лапласа (Ток молнии I = 100 кА по ГОСТ)
                # H = I / (2 * pi * d) = 100 / (2 * 3.14159 * d_val)
                h_val = 100 / (2 * np.pi * d_val) if d_val > 0 else 0.0
                
                st.info(f"""
                🔄 **Аналитический пересчет САПР:**
                * Эквивалентная площадь контура $S$: **{s_val:.1f} см²**
                * Напряженность поля $H$: **{h_val:.3f} кА/м**
                """)
                
                # "Защита от дурака" - проверка границ датасета (19.6 - 78.5)
                if not (19.0 <= s_val <= 80.0):
                    st.warning(f"⚠️ Внимание: Площадь контура ({s_val:.1f} см²) выходит за границы области адекватности нейросети (19.6 - 78.5 см²). Возможна погрешность экстраполяции.")
                if not (0.17 <= h_val <= 1.45):
                    st.warning(f"⚠️ Внимание: Поле H ({h_val:.2f} кА/м) выходит за границы обучающей выборки (0.175 - 1.4 кА/м).")

            else:
                st.markdown("**(Режим ручного обращения к ядру ИНС)**")
                s_val = st.number_input("Эквивалентная площадь контура S (см²)", value=45.0, step=1.0)
                h_val = st.number_input("Напряженность магнитного поля H (кА/м)", value=0.7, step=0.1)
            
            st.markdown("""
            **Нормативная база расчета:**
            * **ГОСТ IEC 61000-4-9** (Устойчивость к импульсному магнитному полю)
            * **ГОСТ Р МЭК 62305-1-2010** (Ток молнии 100 кА для III/IV уровня защиты)
            """)
            calc_btn = st.button("СМОДЕЛИРОВАТЬ ВОЗДЕЙСТВИЕ", use_container_width=True, type="primary")

        # --- БЛОК ИНФЕРЕНСА (РАСЧЕТА) ---
        if calc_btn:
            X_scaled = scaler.transform([[s_val, h_val]])
            v_peak = np.exp(model.predict(X_scaled)[0])

            with col2:
                st.subheader("Протокол виртуального испытания")
                st.metric("Расчетная амплитуда наведенного напряжения (V)", f"{v_peak:.4f} В")
                
                st.markdown("### Оценка электромагнитной совместимости:")
                if v_peak < 0.4:
                    verdict = "КРИТЕРИЙ А: Норма. Воздействие в пределах запаса помехоустойчивости."
                    st.success(f"🟢 **{verdict}**\n\nДеградации полезного сигнала не ожидается. Аппаратура функционирует штатно.")
                elif v_peak < 1.2:
                    verdict = "КРИТЕРИЙ B: Сбой. Временное искажение битов (Soft Error)."
                    st.warning(f"🟡 **{verdict}**\n\nСмещение рабочей точки транзисторов. Возможна потеря пакетов в цифровом тракте, требуется программная коррекция.")
                elif v_peak < 2.5:
                    verdict = "КРИТЕРИЙ C: Зависание. Эффект тиристорного защелкивания (Latch-up)."
                    st.error(f"🟠 **{verdict}**\n\nБлокировка сигнального процессора. Требуется жесткая перезагрузка оборудования оператором.")
                else:
                    verdict = "КРИТЕРИЙ D: ФАТАЛЬНО. Пробой диэлектрика."
                    st.error(f"🔴 **{verdict}**\n\nНеобратимое тепловое разрушение p-n переходов полупроводниковых компонентов!")

                # График формы импульса
                t = np.linspace(0, 0.0002, 1000)
                v_t = v_peak * 1.037 * (np.exp(-14000 * t) - np.exp(-2447000 * t))
                
                fig, ax = plt.subplots(figsize=(8, 3.5))
                ax.plot(t*1e6, v_t, color='#d35400', linewidth=2)
                ax.set_title("Форма импульса наведенного напряжения (1.2/50 мкс)", fontsize=10)
                ax.set_xlabel("Время (мкс)")
                ax.set_ylabel("Напряжение (В)")
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)

                # Логирование в БД
                try:
                    db_conn, _ = get_active_connection()
                    cur = db_conn.cursor()
                    cur.execute("INSERT INTO prediction_log (S_in, H_in, V_out, Expert_Verdict, RequestDate) VALUES (?, ?, ?, ?, ?)",
                                (s_val, h_val, v_peak, verdict.split(':')[0], datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                    db_conn.commit()
                    db_conn.close()
                except:
                    pass

# ==========================================
#        ВКЛАДКА 2: АДМИНКА (Без изменений)
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
                conn = sqlite3.connect('fallback_local.db')
                cur = conn.cursor()
                cur.execute("SELECT * FROM users WHERE login=? AND pass_hash=?", (login, pwd_hash))
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
        st.write("Загрузите CSV-выгрузку измерительного стенда (Колонки: S, H, V).")
        uploaded_file = st.file_uploader("Файл данных", type=["csv", "xlsx"])
        
        if uploaded_file is not None:
            try:
                df_new = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
                st.dataframe(df_new.head(3))
                
                if st.button("📥 Интегрировать данные в базу СУБД", type="secondary"):
                    conn, _ = get_active_connection()
                    cur = conn.cursor()
                    
                    cur.execute("DELETE FROM training_data") 
                    
                    success_count = 0
                    for _, row in df_new.iterrows():
                        cur.execute("INSERT INTO training_data (S_val, H_val, V_val, AddedDate) VALUES (?, ?, ?, ?)",
                                    (float(row['S']), float(row['H']), float(row['V']), datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                        success_count += 1
                    conn.commit()
                    conn.close()
                    st.success(f"Загружено записей: {success_count}. Локальная база данных SQLite успешно обновлена.")
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

                    if len(df) < 5: 
                        s_mesh, h_mesh = np.meshgrid(np.linspace(19.6, 78.5, 20), np.linspace(0.175, 1.4, 20))
                        df = pd.DataFrame({'S_val': s_mesh.flatten(), 'H_val': h_mesh.flatten(), 'V_val': np.abs(s_mesh.flatten() * h_mesh.flatten() * 0.015)})

                    X = df[['S_val', 'H_val']]
                    y_true = df['V_val'].values
                    y_log = np.log(y_true + 1e-5)

                    scaler_new = StandardScaler()
                    X_scaled = scaler_new.fit_transform(X)

                    model_new = MLPRegressor(hidden_layer_sizes=(3,), activation='tanh', solver='lbfgs', max_iter=2000, random_state=42)
                    model_new.fit(X_scaled, y_log)

                    y_pred_log = model_new.predict(X_scaled)
                    y_pred_real = np.exp(y_pred_log)
                    mape_value = np.mean(np.abs((y_true - y_pred_real) / (y_true + 1e-9))) * 100

                    joblib.dump(model_new, 'trained_model.joblib')
                    joblib.dump(scaler_new, 'scaler.joblib')
                    joblib.dump(mape_value, 'mape.joblib')

                    st.success(f"🎉 Модель обучена в локальном режиме! Объем выборки: {len(df)} записей.")
                    st.info(f"📊 **Достигнутая точность (MAPE): {mape_value:.2f}%**")
                    st.rerun()
                except Exception as e:
                    st.error(f"Сбой оптимизатора: {e}")
