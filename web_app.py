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
st.set_page_config(page_title="САПР ЭМС: Оценка наведенной помехи", layout="wide")

# --- 2. ИНИЦИАЛИЗАЦИЯ ЛОКАЛЬНОЙ БД (SQLite) ---
def init_local_db():
    conn = sqlite3.connect('fallback_local.db')
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS training_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            S_val REAL,
            H_val REAL,
            V_val REAL,
            AddedDate TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS prediction_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            S_in REAL,
            H_in REAL,
            V_out REAL,
            Expert_Verdict TEXT,
            RequestDate TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            login TEXT,
            pass_hash TEXT
        )
    """)

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


# --- 4. ИНТЕРФЕЙС ---
st.sidebar.info("● БД: SQLite (Локальный режим)")

st.title("📡 САПР ЭМС: оценка наведенной помехи в линии связи")
st.markdown(
    "Моделирование амплитуды наведенного напряжения в воспринимающем контуре линии связи "
    "электронного средства при воздействии импульсного магнитного поля разряда молнии."
)

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
            st.info(f"🧠 Нейронная сеть активна. **Погрешность аппроксимации (MAPE): {current_mape:.2f}%**")
        else:
            st.info("🧠 Нейронная сеть активна. Точность будет рассчитана при следующем обучении.")

        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("Входные параметры эксперимента")

            s_val = st.number_input(
                "Эквивалентная площадь контура S (см²)",
                value=45.0,
                step=1.0
            )

            h_val = st.number_input(
                "Напряженность магнитного поля H (кА/м)",
                value=0.7,
                step=0.1
            )

            st.markdown("""
            **Нормативная база расчета:**
            * **ГОСТ 30336-95 / ГОСТ Р 50649-94** — устойчивость технических средств к импульсному магнитному полю;
            * форма испытательного импульса: **6,4/16 мкс**;
            * нормируемые уровни H: **100, 300, 1000 А/м**.
            """)

            st.markdown("""
            **Площади контуров К4 и К5:**
            * К4 — S = **19,6 см²**;
            * К5 — S = **78,5 см²**.

            Нейросеть может выполнять расчет не только в исходных экспериментальных точках,
            но и для промежуточных значений внутри обученного диапазона.
            """)

            calc_btn = st.button(
                "СМОДЕЛИРОВАТЬ ВОЗДЕЙСТВИЕ",
                use_container_width=True,
                type="primary"
            )

        if calc_btn:
            X_scaled = scaler.transform([[s_val, h_val]])
            v_peak = np.exp(model.predict(X_scaled)[0])

            with col2:
                st.subheader("Протокол виртуального испытания")

                st.metric("Амплитуда наведенного напряжения", f"{v_peak:.4f} В")

                st.markdown("### Исходные параметры расчета")
                st.write(f"Эквивалентная площадь воспринимающего контура: **{s_val:.2f} см²**")
                st.write(f"Напряженность импульсного магнитного поля: **{h_val:.3f} кА/м**")

                st.markdown("### Интерпретация результата")

                if abs(s_val - 19.6) <= 1.0:
                    contour_text = "К4"
                    st.info(
                        "Расчет выполнен для параметров, близких к контуру **К4** "
                        "из экспериментальной установки."
                    )
                elif abs(s_val - 78.5) <= 2.0:
                    contour_text = "К5"
                    st.info(
                        "Расчет выполнен для параметров, близких к контуру **К5** "
                        "из экспериментальной установки."
                    )
                else:
                    contour_text = "пользовательский контур"
                    st.info(
                        "Расчет выполнен для пользовательского значения площади контура. "
                        "Для контуров К4 и К5 используются площади 19,6 см² и 78,5 см²."
                    )

                if 0.175 <= h_val <= 1.4:
                    mode_text = "интерполяция"
                    st.success(
                        "Значение H находится внутри экспериментального диапазона "
                        "0,175–1,4 кА/м. Результат относится к режиму интерполяции."
                    )
                else:
                    mode_text = "экстраполяция"
                    st.warning(
                        "Значение H находится за пределами экспериментального диапазона "
                        "0,175–1,4 кА/м. Результат является экстраполяционной оценкой."
                    )

                st.markdown(
                    "Полученное значение рассматривается как прогноз максимальной "
                    "наведенной электромагнитной помехи в воспринимающем контуре линии связи "
                    "электронного средства при воздействии импульсного магнитного поля."
                )

                # График наведенного импульса 6,4/16 мкс
                t = np.linspace(0, 60e-6, 1000)

                t_peak = 6.4e-6
                t_half = 16e-6

                x_half = t_half / t_peak
                n = np.log(0.5) / (np.log(x_half) + 1 - x_half)

                x = np.maximum(t / t_peak, 1e-12)
                v_t = v_peak * (x ** n) * np.exp(n * (1 - x))

                fig, ax = plt.subplots(figsize=(8, 3.5))
                ax.plot(t * 1e6, v_t, color='#d35400', linewidth=2)
                ax.set_title("Форма наведенного импульса ИМП 6,4/16 мкс", fontsize=10)
                ax.set_xlabel("Время (мкс)")
                ax.set_ylabel("Наведенное напряжение (В)")
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)

                # Логирование
                try:
                    db_conn, _ = get_active_connection()
                    cur = db_conn.cursor()

                    log_text = f"{contour_text}; режим: {mode_text}"

                    cur.execute(
                        """
                        INSERT INTO prediction_log
                        (S_in, H_in, V_out, Expert_Verdict, RequestDate)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            s_val,
                            h_val,
                            v_peak,
                            log_text,
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        )
                    )

                    db_conn.commit()
                    db_conn.close()

                except Exception:
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

                conn = sqlite3.connect('fallback_local.db')
                cur = conn.cursor()
                cur.execute(
                    "SELECT * FROM users WHERE login=? AND pass_hash=?",
                    (login, pwd_hash)
                )
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
        st.write("Загрузите CSV-выгрузку измерительного стенда. Обязательные колонки: S, H, V.")

        uploaded_file = st.file_uploader("Файл данных", type=["csv", "xlsx"])

        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df_new = pd.read_csv(uploaded_file)
                else:
                    df_new = pd.read_excel(uploaded_file)

                st.dataframe(df_new.head(3))

                if st.button("📥 Интегрировать данные в базу СУБД", type="secondary"):
                    conn, _ = get_active_connection()
                    cur = conn.cursor()

                    # Очищаем таблицу перед новой заливкой чистого датасета
                    cur.execute("DELETE FROM training_data")

                    success_count = 0

                    for _, row in df_new.iterrows():
                        cur.execute(
                            """
                            INSERT INTO training_data
                            (S_val, H_val, V_val, AddedDate)
                            VALUES (?, ?, ?, ?)
                            """,
                            (
                                float(row['S']),
                                float(row['H']),
                                float(row['V']),
                                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            )
                        )
                        success_count += 1

                    conn.commit()
                    conn.close()

                    st.success(
                        f"Загружено записей: {success_count}. "
                        "Локальная база данных SQLite успешно обновлена."
                    )

            except Exception as e:
                st.error(f"Ошибка чтения: {e}")

        st.divider()

        st.subheader("☁️ Тренировка ИНС на экспериментальных данных")

        if st.button("🚀 Выполнить градиентное переобучение (MLPRegressor)", type="primary"):
            with st.spinner('Анализ датасета и корректировка весов...'):
                try:
                    conn, _ = get_active_connection()
                    df = pd.read_sql(
                        "SELECT S_val, H_val, V_val FROM training_data",
                        conn
                    )
                    conn.close()

                    # Автогенерация базовой сетки, если локальная таблица пуста
                    if len(df) < 5:
                        s_mesh, h_mesh = np.meshgrid(
                            np.linspace(19.6, 78.5, 20),
                            np.linspace(0.175, 1.4, 20)
                        )

                        df = pd.DataFrame({
                            'S_val': s_mesh.flatten(),
                            'H_val': h_mesh.flatten(),
                            'V_val': np.abs(s_mesh.flatten() * h_mesh.flatten() * 0.015)
                        })

                    X = df[['S_val', 'H_val']]
                    y_true = df['V_val'].values
                    y_log = np.log(y_true + 1e-5)

                    scaler_new = StandardScaler()
                    X_scaled = scaler_new.fit_transform(X)

                    # Архитектура нейронной сети
                    model_new = MLPRegressor(
                        hidden_layer_sizes=(3,),
                        activation='tanh',
                        solver='lbfgs',
                        max_iter=2000,
                        random_state=42
                    )

                    model_new.fit(X_scaled, y_log)

                    # Вычисление MAPE на обучающей выборке
                    y_pred_log = model_new.predict(X_scaled)
                    y_pred_real = np.exp(y_pred_log)

                    mape_value = np.mean(
                        np.abs((y_true - y_pred_real) / (y_true + 1e-9))
                    ) * 100

                    joblib.dump(model_new, 'trained_model.joblib')
                    joblib.dump(scaler_new, 'scaler.joblib')
                    joblib.dump(mape_value, 'mape.joblib')

                    st.success(
                        f"🎉 Модель обучена в локальном режиме! "
                        f"Объем выборки: {len(df)} записей."
                    )

                    st.info(f"📊 **Достигнутая точность аппроксимации (MAPE): {mape_value:.2f}%**")

                    st.rerun()

                except Exception as e:
                    st.error(f"Сбой оптимизатора: {e}")
