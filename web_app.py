import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sqlite3
import hashlib
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from datetime import datetime
import os


# ============================================================
# 1. НАСТРОЙКИ СТРАНИЦЫ
# ============================================================

st.set_page_config(
    page_title="САПР ЭМС: моделирование помех в К4 и К5",
    layout="wide"
)


# ============================================================
# 2. ИНИЦИАЛИЗАЦИЯ ЛОКАЛЬНОЙ БД
# ============================================================

DB_NAME = "fallback_local.db"

MODEL_FILE = "trained_model.joblib"
SCALER_FILE = "scaler.joblib"
MAPE_FILE = "mape.joblib"


def init_local_db():
    conn = sqlite3.connect(DB_NAME)
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
            ContourName TEXT,
            S_in REAL,
            H_in REAL,
            V_out REAL,
            Expert_Verdict TEXT,
            RequestDate TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            login TEXT UNIQUE,
            pass_hash TEXT
        )
    """)

    cur.execute("SELECT * FROM users WHERE login='admin'")
    if not cur.fetchone():
        default_password = os.getenv("APP_ADMIN_PASSWORD", "1234")
        admin_hash = hashlib.sha256(default_password.encode("utf-8")).hexdigest()
        cur.execute(
            "INSERT INTO users (login, pass_hash) VALUES (?, ?)",
            ("admin", admin_hash)
        )

    conn.commit()
    conn.close()


def get_active_connection():
    return sqlite3.connect(DB_NAME), "SQLite 3.0 — локальный режим"


init_local_db()


# ============================================================
# 3. СЛУЖЕБНЫЕ ФУНКЦИИ
# ============================================================

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def load_nn_model():
    if os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE):
        model = joblib.load(MODEL_FILE)
        scaler = joblib.load(SCALER_FILE)
        mape = joblib.load(MAPE_FILE) if os.path.exists(MAPE_FILE) else None
        return model, scaler, mape

    return None, None, None


def impulse_6416(t: np.ndarray, v_peak: float) -> np.ndarray:
    """
    Упрощенная нормированная форма наведенного импульса 6,4/16 мкс.

    Принято:
    - максимум импульса около 6,4 мкс;
    - к 16 мкс значение спадает примерно до 50 % от максимума.

    Это не полная физическая модель тока молнии, а визуализация формы
    наведенного напряжения для протокола виртуального испытания.
    """
    t_peak = 6.4e-6
    t_half = 16e-6

    x_half = t_half / t_peak
    n = np.log(0.5) / (np.log(x_half) + 1 - x_half)

    x = np.maximum(t / t_peak, 1e-12)
    shape = (x ** n) * np.exp(n * (1 - x))

    return v_peak * shape


def normalize_uploaded_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Приведение загруженной таблицы к формату:
    S — площадь контура, см²;
    H — напряженность магнитного поля, кА/м;
    V — максимальное наведенное напряжение, В.
    """
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    required_cols = {"S", "H", "V"}
    if not required_cols.issubset(set(df.columns)):
        raise ValueError("В файле должны быть колонки: S, H, V")

    df = df[["S", "H", "V"]]
    df = df.rename(columns={"S": "S_val", "H": "H_val", "V": "V_val"})

    df["S_val"] = pd.to_numeric(df["S_val"], errors="coerce")
    df["H_val"] = pd.to_numeric(df["H_val"], errors="coerce")
    df["V_val"] = pd.to_numeric(df["V_val"], errors="coerce")

    df = df.dropna()

    if len(df) == 0:
        raise ValueError("После очистки данных не осталось корректных строк.")

    if (df["S_val"] <= 0).any():
        raise ValueError("Площадь S должна быть больше нуля.")

    if (df["H_val"] <= 0).any():
        raise ValueError("Напряженность H должна быть больше нуля.")

    if (df["V_val"] <= 0).any():
        raise ValueError("Наведенное напряжение V должно быть больше нуля.")

    return df


def get_contour_area(contour_name: str) -> float:
    """
    Площади контуров К4 и К5, рассчитанные по диаметрам:
    К4 — Ø50 мм;
    К5 — Ø100 мм.

    Возвращается площадь в см².
    """
    if contour_name.startswith("К4"):
        return 19.6

    if contour_name.startswith("К5"):
        return 78.5

    raise ValueError("Неизвестный контур.")


def make_prediction(model, scaler, s_val: float, h_val: float) -> float:
    X_input = pd.DataFrame(
        [[s_val, h_val]],
        columns=["S_val", "H_val"]
    )

    X_scaled = scaler.transform(X_input)
    pred_log = model.predict(X_scaled)[0]

    v_peak = np.expm1(pred_log)
    v_peak = max(float(v_peak), 0.0)

    return v_peak


def get_expert_verdict(v_peak: float, v_dop: float) -> tuple[str, str]:
    """
    Оценка результата относительно допустимого уровня Uдоп.
    Это корректнее, чем жестко задавать универсальные пороги для всех ЭС.
    """
    if v_peak <= 0.5 * v_dop:
        return (
            "КРИТЕРИЙ А",
            "Норма. Наведенная помеха находится в безопасной зоне относительно допустимого уровня."
        )

    if v_peak <= v_dop:
        return (
            "КРИТЕРИЙ B",
            "Возможное временное ухудшение качества функционирования без повреждения аппаратуры."
        )

    if v_peak <= 2.0 * v_dop:
        return (
            "КРИТЕРИЙ C",
            "Вероятно нарушение функционирования. Для восстановления может потребоваться перезапуск или вмешательство оператора."
        )

    return (
        "КРИТЕРИЙ D",
        "Высокий риск отказа или повреждения. Наведенное напряжение существенно превышает допустимый уровень."
    )


# ============================================================
# 4. СОСТОЯНИЕ СЕССИИ
# ============================================================

if "admin_logged_in" not in st.session_state:
    st.session_state.admin_logged_in = False


# ============================================================
# 5. ИНТЕРФЕЙС
# ============================================================

st.sidebar.info("● БД: SQLite — локальный режим")

st.title("📡 САПР ЭМС: моделирование помех в контурах К4 и К5")
st.markdown(
    "Прогнозирование наведенного напряжения в контурах линий связи электронного средства "
    "при воздействии импульсного магнитного поля разряда молнии на основе искусственной нейронной сети."
)

tab_engineer, tab_admin = st.tabs(
    ["🚀 АРМ инженера", "⚙️ Панель разработчика"]
)


# ============================================================
# 6. ВКЛАДКА ИНЖЕНЕРА
# ============================================================

with tab_engineer:
    model, scaler, current_mape = load_nn_model()

    if model is None:
        st.warning(
            "⚠️ Ядро ИНС не обучено. Загрузите экспериментальные данные "
            "в панели разработчика и выполните обучение модели."
        )

    else:
        if isinstance(current_mape, (float, int, np.floating)):
            st.info(f"🧠 Нейронная сеть активна. Проверочная погрешность MAPE: **{current_mape:.2f}%**")
        else:
            st.info("🧠 Нейронная сеть активна.")

        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("Входные параметры виртуального испытания")

            contour_name = st.selectbox(
                "Исследуемый контур",
                ["К4 — Ø50 мм", "К5 — Ø100 мм"]
            )

            s_val = get_contour_area(contour_name)

            st.info(
                f"Эквивалентная площадь выбранного контура: **{s_val:.1f} см²**"
            )

            h_val = st.slider(
                "Напряженность импульсного магнитного поля H, кА/м",
                min_value=0.10,
                max_value=1.60,
                value=0.70,
                step=0.025
            )

            if 0.175 <= h_val <= 1.4:
                st.success(
                    "Режим интерполяции: значение H находится внутри экспериментального диапазона 0,175–1,4 кА/м."
                )
            else:
                st.warning(
                    "Режим слабой экстраполяции: значение H находится за пределами экспериментального диапазона 0,175–1,4 кА/м."
                )

            st.markdown("""
            **Опорные экспериментальные значения H:**  
            0,175; 0,35; 0,7; 1,4 кА/м.

            **Нормативный уровень ГОСТ:**  
            1,0 кА/м = 1000 А/м.

            **Форма импульсного магнитного поля:**  
            6,4/16 мкс.
            """)

            v_dop = st.number_input(
                "Допустимый уровень наведенного напряжения Uдоп, В",
                min_value=0.01,
                value=1.0,
                step=0.1
            )

            st.markdown("""
            **Нормативная база:**
            - ГОСТ 30336-95 / ГОСТ Р 50649-94;
            - испытание на устойчивость к импульсному магнитному полю;
            - воздействие формируется индукционной катушкой;
            - нормируемые уровни H: 100, 300, 1000 А/м.
            """)

            calc_btn = st.button(
                "Смоделировать воздействие",
                use_container_width=True,
                type="primary"
            )

        if calc_btn:
            v_peak = make_prediction(model, scaler, s_val, h_val)
            verdict_code, verdict_text = get_expert_verdict(v_peak, v_dop)

            with col2:
                st.subheader("Протокол виртуального испытания")

                c1, c2, c3 = st.columns(3)
                c1.metric("Контур", contour_name.split(" — ")[0])
                c2.metric("H", f"{h_val:.3f} кА/м")
                c3.metric("S", f"{s_val:.1f} см²")

                st.metric(
                    "Прогнозируемая амплитуда наведенного напряжения",
                    f"{v_peak:.4f} В"
                )

                st.markdown("### Оценка электромагнитной совместимости")

                if verdict_code == "КРИТЕРИЙ А":
                    st.success(f"🟢 **{verdict_code}:** {verdict_text}")
                elif verdict_code == "КРИТЕРИЙ B":
                    st.warning(f"🟡 **{verdict_code}:** {verdict_text}")
                elif verdict_code == "КРИТЕРИЙ C":
                    st.error(f"🟠 **{verdict_code}:** {verdict_text}")
                else:
                    st.error(f"🔴 **{verdict_code}:** {verdict_text}")

                # График формы наведенного импульса
                t = np.linspace(0, 60e-6, 1000)
                v_t = impulse_6416(t, v_peak)

                fig, ax = plt.subplots(figsize=(8, 3.5))
                ax.plot(t * 1e6, v_t, linewidth=2)
                ax.set_title(
                    "Форма наведенного импульса при воздействии ИМП 6,4/16 мкс",
                    fontsize=10
                )
                ax.set_xlabel("Время, мкс")
                ax.set_ylabel("Наведенное напряжение, В")
                ax.grid(True, alpha=0.3)

                st.pyplot(fig)

                # Логирование результата
                try:
                    db_conn, _ = get_active_connection()
                    cur = db_conn.cursor()

                    cur.execute("""
                        INSERT INTO prediction_log
                        (ContourName, S_in, H_in, V_out, Expert_Verdict, RequestDate)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        contour_name,
                        s_val,
                        h_val,
                        v_peak,
                        verdict_code,
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    ))

                    db_conn.commit()
                    db_conn.close()

                except Exception as e:
                    st.warning(f"Результат рассчитан, но не записан в журнал: {e}")


# ============================================================
# 7. ВКЛАДКА РАЗРАБОТЧИКА
# ============================================================

with tab_admin:
    if not st.session_state.admin_logged_in:
        st.subheader("🔒 Идентификация инженера-разработчика")

        with st.form("login_form"):
            login = st.text_input("Логин")
            pwd = st.text_input("Пароль", type="password")
            submit = st.form_submit_button("Войти в систему")

            if submit:
                pwd_hash = hash_password(pwd)

                conn = sqlite3.connect(DB_NAME)
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

        st.subheader("📂 Импорт экспериментальных данных")

        st.write(
            "Загрузите таблицу с результатами измерений. "
            "Обязательные колонки: **S**, **H**, **V**."
        )

        st.markdown("""
        Где:
        - **S** — площадь контура, см²;
        - **H** — напряженность импульсного магнитного поля, кА/м;
        - **V** — максимальное наведенное напряжение, В.

        Для текущей постановки используются контуры:
        - К4: S = 19,6 см²;
        - К5: S = 78,5 см².
        """)

        uploaded_file = st.file_uploader(
            "Файл данных",
            type=["csv", "xlsx"]
        )

        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith(".csv"):
                    df_raw = pd.read_csv(uploaded_file)
                else:
                    df_raw = pd.read_excel(uploaded_file)

                df_new = normalize_uploaded_dataset(df_raw)

                st.write("Предпросмотр очищенных данных:")
                st.dataframe(df_new.head(10))

                if st.button("📥 Интегрировать данные в базу", type="secondary"):
                    conn, _ = get_active_connection()
                    cur = conn.cursor()

                    # В данной версии база заменяется новым чистым набором
                    cur.execute("DELETE FROM training_data")

                    for _, row in df_new.iterrows():
                        cur.execute("""
                            INSERT INTO training_data
                            (S_val, H_val, V_val, AddedDate)
                            VALUES (?, ?, ?, ?)
                        """, (
                            float(row["S_val"]),
                            float(row["H_val"]),
                            float(row["V_val"]),
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        ))

                    conn.commit()
                    conn.close()

                    st.success(
                        f"Загружено записей: {len(df_new)}. "
                        "Локальная база данных обновлена."
                    )

            except Exception as e:
                st.error(f"Ошибка чтения файла: {e}")

        st.divider()

        st.subheader("☁️ Обучение искусственной нейронной сети")

        st.markdown("""
        Модель обучается на дискретных экспериментальных точках, но после обучения
        работает как аппроксимирующая функция:

        \\[
        U_{нав} = f(S, H)
        \\]

        Поэтому она может прогнозировать наведенное напряжение для промежуточных
        значений H внутри диапазона 0,175–1,4 кА/м. При небольшом выходе за этот
        диапазон результат считается экстраполяционным.
        """)

        if st.button("🚀 Выполнить обучение ИНС", type="primary"):
            with st.spinner("Обработка данных и настройка весов нейронной сети..."):
                try:
                    conn, _ = get_active_connection()
                    df = pd.read_sql(
                        "SELECT S_val, H_val, V_val FROM training_data",
                        conn
                    )
                    conn.close()

                    if len(df) < 6:
                        st.error(
                            "Недостаточно экспериментальных данных для обучения. "
                            "Загрузите минимум 6 строк, лучше — данные для К4 и К5 "
                            "при нескольких значениях H."
                        )
                        st.stop()

                    if (df["V_val"] <= 0).any():
                        st.error("Все значения V должны быть больше нуля.")
                        st.stop()

                    X = df[["S_val", "H_val"]]
                    y_real = df["V_val"].values
                    y_log = np.log1p(y_real)

                    # Оценка качества на проверочной выборке
                    if len(df) >= 8:
                        X_train, X_test, y_train, y_test = train_test_split(
                            X,
                            y_log,
                            test_size=0.25,
                            random_state=42
                        )

                        scaler_eval = StandardScaler()
                        X_train_scaled = scaler_eval.fit_transform(X_train)
                        X_test_scaled = scaler_eval.transform(X_test)

                        model_eval = MLPRegressor(
                            hidden_layer_sizes=(5, 3),
                            activation="tanh",
                            solver="lbfgs",
                            alpha=0.001,
                            max_iter=3000,
                            random_state=42
                        )

                        model_eval.fit(X_train_scaled, y_train)

                        y_pred_log = model_eval.predict(X_test_scaled)
                        y_pred_real = np.expm1(y_pred_log)
                        y_test_real = np.expm1(y_test)

                        y_pred_real = np.maximum(y_pred_real, 0)

                        mape_value = np.mean(
                            np.abs((y_test_real - y_pred_real) / (y_test_real + 1e-9))
                        ) * 100

                        mape_note = "MAPE рассчитана на проверочной выборке."

                    else:
                        mape_value = None
                        mape_note = (
                            "Данных мало для корректной проверочной выборки. "
                            "Модель будет обучена, но MAPE не рассчитана."
                        )

                    # Итоговая модель обучается на всех доступных данных
                    scaler_new = StandardScaler()
                    X_scaled_all = scaler_new.fit_transform(X)

                    model_new = MLPRegressor(
                        hidden_layer_sizes=(5, 3),
                        activation="tanh",
                        solver="lbfgs",
                        alpha=0.001,
                        max_iter=3000,
                        random_state=42
                    )

                    model_new.fit(X_scaled_all, y_log)

                    joblib.dump(model_new, MODEL_FILE)
                    joblib.dump(scaler_new, SCALER_FILE)

                    if mape_value is not None:
                        joblib.dump(float(mape_value), MAPE_FILE)
                    elif os.path.exists(MAPE_FILE):
                        os.remove(MAPE_FILE)

                    st.success(
                        f"🎉 Модель обучена. Объем обучающей выборки: {len(df)} записей."
                    )

                    if mape_value is not None:
                        st.info(f"📊 {mape_note} Значение MAPE: **{mape_value:.2f}%**")
                    else:
                        st.warning(mape_note)

                    st.rerun()

                except Exception as e:
                    st.error(f"Сбой обучения модели: {e}")

        st.divider()

        st.subheader("📘 Просмотр обучающей базы")

        if st.button("Показать текущие данные"):
            conn, _ = get_active_connection()
            df_show = pd.read_sql(
                "SELECT S_val, H_val, V_val, AddedDate FROM training_data",
                conn
            )
            conn.close()

            if len(df_show) == 0:
                st.warning("Обучающая база пока пуста.")
            else:
                st.dataframe(df_show)
