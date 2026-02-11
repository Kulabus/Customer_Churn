import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt


# Загрузка модели и вспомогательных файлов
@st.cache_resource
def load_model():
    model = joblib.load('models/model.pkl')
    ohe_scaler = joblib.load('models/ohe_scaler.pkl')
    mms_scaler = joblib.load('models/mms_scaler.pkl')
    return model, ohe_scaler, mms_scaler

# Интерфейс Streamlit
st.set_page_config(page_title="Предсказание оттока клиентов", layout="wide")

model, ohe_scaler, mms_scaler = load_model()


st.title("📊 Система предсказания оттока клиентов")
st.markdown("---")

st.sidebar.header("📝 Ввод параметров клиента")

# 1. Tenure (время обслуживания в месяцах)
tenure = st.sidebar.slider(
    "Время обслуживания (месяцев)",
    min_value=0,
    max_value=72,
    value=12,
    help="Сколько месяцев клиент пользуется услугами"
)

# 2. Monthly Charges (ежемесячный платеж)
monthly_charges = st.sidebar.number_input(
    "Ежемесячный платеж ($)",
    min_value=0.0,
    max_value=200.0,
    value=50.0,
    step=5.0,
    help="Сумма ежемесячного платежа"
)

# 3. Total Charges (общая сумма платежей)
total_charges = st.sidebar.number_input(
    "Общая сумма платежей ($)",
    min_value=0.0,
    max_value=10000.0,
    value=1000.0,
    step=100.0,
    help="Общая сумма, уплаченная клиентом"
)

# 4. Contract Type (тип контракта)
contract = st.sidebar.selectbox(
    "Тип контракта",
    ["Month-to-month", "One year", "Two year"],
    help="Срок действия контракта"
)

# 5. Internet Service (тип интернет-услуг)
payment_method = st.sidebar.selectbox(
    "Способ оплаты",
    ['Bank transfer (automatic)',
     'Credit card (automatic)',
     'Electronic check',
     'Mailed check'],
    help="Способ оплаты интернет-услуг"
)

# Кнопка предсказания
if st.sidebar.button("🔮 Предсказать вероятность оттока", type="primary"):

    ohe_feats = ohe_scaler.transform([[contract, payment_method]]).toarray()
    mms_feats = mms_scaler.transform([[tenure, monthly_charges, total_charges]])
    features = np.concatenate((ohe_feats[0], mms_feats[0]))
    # result = model.predict(features.reshape(1, -1))

    # Предсказание
    prediction = model.predict(features.reshape(1, -1))[0]
    probability = model.predict_proba(features.reshape(1, -1))[0][1]

    # Отображение результатов
    st.markdown("---")
    st.subheader("📈 Результаты предсказания")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="Вероятность оттока",
            value=f"{probability:.1%}",
            delta=f"{(probability - 0.5):.1%}" if probability != 0.5 else "0%",
            delta_color="inverse"
        )

    with col2:
        status = "Высокий риск" if probability > 0.7 else "Средний риск" if probability > 0.4 else "Низкий риск"
        color = "🔴" if probability > 0.7 else "🟡" if probability > 0.4 else "🟢"
        st.metric(label="Уровень риска", value=f"{color} {status}")

    with col3:
        action = "Требуется удержание!" if prediction == 1 else "Клиент стабилен"
        st.metric(label="Рекомендация", value=action)

    # Визуализация
    st.markdown("---")
    st.subheader("📊 Визуализация вероятности")

    fig, ax = plt.subplots(figsize=(10, 2))
    ax.barh(['Вероятность оттока'], [probability], color='red' if probability > 0.5 else 'green')
    ax.barh(['Вероятность оттока'], [1 - probability], left=[probability],
            color='green' if probability > 0.5 else 'lightgray')
    ax.set_xlim(0, 1)
    ax.set_xlabel('Вероятность')
    ax.axvline(x=0.5, color='black', linestyle='--', alpha=0.5)
    st.pyplot(fig)

    # Рекомендации
    st.markdown("---")
    st.subheader("💡 Рекомендации")

    if probability > 0.7:
        st.warning("""
        **Высокий риск оттока! Рекомендуемые действия:**
        - Связаться с клиентом для выяснения причин недовольства
        - Предложить специальные условия или скидку
        - Провести опрос удовлетворенности
        - Назначить персонального менеджера
        """)
    elif probability > 0.4:
        st.info("""
        **Средний риск оттока. Рекомендуемые действия:**
        - Мониторинг активности клиента
        - Предложить дополнительные услуги
        - Проверить качество обслуживания
        - Отправить персонализированное предложение
        """)
    else:
        st.success("""
        **Низкий риск оттока. Рекомендуемые действия:**
        - Продолжать стандартное обслуживание
        - Предложить программы лояльности
        - Информировать о новых услугах
        - Поддерживать регулярный контакт
        """)

else:
    st.info("👈 Введите параметры клиента в боковой панели и нажмите кнопку 'Предсказать вероятность оттока'")

    # Пример данных
    st.markdown("---")
    st.subheader("📋 Пример типичных случаев")

    examples = pd.DataFrame({
        "Сценарий": ["Рисковый клиент", "Стабильный клиент", "Новый клиент"],
        "Время обслуживания (мес)": [1, 36, 3],
        "Ежемесячный платеж ($)": [90, 60, 70],
        "Общая сумма ($)": [90, 2160, 210],
        "Тип контракта": ["Month-to-month", "Two year", "Month-to-month"],
        "Способ оплаты": ["Electronic check", "Credit card (automatic)", "Mailed check"],
        "Ожидаемый риск": ["Высокий", "Низкий", "Средний"]
    })

    st.table(examples)

# Информация о модели
with st.sidebar.expander("ℹ️ О модели"):
    st.write("""
    **Модель:** Categorial Boosting Classifier
    **Точность (ROC-AUC):** 0.87
    **Используемые признаки:**
    1. Время обслуживания (tenure)
    2. Ежемесячный платеж
    3. Общая сумма платежей
    4. Тип контракта
    5. Способ оплаты
    """)