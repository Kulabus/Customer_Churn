import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.metrics import classification_report, roc_auc_score, recall_score
from imblearn.over_sampling import RandomOverSampler
from catboost import CatBoostClassifier
import joblib
import json


def load_and_preprocess_data():
    """Загрузка и предобработка данных"""
    df = pd.read_csv('../data/WA_Fn-UseC_-Telco-Customer-Churn.csv')

    # Преобразуем метрику в числовой вид, где 0 - клиент остался, 1 - клиент ушёл
    df['Churn'] = LabelEncoder().fit_transform(df['Churn'])
    # Колонка TotalCharges состоит из чисел в формате string и содержит несколько пустых строк.
    # Преобразуем числа в float и заменим пустые значения средними
    df['TotalCharges'] = df['TotalCharges'].apply(lambda x: None if x == ' ' else float(x))
    df['TotalCharges'].fillna(df['TotalCharges'].mean(), inplace=True)
    # Приведение названий признаков к единому формату имен
    df.rename(columns={'gender': 'Gender', 'tenure': 'Tenure'}, inplace=True)

    # Определим вектор меток
    y = df['Churn'].to_numpy()

    # Соберем количественные признаки в отдельную матрицу
    X_mms = df[['Tenure', 'MonthlyCharges', 'TotalCharges']].to_numpy()

    # Компактный список названий колонок для OHE
    cols_to_ohe_small = ['Contract', 'PaymentMethod']
    # Применяем OneHotEncoding к компактному набору признаков
    ohe_small = OneHotEncoder()
    # ohe_small.fit(df[cols_to_ohe_small])
    X_ohe_small = ohe_small.fit_transform(df[cols_to_ohe_small]).toarray()

    # Собираем категориальные и количественные признаки
    X_small = np.concatenate((X_ohe_small, X_mms), axis=1)

    # Разделим данные на обучающую и тестовую выборки
    X_train_small, X_test_small, y_train_small, y_test_small = train_test_split(
        X_small, y, test_size=0.2, random_state=43, stratify=y, shuffle=True)

    # Нормализуем количественные признаки отдельно в обучающей и тестовой выборках
    mms_small = MinMaxScaler()
    X_train_small[:, -3:] = mms_small.fit_transform(X_train_small[:, -3:])
    X_test_small[:, -3:] = mms_small.transform(X_test_small[:, -3:])

    # Семплируем тренировочную выборку
    over_sampler = RandomOverSampler(random_state=42)
    X_small_over, y_small_over = over_sampler.fit_resample(X_train_small, y_train_small)

    joblib.dump(ohe_small, '../models/ohe_scaler.pkl')
    joblib.dump(mms_small, '../models/mms_scaler.pkl')

    return X_small_over, X_test_small, y_small_over, y_test_small


# def train_model(X_train, y_train):
#     """Обучение модели с подбором гиперпараметров"""
#
#     # Параметры для GridSearch
#     param_grid = {
#         'n_estimators': [100, 200],
#         'learning_rate': [0.01, 0.1, 0.2],
#         'max_depth': [3, 5, 7],
#         'min_samples_split': [2, 5]
#     }
#
#     # Инициализация модели
#     gb = GradientBoostingClassifier(random_state=42)
#     model = XGBClassifier(n_estimators=100, learning_rate=0.1)
#
#     # GridSearch
#     grid_search = GridSearchCV(
#         gb,
#         param_grid,
#         cv=5,
#         scoring='roc_auc',
#         n_jobs=-1,
#         verbose=1
#     )
#
#     grid_search.fit(X_train, y_train)
#
#     print(f"Лучшие параметры: {grid_search.best_params_}")
#     print(f"Лучший ROC-AUC: {grid_search.best_score_:.4f}")
#
#     return grid_search.best_estimator_


def train_model(X_train, y_train):
    """Обучение модели с подбором гиперпараметров"""

    CB_model = CatBoostClassifier(
        iterations=200,
        learning_rate=0.05,
        depth=3,
        logging_level="Silent",
        random_seed=42,
    )
    CB_model.fit(X_train, y_train)

    joblib.dump(CB_model, '../models/model.pkl')

    return CB_model


def evaluate_model(model, X_test, y_test):
    """Оценка модели"""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        'recall': recall_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }

    print("\n" + "=" * 50)
    print("ОЦЕНКА МОДЕЛИ")
    print("=" * 50)
    print(f"\nRecall: {metrics['recall']:.4f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return metrics


def main():
    """Основная функция обучения"""

    # Загрузка данных
    print("Загрузка данных...")

    X_train_small, X_test_small, y_train_small, y_test_small = load_and_preprocess_data()


    # Обучение модели
    print("\nОбучение модели...")
    model = train_model(X_train_small, y_train_small)

    # Оценка модели
    metrics = evaluate_model(model, X_test_small, y_test_small)

    # Сохранение метрик
    with open('../models/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    print("\nМодель успешно обучена и сохранена!")


if __name__ == "__main__":
    main()