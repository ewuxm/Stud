# ============================================================
# 0. УСТАНОВКА И ИМПОРТ БИБЛИОТЕК (COLAB)
# ============================================================
!pip install -q pandas numpy scikit-learn tensorflow matplotlib seaborn xgboost plotly

import warnings
warnings.filterwarnings('ignore')

import os
import io
import json
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Машинное обучение
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

# TensorFlow / Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.regularizers import l2

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import plot

print(f"TensorFlow version: {tf.__version__}")
print(f"Keras version: {keras.__version__}")

# ============================================================
# 1. ЗАГРУЗКА ТВОЕГО CSV-ФАЙЛА
# ============================================================
from google.colab import files

print("Загрузи свой CSV-файл с данными о студентах.")
uploaded = files.upload()
filename = list(uploaded.keys())[0]
df = pd.read_csv(io.BytesIO(uploaded[filename]))

print("=" * 80)
print("ПЕРВИЧНЫЕ ДАННЫЕ")
print("=" * 80)
print("Размер датасета:", df.shape)
print(df.head())
print(df.dtypes)

# ОЖИДАЕМАЯ СТРУКТУРА:
# - колонка student_id (уникальный ID) — если нет, создадим
# - колонка exam_score (целевой балл) — если нет, выбери сам и переименуй
# - числовые признаки: age, study_hours, class_attendance, sleep_hours и т.п.
# - категориальные признаки: gender, course, internet_access, sleep_quality, study_method, facility_rating, exam_difficulty и т.п.

if 'student_id' not in df.columns:
    df['student_id'] = np.arange(1, len(df) + 1)

# Если у тебя целевая колонка называется иначе — замени здесь:
TARGET_COL = 'exam_score'
if TARGET_COL not in df.columns:
    raise ValueError(f"В датасете нет колонки '{TARGET_COL}'. Переименуй свою целевую колонку в exam_score перед запуском.")

# ============================================================
# 2. ОСНОВНАЯ ПРЕДОБРАБОТКА
# ============================================================

def basic_preprocessing(df_input):
    """Предобработка: выбросы, новые признаки, кодирование, масштабирование."""
    print("\n" + "="*80)
    print("ПРЕДОБРАБОТКА ДАННЫХ")
    print("="*80)

    df_processed = df_input.copy()

    # Числовые колонки, которые точно есть/ожидаются (фильтруем по факту)
    numeric_cols_candidate = ['age', 'study_hours', 'class_attendance', 'sleep_hours']
    numeric_cols = [c for c in numeric_cols_candidate if c in df_processed.columns]

    print("1. Обработка выбросов...")
    for col in numeric_cols:
        Q1 = df_processed[col].quantile(0.25)
        Q3 = df_processed[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df_processed[col] = df_processed[col].clip(lower, upper)
        print(f"  {col}: clip [{lower:.2f}, {upper:.2f}]")

    # Новые признаки
    print("\n2. Создание новых признаков...")
    if {'study_hours', 'class_attendance'}.issubset(df_processed.columns):
        df_processed['study_efficiency'] = (
            df_processed['study_hours'] * df_processed['class_attendance'] / 100
        )
    else:
        df_processed['study_efficiency'] = 0.0

    if {'sleep_hours', 'class_attendance'}.issubset(df_processed.columns):
        df_processed['total_sleep_quality'] = (
            df_processed['sleep_hours'] * df_processed['class_attendance'] / 100
        )
    else:
        df_processed['total_sleep_quality'] = 0.0

    if {'study_hours', 'sleep_hours'}.issubset(df_processed.columns):
        df_processed['study_sleep_balance'] = (
            df_processed['study_hours'] / (df_processed['sleep_hours'] + 0.1)
        )
    else:
        df_processed['study_sleep_balance'] = 0.0

    print("  Создано 3 новых признака.")

    # Кодирование категориальных признаков
    print("\n3. Кодирование категориальных признаков...")
    cat_cols = df_processed.select_dtypes(include=['object']).columns.tolist()
    cat_cols = [c for c in cat_cols if c not in ['student_id', TARGET_COL]]

    binary_cols = []
    multi_cols = []
    for c in cat_cols:
        nun = df_processed[c].nunique()
        if nun == 2:
            binary_cols.append(c)
        elif nun > 2:
            multi_cols.append(c)
    print("  Бинарные:", binary_cols)
    print("  Мультиклассовые:", multi_cols)

    for c in binary_cols:
        le = LabelEncoder()
        df_processed[c] = le.fit_transform(df_processed[c])

    for c in multi_cols:
        dummies = pd.get_dummies(df_processed[c], prefix=c, drop_first=True)
        df_processed = pd.concat([df_processed.drop(columns=[c]), dummies], axis=1)

    # Масштабирование
    print("\n4. Масштабирование числовых признаков...")
    num_all = df_processed.select_dtypes(include=[np.number]).columns.tolist()
    to_scale = [c for c in num_all if c not in ['student_id', TARGET_COL]]

    scaler = None
    if to_scale:
        scaler = StandardScaler()
        df_processed[to_scale] = scaler.fit_transform(df_processed[to_scale])
        print(f"  Масштабировано признаков: {len(to_scale)}")
    else:
        print("  Нет признаков для масштабирования.")

    # Формирование X, y
    print("\n5. Формирование X, y...")
    X = df_processed.drop(columns=['student_id', TARGET_COL])
    y = df_processed[TARGET_COL]
    feature_names = X.columns.tolist()

    print("  X shape:", X.shape, "y shape:", y.shape)
    return X, y, scaler, feature_names

X, y, scaler, feature_names = basic_preprocessing(df)

# ============================================================
# 3. ВЫБОР ВАЖНЫХ ПРИЗНАКОВ (RandomForest)
# ============================================================
print("\n" + "="*80)
print("ВЫБОР ВАЖНЫХ ПРИЗНАКОВ")
print("="*80)

X_temp, _, y_temp, _ = train_test_split(X, y, test_size=0.3, random_state=42)
rf_selector = RandomForestRegressor(n_estimators=150, random_state=42)
rf_selector.fit(X_temp, y_temp)

feat_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': rf_selector.feature_importances_
}).sort_values('importance', ascending=False)

print("\nТоп-15 признаков:")
print(feat_importance.head(15).to_string(index=False))

plt.figure(figsize=(10,6))
top_feats = feat_importance.head(15)
plt.barh(range(len(top_feats)), top_feats['importance'][::-1])
plt.yticks(range(len(top_feats)), top_feats['feature'][::-1])
plt.xlabel("Важность")
plt.title("Топ-15 признаков")
plt.tight_layout()
plt.show()

N_FEATURES = min(20, len(feature_names))
selected_features = feat_importance.head(N_FEATURES)['feature'].tolist()
print(f"\nВыбрано {len(selected_features)} признаков:")
print(selected_features)

X_selected = X[selected_features]

# ============================================================
# 4. TRAIN / TEST SPLIT
# ============================================================
print("\n" + "="*80)
print("РАЗДЕЛЕНИЕ ДАННЫХ")
print("="*80)

X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, random_state=42, shuffle=True
)
print("Train:", X_train.shape, "Test:", X_test.shape)

# ============================================================
# 5. БАЗОВЫЕ МОДЕЛИ
# ============================================================
print("\n" + "="*80)
print("БАЗОВЫЕ МОДЕЛИ")
print("="*80)

benchmark_results = {}

benchmark_models = {
    'Random Forest': RandomForestRegressor(n_estimators=200, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, random_state=42),
    'XGBoost': XGBRegressor(
        n_estimators=250,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        verbosity=0
    ),
    # Простая линейная регрессия (её потом встроим прямо в HTML)
    'Linear Regression': LinearRegression()
}

for name, model in benchmark_models.items():
    print(f"\nОбучение {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    benchmark_results[name] = {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'model': model
    }
    print(f"  MAE={mae:.3f}  RMSE={rmse:.3f}  R2={r2:.4f}")

# ============================================================
# 6. НЕЙРОННАЯ СЕТЬ НА TENSORFLOW
# ============================================================

def build_optimized_nn(input_dim,
                       units_1=128,
                       units_2=64,
                       units_3=32,
                       units_4=16,
                       dropout_1=0.3,
                       dropout_2=0.3,
                       dropout_3=0.2,
                       dropout_4=0.1,
                       l2_reg=0.001,
                       lr=1e-3):
    model = Sequential(name="ExamScore_RegressionNN")
    model.add(Input(shape=(input_dim,)))

    model.add(Dense(units_1, kernel_regularizer=l2(l2_reg)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(dropout_1))

    model.add(Dense(units_2, kernel_regularizer=l2(l2_reg)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(dropout_2))

    model.add(Dense(units_3, kernel_regularizer=l2(l2_reg)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(dropout_3))

    model.add(Dense(units_4))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(dropout_4))

    model.add(Dense(1, activation='linear'))

    optimizer = Nadam(learning_rate=lr)
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.Huber(),
        metrics=['mae', 'mse']
    )
    return model

print("\n" + "="*80)
print("НЕЙРОННАЯ СЕТЬ")
print("="*80)

input_dim = X_train.shape[1]
nn_model = build_optimized_nn(
    input_dim=input_dim,
    units_1=128, units_2=64, units_3=32, units_4=16,
    dropout_1=0.25, dropout_2=0.25, dropout_3=0.2, dropout_4=0.1,
    l2_reg=0.0008, lr=5e-4
)

callbacks = [
    EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6, verbose=1),
    ModelCheckpoint('best_nn_model.keras', monitor='val_loss', save_best_only=True, verbose=0)
]

history = nn_model.fit(
    X_train, y_train,
    validation_split=0.15,
    epochs=200,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

print("\nАрхитектура нейросети:")
nn_model.summary()

# Оценка НС
y_pred_nn = nn_model.predict(X_test, verbose=0).flatten()
mae_nn = mean_absolute_error(y_test, y_pred_nn)
rmse_nn = np.sqrt(mean_squared_error(y_test, y_pred_nn))
r2_nn = r2_score(y_test, y_pred_nn)

benchmark_results['Neural Network'] = {
    'MAE': mae_nn,
    'RMSE': rmse_nn,
    'R2': r2_nn,
    'model': nn_model
}
print(f"\nNeural Network: MAE={mae_nn:.3f}  RMSE={rmse_nn:.3f}  R2={r2_nn:.4f}")

# График обучения
fig, axes = plt.subplots(1, 2, figsize=(14,5))
axes[0].plot(history.history['loss'], label='train')
axes[0].plot(history.history['val_loss'], label='val')
axes[0].set_title('Loss (Huber)')
axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(history.history['mae'], label='train')
axes[1].plot(history.history['val_mae'], label='val')
axes[1].set_title('MAE')
axes[1].legend(); axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ============================================================
# 7. АНСАМБЛЬ МОДЕЛЕЙ
# ============================================================
print("\n" + "="*80)
print("АНСАМБЛЬ МОДЕЛЕЙ")
print("="*80)

class SimpleEnsemble:
    def __init__(self, models_dict):
        self.models_dict = models_dict

    def predict(self, X):
        preds = []
        for name, info in self.models_dict.items():
            m = info['model']
            if 'keras' in str(type(m)).lower():
                p = m.predict(X, verbose=0).flatten()
            else:
                p = m.predict(X)
            preds.append(p)
        return np.mean(preds, axis=0)

ensemble_models = {
    'Neural Network': benchmark_results['Neural Network'],
    'XGBoost': benchmark_results['XGBoost'],
    'Random Forest': benchmark_results['Random Forest']
}
ensemble = SimpleEnsemble(ensemble_models)
y_pred_ens = ensemble.predict(X_test)

mae_ens = mean_absolute_error(y_test, y_pred_ens)
rmse_ens = np.sqrt(mean_squared_error(y_test, y_pred_ens))
r2_ens = r2_score(y_test, y_pred_ens)

benchmark_results['Ensemble'] = {
    'MAE': mae_ens,
    'RMSE': rmse_ens,
    'R2': r2_ens,
    'model': ensemble
}
print(f"Ensemble: MAE={mae_ens:.3f}  RMSE={rmse_ens:.3f}  R2={r2_ens:.4f}")

# ============================================================
# 8. СРАВНЕНИЕ МОДЕЛЕЙ (ТАБЛИЦА)
# ============================================================
comparison_df = pd.DataFrame({
    'Model': list(benchmark_results.keys()),
    'MAE': [benchmark_results[m]['MAE'] for m in benchmark_results],
    'RMSE': [benchmark_results[m]['RMSE'] for m in benchmark_results],
    'R2': [benchmark_results[m]['R2'] for m in benchmark_results]
}).sort_values('MAE')

print("\nСРАВНЕНИЕ МОДЕЛЕЙ:")
print(comparison_df.to_string(index=False))

best_model_name = comparison_df.iloc[0]['Model']
best_model_info = benchmark_results[best_model_name]
best_model = best_model_info['model']

print(f"\nЛУЧШАЯ МОДЕЛЬ: {best_model_name}")
print(f"MAE={best_model_info['MAE']:.3f}, RMSE={best_model_info['RMSE']:.3f}, R2={best_model_info['R2']:.4f}")

# ============================================================
# 9. ВАЖНОСТЬ ПРИЗНАКОВ (RandomForest)
# ============================================================
rf_model = benchmark_results['Random Forest']['model']
if hasattr(rf_model, 'feature_importances_'):
    importance_df = pd.DataFrame({
        'Признак': selected_features,
        'Важность': rf_model.feature_importances_
    }).sort_values('Важность', ascending=False)

    print("\nТоп-10 факторов по важности:")
    print(importance_df.head(10).to_string(index=False))

    plt.figure(figsize=(10,5))
    top10 = importance_df.head(10)
    plt.barh(range(len(top10)), top10['Важность'][::-1])
    plt.yticks(range(len(top10)), top10['Признак'][::-1])
    plt.title("Топ-10 факторов (RandomForest)")
    plt.tight_layout()
    plt.show()
else:
    importance_df = pd.DataFrame({'Признак': selected_features, 'Важность': np.nan})

# ============================================================
# 10. ПРОСТАЯ ФУНКЦИЯ ПРОГНОЗА
# ============================================================

def predict_exam_score_simple(student_data, model, df_original):
    """
    Готовит одну запись как df, конкатенирует с исходным df,
    прогоняет через basic_preprocessing и выдаёт прогноз.
    """
    try:
        student_df = pd.DataFrame([student_data])
        if 'student_id' not in student_df.columns:
            student_df['student_id'] = 999999

        # Для надёжности: если нет каких-то числовых полей, заполним средними
        for c in ['study_hours', 'class_attendance', 'sleep_hours']:
            if c not in student_df.columns:
                if c in df_original.columns:
                    student_df[c] = df_original[c].mean()
                else:
                    student_df[c] = 0.0

        # Соединяем, чтобы повторно применить всю предобработку
        df_combined = pd.concat([df_original, student_df], ignore_index=True)
        X_proc, _, _, feats_all = basic_preprocessing(df_combined)

        X_new = X_proc.iloc[[-1]]  # последняя строка — наш студент

        # Оставляем только выбранные признаки (selected_features),
        # если чего-то нет — добавим 0
        for f in selected_features:
            if f not in X_new.columns:
                X_new[f] = 0.0
        X_new = X_new[selected_features]

        # Прогноз модели
        if 'keras' in str(type(model)).lower():
            pred = model.predict(X_new, verbose=0)[0][0]
        else:
            pred = model.predict(X_new)[0]
        return float(pred)
    except Exception as e:
        print("Ошибка прогноза:", e)
        return None

# ============================================================
# 11. ИНТЕРАКТИВНЫЙ HTML-САЙТ С ФОРМОЙ И ГРАФИКАМИ
# ============================================================

# 11.1 Plotly-графики (сравнение моделей, важность, scatter)

fig_models = make_subplots(rows=1, cols=3, subplot_titles=("MAE", "RMSE", "R2"))
fig_models.add_trace(go.Bar(x=comparison_df['Model'], y=comparison_df['MAE'], name='MAE'), row=1, col=1)
fig_models.add_trace(go.Bar(x=comparison_df['Model'], y=comparison_df['RMSE'], name='RMSE'), row=1, col=2)
fig_models.add_trace(go.Bar(x=comparison_df['Model'], y=comparison_df['R2'], name='R2'), row=1, col=3)
fig_models.update_layout(title_text="Сравнение моделей", showlegend=False)

imp_top = importance_df.head(15)
fig_importance = go.Figure()
fig_importance.add_trace(
    go.Bar(
        x=imp_top['Важность'][::-1],
        y=imp_top['Признак'][::-1],
        orientation='h'
    )
)
fig_importance.update_layout(title="Топ-15 факторов (RandomForest)", xaxis_title="Важность", yaxis_title="Признак")

# для scatter используем лучшую модель
y_pred_best = (
    best_model.predict(X_test).flatten()
    if 'keras' in str(type(best_model)).lower()
    else best_model.predict(X_test)
)
fig_scatter = go.Figure()
fig_scatter.add_trace(go.Scatter(x=y_test, y=y_pred_best, mode='markers', name='Предсказания'))
fig_scatter.add_trace(go.Scatter(
    x=[y_test.min(), y_test.max()],
    y=[y_test.min(), y_test.max()],
    mode='lines',
    name='Идеальная линия',
    line=dict(color='red', dash='dash')
))
fig_scatter.update_layout(
    title="Фактический vs предсказанный балл (лучшая модель)",
    xaxis_title="Фактический балл",
    yaxis_title="Предсказанный балл"
)

# 11.2 Сохраняем коэффициенты линейной регрессии для использования в JS
lin_reg = benchmark_results['Linear Regression']['model']
lin_coef = lin_reg.coef_.tolist()
lin_intercept = float(lin_reg.intercept_)
lin_features = selected_features  # упорядоченные признаки

lin_reg_js = {
    "coef": lin_coef,
    "intercept": lin_intercept,
    "features": lin_features
}

with open("linear_regression_for_js.json", "w") as f:
    json.dump(lin_reg_js, f, indent=4)

# 11.3 Генерация HTML-приложения
# Веб-форма: пользователь вводит основные числовые признаки,
# JS формирует вектор признаков по selected_features и считает y = w^T x + b.

html_path = "exam_app.html"

# Подготовим список простых полей для формы
# Возьмём пересечение ожидаемых полей и реально существующих
simple_form_fields = [c for c in ['age', 'study_hours', 'class_attendance', 'sleep_hours'] if c in df.columns]

with open(html_path, "w", encoding="utf-8") as f:
    f.write("""
<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<title>Exam Score Prediction App</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
body { font-family: Arial, sans-serif; margin: 20px; }
.container { max-width: 1200px; margin: auto; }
input[type="number"] { width: 100px; }
label { margin-right: 10px; }
.section { margin-top: 30px; }
</style>
</head>
<body>
<div class="container">
<h1>Веб-приложение: прогноз результата экзамена</h1>
<p>Эта страница сгенерирована в Google Colab. Можно вводить параметры студента и получать прогноз средней оценки.</p>

<div class="section">
<h2>1. Форма для прогноза (линейная регрессия)</h2>
<form id="predictForm" onsubmit="return false;">
""")
    # Поля формы
    for field in simple_form_fields:
        f.write(f'<label>{field}: <input type="number" step="0.01" id="{field}" /></label><br/>\n')
    f.write("""
<br/>
<button onclick="predictScore()">Сделать прогноз</button>
</form>
<p><b>Прогнозируемый балл:</b> <span id="predictionResult">—</span></p>
</div>

<div class="section">
<h2>2. Сравнение моделей</h2>
<div id="modelsDiv"></div>
</div>

<div class="section">
<h2>3. Важнейшие факторы (RandomForest)</h2>
<div id="importanceDiv"></div>
</div>

<div class="section">
<h2>4. Фактический vs предсказанный (лучшая модель)</h2>
<div id="scatterDiv"></div>
</div>

<script>
// ===== 1. Линейная регрессия (из JSON) =====
let linRegParams = null;

async function loadLinReg() {
  const resp = await fetch('linear_regression_for_js.json');
  linRegParams = await resp.json();
}

function predictScore() {
  if (!linRegParams) {
    alert('Модель ещё не загружена. Подожди пару секунд и попробуй снова.');
    return;
  }
  const coef = linRegParams.coef;
  const intercept = linRegParams.intercept;
  const features = linRegParams.features;

  // Сформируем вектор признаков X (по порядку features).
  // Для простоты: для полей формы используем значения, остальные = 0.
  let x = new Array(features.length).fill(0.0);

""")
    # JavaScript: считывание числовых полей
    for field in simple_form_fields:
        f.write(f"""
  // Поле {field}
  let val_{field} = parseFloat(document.getElementById("{field}").value);
  if (isNaN(val_{field})) val_{field} = 0.0;
  // Находим индекс признака в массиве features и записываем значение.
  let idx_{field} = linRegParams.features.indexOf("{field}");
  if (idx_{field} >= 0) x[idx_{field}] = val_{field};
""")
    f.write("""
  // Считаем y = w^T x + b
  let y = intercept;
  for (let i = 0; i < coef.length; i++) {
    y += coef[i] * x[i];
  }
  document.getElementById('predictionResult').innerText = y.toFixed(2);
}

// ===== 2. Отрисовка Plotly-графиков =====
""")

    # Встраиваем Plotly-графики как div через plot(..., output_type='div')
    div_models = plot(fig_models, output_type='div', include_plotlyjs=False)
    div_importance = plot(fig_importance, output_type='div', include_plotlyjs=False)
    div_scatter = plot(fig_scatter, output_type='div', include_plotlyjs=False)

    # JS-код для вставки этих дивов
    f.write(f"""
document.getElementById('modelsDiv').innerHTML = `{div_models}`;
document.getElementById('importanceDiv').innerHTML = `{div_importance}`;
document.getElementById('scatterDiv').innerHTML = `{div_scatter}`;

// Загружаем параметры линейной регрессии
loadLinReg();
</script>

</div>
</body>
</html>
""")

print(f"\n✓ Интерктивный сайт сохранён в файле: {html_path}")

# ============================================================
# 12. СОХРАНЕНИЕ МОДЕЛЕЙ, МЕТАДАННЫХ И АРХИВА
# ============================================================
import joblib

if 'keras' in str(type(best_model)).lower():
    best_model.save('best_exam_predictor.keras')
else:
    joblib.dump(best_model, 'best_exam_predictor.pkl')

if scaler is not None:
    joblib.dump(scaler, 'scaler.pkl')

with open('feature_names.txt', 'w') as f:
    for feat in selected_features:
        f.write(feat + "\n")

metadata = {
    'best_model_name': best_model_name,
    'metrics': {
        'MAE': float(best_model_info['MAE']),
        'RMSE': float(best_model_info['RMSE']),
        'R2': float(best_model_info['R2'])
    },
    'features_count': len(selected_features),
    'features': selected_features,
    'data_shape': {
        'train': list(X_train.shape),
        'test': list(X_test.shape)
    }
}
with open('model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=4)

# Текстовый отчёт
final_report = f"""
ФИНАЛЬНЫЙ ОТЧЁТ ПО ПРОЕКТУ (NS на TensorFlow + ML):

Сравнение моделей:
{comparison_df.to_string(index=False)}

Лучшая модель: {best_model_name}
  MAE:  {best_model_info['MAE']:.3f}
  RMSE: {best_model_info['RMSE']:.3f}
  R2:   {best_model_info['R2']:.4f}
"""

with open('final_project_report.txt', 'w', encoding='utf-8') as f:
    f.write(final_report)

# Архив
files_to_zip = [
    'best_exam_predictor.keras',
    'best_exam_predictor.pkl',
    'scaler.pkl',
    'feature_names.txt',
    'model_metadata.json',
    'linear_regression_for_js.json',
    'exam_app.html',
    'final_project_report.txt'
]
existing_files = [fp for fp in files_to_zip if os.path.exists(fp)]

if existing_files:
    with zipfile.ZipFile('exam_prediction_project_final.zip', 'w') as z:
        for fp in existing_files:
            z.write(fp)
    print("\n✓ Архив exam_prediction_project_final.zip создан.")
else:
    print("\nФайлы для архива не найдены.")

print("\nГОТОВО. Скачивай exam_app.html или архив exam_prediction_project_final.zip через Colab.")
