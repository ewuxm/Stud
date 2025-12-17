# ============================================================
# 11. ИНТЕРАКТИВНЫЙ HTML-САЙТ С ФОРМОЙ И ГРАФИКАМИ
# ============================================================

from plotly.offline import plot
import re

# 11.1 Plotly‑графики (сравнение моделей, важность, scatter)

if 'comparison_df' not in globals():
    comparison_df = pd.DataFrame({
        'Model': list(benchmark_results.keys()),
        'MAE': [benchmark_results[m]['MAE'] for m in benchmark_results],
        'RMSE': [benchmark_results[m]['RMSE'] for m in benchmark_results],
        'R2': [benchmark_results[m]['R2'] for m in benchmark_results]
    }).sort_values('MAE')

fig_models = make_subplots(rows=1, cols=3, subplot_titles=("MAE", "RMSE", "R²"))
fig_models.add_trace(go.Bar(
    x=comparison_df['Model'],
    y=comparison_df['MAE'],
    name='MAE',
    marker=dict(color='#b39ddb')
), row=1, col=1)
fig_models.add_trace(go.Bar(
    x=comparison_df['Model'],
    y=comparison_df['RMSE'],
    name='RMSE',
    marker=dict(color='#9575cd')
), row=1, col=2)
fig_models.add_trace(go.Bar(
    x=comparison_df['Model'],
    y=comparison_df['R2'],
    name='R²',
    marker=dict(color='#7e57c2')
), row=1, col=3)
fig_models.update_layout(
    title_text="Сравнение моделей по метрикам ошибки",
    showlegend=False,
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#3a2c5f')
)

imp_top = importance_df.head(15)
fig_importance = go.Figure()
fig_importance.add_trace(
    go.Bar(
        x=imp_top['Важность'][::-1],
        y=imp_top['Признак'][::-1],
        orientation='h',
        marker=dict(
            color='#ce93d8',
            line=dict(color='#6a1b9a', width=1)
        )
    )
)
fig_importance.update_layout(
    title="Топ‑15 наиболее значимых факторов (RandomForest)",
    xaxis_title="Важность признака",
    yaxis_title="Признак",
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#3a2c5f')
)

y_pred_best = (
    best_model.predict(X_test).flatten()
    if 'keras' in str(type(best_model)).lower()
    else best_model.predict(X_test)
)
fig_scatter = go.Figure()
fig_scatter.add_trace(go.Scatter(
    x=y_test,
    y=y_pred_best,
    mode='markers',
    name='Предсказания',
    marker=dict(color='#ba68c8', size=7, opacity=0.7)
))
fig_scatter.add_trace(go.Scatter(
    x=[y_test.min(), y_test.max()],
    y=[y_test.min(), y_test.max()],
    mode='lines',
    name='Идеальная линия',
    line=dict(color='#4a148c', dash='dash', width=2)
))
fig_scatter.update_layout(
    title="Фактический vs предсказанный балл (лучшая модель)",
    xaxis_title="Фактический балл",
    yaxis_title="Предсказанный балл",
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#3a2c5f')
)

# 11.1.1 Графики обучения нейросети (history: loss, val_loss, mae, val_mae)

fig_train_loss = go.Figure()
if 'loss' in history.history:
    fig_train_loss.add_trace(go.Scatter(
        y=history.history['loss'],
        mode='lines',
        name='Train loss',
        line=dict(color='#7e57c2')
    ))
if 'val_loss' in history.history:
    fig_train_loss.add_trace(go.Scatter(
        y=history.history['val_loss'],
        mode='lines',
        name='Val loss',
        line=dict(color='#ce93d8', dash='dash')
    ))
fig_train_loss.update_layout(
    title="Динамика функции потерь (нейросеть, Huber)",
    xaxis_title="Эпоха",
    yaxis_title="Loss",
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#3a2c5f'),
    legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
)

fig_train_mae = go.Figure()
if 'mae' in history.history:
    fig_train_mae.add_trace(go.Scatter(
        y=history.history['mae'],
        mode='lines',
        name='Train MAE',
        line=dict(color='#5e35b1')
    ))
if 'val_mae' in history.history:
    fig_train_mae.add_trace(go.Scatter(
        y=history.history['val_mae'],
        mode='lines',
        name='Val MAE',
        line=dict(color='#ab47bc', dash='dash')
    ))
fig_train_mae.update_layout(
    title="MAE по эпохам (нейросеть)",
    xaxis_title="Эпоха",
    yaxis_title="MAE",
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#3a2c5f'),
    legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
)

# 11.2 Линейная регрессия для JS

lin_reg = benchmark_results['Linear Regression']['model']
lin_coef = lin_reg.coef_.tolist()
lin_intercept = float(lin_reg.intercept_)
lin_features = selected_features

lin_reg_js = {
    "coef": lin_coef,
    "intercept": lin_intercept,
    "features": lin_features
}

with open("linear_regression_for_js.json", "w") as f:
    json.dump(lin_reg_js, f, indent=4)

# 11.3 HTML (лавандовый дизайн + div‑ы под каждый график)

html_path = "exam_app.html"
simple_form_fields = [c for c in ['age', 'study_hours', 'class_attendance', 'sleep_hours'] if c in df.columns]

with open(html_path, "w", encoding="utf-8") as f:
    f.write("""
<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<title>Сайт прогноза результатов экзамена (TensorFlow + ML)</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
:root {
  --lavender-bg: #f3e5f5;
  --lavender-light: #ede7f6;
  --lavender-mid: #ce93d8;
  --lavender-dark: #4a148c;
  --text-main: #2e2440;
  --accent: #7e57c2;
}
body {
  font-family: Arial, sans-serif;
  margin: 0;
  padding: 20px;
  background: radial-gradient(circle at top left, #f8bbd0, #f3e5f5 40%, #ede7f6 80%);
}
.container {
  max-width: 1300px;
  margin: auto;
  background-color: #ffffff;
  padding: 25px 30px 35px 30px;
  border-radius: 16px;
  box-shadow: 0 14px 30px rgba(74,20,140,0.25);
  border: 1px solid rgba(126,87,194,0.2);
}
h1, h2, h3 {
  color: var(--text-main);
  font-weight: 700;
}
h1 {
  text-align: center;
  margin-bottom: 5px;
}
.subtitle {
  text-align: center;
  color: #6a4f9c;
  margin-bottom: 20px;
}
.section {
  margin-top: 30px;
  padding: 18px 20px;
  border-radius: 14px;
  background: linear-gradient(135deg, rgba(236,231,248,0.75), rgba(243,229,245,0.9));
  border: 1px solid rgba(179,157,219,0.4);
}
.section-title {
  display: flex;
  align-items: center;
  gap: 8px;
}
.section-title-badge {
  width: 26px;
  height: 26px;
  border-radius: 50%;
  background: linear-gradient(145deg, #7e57c2, #ab47bc);
  display: inline-flex;
  align-items: center;
  justify-content: center;
  color: #fff;
  font-size: 13px;
  box-shadow: 0 3px 8px rgba(74,20,140,0.35);
}
.info-box {
  background: linear-gradient(135deg, #ede7f6, #f3e5f5);
  padding: 12px 16px;
  border-radius: 10px;
  border-left: 4px solid var(--accent);
  margin-top: 10px;
}
.small-text {
  font-size: 13px;
  color: #5d5378;
}
label {
  display: block;
  margin-top: 10px;
  color: var(--text-main);
}
input[type="number"] {
  width: 140px;
  padding: 5px 7px;
  margin-left: 10px;
  border-radius: 8px;
  border: 1px solid #b39ddb;
  background-color: #faf5ff;
  color: var(--text-main);
  outline: none;
  transition: all 0.15s ease-in-out;
}
input[type="number"]:focus {
  border-color: var(--accent);
  box-shadow: 0 0 0 3px rgba(126,87,194,0.18);
  background-color: #ffffff;
}
button {
  margin-top: 15px;
  padding: 9px 20px;
  font-size: 14px;
  cursor: pointer;
  border-radius: 999px;
  border: none;
  background: linear-gradient(135deg, #7e57c2, #ab47bc);
  color: #ffffff;
  font-weight: 600;
  box-shadow: 0 6px 16px rgba(74,20,140,0.35);
  transition: transform 0.1s ease, box-shadow 0.1s ease;
}
button:hover {
  transform: translateY(-1px);
  box-shadow: 0 8px 20px rgba(74,20,140,0.45);
}
button:active {
  transform: translateY(0);
  box-shadow: 0 3px 10px rgba(74,20,140,0.3);
}
.result-box {
  font-size: 18px;
  margin-top: 14px;
  padding: 10px 14px;
  background: linear-gradient(135deg, #e8f5e9, #f1f8e9);
  border-radius: 10px;
  display: inline-block;
  border: 1px solid #c5e1a5;
}
.graph-grid {
  display: grid;
  grid-template-columns: minmax(0, 1.5fr) minmax(0, 1.5fr);
  gap: 18px;
  margin-top: 10px;
}
.advice-cards {
  display: flex;
  flex-wrap: wrap;
  gap: 14px;
  margin-top: 14px;
}
.advice-card {
  flex: 1 1 220px;
  border-radius: 12px;
  padding: 12px 14px;
}
hr {
  margin-top: 25px;
  margin-bottom: 20px;
  border: none;
  height: 1px;
  background: linear-gradient(to right, rgba(74,20,140,0), rgba(74,20,140,0.35), rgba(74,20,140,0));
}
ul { margin-top: 6px; }
</style>
</head>
<body>
<div class="container">
  <h1>Интерактивный сайт для прогноза результатов экзамена</h1>
  <div class="subtitle">Нейронная сеть TensorFlow + классические модели машинного обучения</div>

  <div class="section">
    <div class="section-title">
      <div class="section-title-badge">1</div>
      <h2>Описание проекта и нейронной сети</h2>
    </div>
    <div class="info-box">
      <p><b>Цель проекта:</b> по характеристикам студента прогнозировать его итоговый средний балл за экзамен с помощью методов машинного обучения и нейросети TensorFlow.</p>
      <p><b>Используемые модели:</b> классические регрессоры (RandomForest, Gradient Boosting, XGBoost), полносвязная нейросеть и линейная регрессия для веб‑формы.</p>
    </div>
  </div>

  <div class="section">
    <div class="section-title">
      <div class="section-title-badge">2</div>
      <h2>Ввод данных и онлайн‑прогноз</h2>
    </div>
    <p>Введите свои учебные характеристики и нажмите «Сделать прогноз». Линейная регрессия рассчитает ожидаемый средний балл прямо в браузере.</p>

    <form id="predictForm" onsubmit="return false;">
""")

    for field in simple_form_fields:
        label = {
            'age': 'Возраст (лет)',
            'study_hours': 'Часов учёбы в день',
            'class_attendance': 'Посещаемость, %',
            'sleep_hours': 'Сон, часов в сутки'
        }.get(field, field)
        f.write(
            f'<label>{label}: '
            f'<input type="number" step="0.01" id="{field}" />'
            f'</label><br/>\n'
        )

    f.write("""
      <button onclick="predictScore()">Сделать прогноз</button>
    </form>

    <div class="result-box">
      <b>Прогнозируемый средний балл:</b> <span id="predictionResult">—</span> / 100
    </div>
  </div>

  <hr/>

  <div class="section">
    <div class="section-title">
      <div class="section-title-badge">3</div>
      <h2>Сравнение моделей</h2>
    </div>
    <div id="modelsDiv"></div>
    <p class="small-text">
      Чем ниже MAE и RMSE и выше R², тем лучше модель описывает оценки студентов.
    </p>
  </div>

  <div class="section">
    <div class="section-title">
      <div class="section-title-badge">4</div>
      <h2>Важнейшие факторы (RandomForest)</h2>
    </div>
    <div id="importanceDiv"></div>
    <p class="small-text">
      Верхние столбцы показывают признаки, на которые модель особенно опирается при прогнозе.
    </p>
  </div>

  <div class="section">
    <div class="section-title">
      <div class="section-title-badge">5</div>
      <h2>Фактический vs предсказанный балл</h2>
    </div>
    <div id="scatterDiv"></div>
    <p class="small-text">
      Чем плотнее точки вокруг диагонали, тем точнее прогнозы модели для отдельных студентов.
    </p>
  </div>

  <div class="section">
    <div class="section-title">
      <div class="section-title-badge">6</div>
      <h2>Как обучалась нейросеть</h2>
    </div>
    <div class="graph-grid">
      <div>
        <div id="trainLossDiv"></div>
        <p class="small-text">
          Кривая loss показывает, как модель учится минимизировать функцию потерь.
        </p>
      </div>
      <div>
        <div id="trainMaeDiv"></div>
        <p class="small-text">
          MAE по эпохам показывает среднюю ошибку в баллах.
        </p>
      </div>
    </div>
  </div>

  <div class="section">
    <div class="section-title">
      <div class="section-title-badge">7</div>
      <h2>Как читать MAE, RMSE и R²</h2>
    </div>
    <div class="info-box">
      <p class="small-text"><b>MAE</b> — средняя абсолютная ошибка: на сколько баллов в среднем модель ошибается относительно реального результата.</p>
      <p class="small-text"><b>RMSE</b> — корень из средней квадратичной ошибки: сильнее наказывает крупные промахи, измеряется в тех же единицах, что и балл.</p>
      <p class="small-text"><b>R²</b> — доля объяснённой дисперсии: ближе к 1 — модель хорошо описывает вариацию оценок, около 0 — почти как простая средняя.</p>
    </div>
  </div>

  <div class="section">
    <div class="section-title">
      <div class="section-title-badge">8</div>
      <h2>Советы студенту на основе модели</h2>
    </div>
    <div class="advice-cards">
      <div class="advice-card" style="background: linear-gradient(135deg, #f3e5f5, #ede7f6); border: 1px solid rgba(126,87,194,0.25);">
        <b>1. Держи высокую посещаемость</b>
        <p class="small-text">Стабильная посещаемость занятий — один из ключевых факторов успеха. Старайся сохранять посещаемость не ниже 80–85&nbsp;%.</p>
      </div>
      <div class="advice-card" style="background: linear-gradient(135deg, #ede7f6, #e8eaf6); border: 1px solid rgba(179,157,219,0.4);">
        <b>2. Регулярные, а не редкие «забеги»</b>
        <p class="small-text">Несколько часов учёбы каждый день дают лучший результат, чем редкие многочасовые марафоны перед экзаменом.</p>
      </div>
      <div class="advice-card" style="background: linear-gradient(135deg, #f3e5f5, #fce4ec); border: 1px solid rgba(244,143,177,0.5);">
        <b>3. Не жертвуй сном</b>
        <p class="small-text">Недосып ухудшает концентрацию и память. Поддерживай режим сна 7–9 часов в сутки.</p>
      </div>
      <div class="advice-card" style="background: linear-gradient(135deg, #ede7f6, #e3f2fd); border: 1px solid rgba(100,181,246,0.5);">
        <b>4. Работай с «слабыми местами»</b>
        <p class="small-text">Дополнительные часы на предметы, где оценки ниже, дают больший прирост итогового балла.</p>
      </div>
    </div>
  </div>

</div>

<script>
let linRegParams = null;

async function loadLinReg() {
  try {
    const resp = await fetch('linear_regression_for_js.json');
    linRegParams = await resp.json();
  } catch (e) {
    console.error('Не удалось загрузить параметры линейной регрессии:', e);
  }
}

function predictScore() {
  if (!linRegParams) {
    alert('Модель ещё не загружена. Подожди пару секунд и попробуй снова.');
    return;
  }
  const coef = linRegParams.coef;
  const intercept = linRegParams.intercept;
  const features = linRegParams.features;

  let x = new Array(features.length).fill(0.0);
""")

    for field in simple_form_fields:
        f.write(f"""
  let val_{field} = parseFloat(document.getElementById("{field}").value);
  if (isNaN(val_{field})) val_{field} = 0.0;
  let idx_{field} = linRegParams.features.indexOf("{field}");
  if (idx_{field} >= 0) x[idx_{field}] = val_{field};
""")

    f.write("""
  let y = intercept;
  for (let i = 0; i < coef.length; i++) {
    y += coef[i] * x[i];
  }
  document.getElementById('predictionResult').innerText = y.toFixed(2);
}

// ===== 2. Вставка Plotly‑графиков =====
""")

    def only_div(html):
        m = re.search(r'(<div[^>]*>.*</div>)', html, flags=re.DOTALL)
        return m.group(1) if m else html

    div_models = only_div(plot(fig_models, output_type='div', include_plotlyjs=False))
    div_importance = only_div(plot(fig_importance, output_type='div', include_plotlyjs=False))
    div_scatter = only_div(plot(fig_scatter, output_type='div', include_plotlyjs=False))
    div_train_loss = only_div(plot(fig_train_loss, output_type='div', include_plotlyjs=False))
    div_train_mae = only_div(plot(fig_train_mae, output_type='div', include_plotlyjs=False))

    f.write(f"""
document.getElementById('modelsDiv').innerHTML = `{div_models}`;
document.getElementById('importanceDiv').innerHTML = `{div_importance}`;
document.getElementById('scatterDiv').innerHTML = `{div_scatter}`;
document.getElementById('trainLossDiv').innerHTML = `{div_train_loss}`;
document.getElementById('trainMaeDiv').innerHTML = `{div_train_mae}`;

loadLinReg();
</script>

</body>
</html>
""")

print(f"\\n✓ Интерктивный сайт сохранён в файле: {html_path}")
