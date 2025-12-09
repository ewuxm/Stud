# Запуск FastAPI бэкенда с uvicorn
# Убедитесь, что в каталоге лежат:
# - scaler.pkl
# - selectedfeatures.pkl
# - modelmetadata.json
# - bestexampredictor.keras или bestexampredictor.pkl

param(
    [int]$Port = 8000
)

Write-Host "Starting backend on http://127.0.0.1:$Port ..." -ForegroundColor Cyan

python -m uvicorn server:app --reload --port $Port

