# syntax=docker/dockerfile:1
FROM python:3.12.3-slim
WORKDIR /app
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --timeout 120 --retries 10 -r requirements.txt
COPY . .
# Streamlit puts the script's own directory (app/) on sys.path, not /app,
# so `from app.data import ...` needs this.
ENV PYTHONPATH=/app
CMD ["streamlit", "run", "app/main.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
