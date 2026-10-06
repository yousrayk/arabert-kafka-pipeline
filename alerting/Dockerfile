# syntax=docker/dockerfile:1
FROM python:3.12.3-slim
WORKDIR /app
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --timeout 120 --retries 10 -r requirements.txt
COPY . .
CMD ["python", "-m", "app.main"]
