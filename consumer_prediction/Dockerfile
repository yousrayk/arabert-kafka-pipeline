# syntax=docker/dockerfile:1
FROM python:3.12.3-slim
WORKDIR /app
# Cache mount survives across build retries — matters on a slow/flaky
# connection, where a failed install shouldn't re-download everything.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --timeout 120 --retries 10 torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --timeout 120 --retries 10 -r requirements.txt
COPY . .
CMD ["python", "-m", "app.main"]
