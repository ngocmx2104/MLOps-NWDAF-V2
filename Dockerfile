FROM python:3.14-slim

WORKDIR /app
COPY pyproject.toml ./
COPY src ./src

# Install the package with the feast extra (online lookup); torch/[lstm] is NOT
# installed -> iforest serving only (the self-contained, Docker-friendly path).
RUN pip install --no-cache-dir -e ".[feast]"

ENV MLOPS_SERVING_HOST=0.0.0.0 \
    MLOPS_SERVING_PORT=8080
EXPOSE 8080

CMD ["python", "-m", "src.serving", "serve"]
