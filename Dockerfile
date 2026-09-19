FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-chache-dir --upgrade pip \
 && pip install --no-chache-dir -r requirments.txt

COPY ai/ ./ai/
COPY src/ ./src/

RUN mkdir -p /app/data/lost /app/data/found
VOLUME ["/app/data1"]

RUN useradd --create-home --uid 1000 appuser \
 && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]