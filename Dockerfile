FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt celery redis
COPY src/ src/
COPY scripts/ scripts/
COPY data/samples/ data/samples/
ENV PYTHONPATH=/app/src
CMD ["streamlit", "run", "src/docproc/review_app.py", "--server.address=0.0.0.0"]
