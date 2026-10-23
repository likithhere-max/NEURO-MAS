# NEURO-MAS API (FastAPI). Build context = backend/.  Works on Render, Railway, Fly.io, Cloud Run, Hugging Face Spaces (Docker).
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /srv

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app
COPY ml ./ml
COPY data ./data
COPY pytest.ini ./pytest.ini

# Hosts inject $PORT (Render, Railway, Cloud Run). 8000 is the local default.
EXPOSE 8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers"]
