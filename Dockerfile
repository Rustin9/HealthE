FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy all project files (app.py, backend/, models/, data/, tests, etc.)
COPY . .

ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8501 \
    PYTHONUNBUFFERED=1

# open both ports (frontend + backend)
EXPOSE 8501 8000

# default: run frontend (can be overridden in docker-compose)
CMD ["streamlit", "run", "app.py"]
