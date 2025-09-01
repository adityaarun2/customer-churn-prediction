# Dockerfile (Streamlit UI)
FROM python:3.11-slim

# System deps (optional but useful)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# copy the whole repo
COPY . .

# train at build time so the image includes the model artifact
RUN python -m src.train --config configs/config.yaml

EXPOSE 8501
CMD ["streamlit", "run", "ui/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
