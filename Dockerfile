# Dockerfile (Streamlit UI)
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# (optional) if you still train in Docker, keep your training RUN step here
RUN python -m src.train --config configs/config.yaml

# Let Streamlit bind to whatever port Render sets
CMD ["bash","-lc","streamlit run ui/app.py --server.port=${PORT:-8501} --server.address=0.0.0.0"]
