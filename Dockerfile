FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Ensure all data directories exist inside the image as fallback
RUN mkdir -p /app/data/models /app/data/raw /app/data/processed \
             /app/data/kaggle /app/data/cache /app/logs

CMD ["python", "-m", "bot.main"]
