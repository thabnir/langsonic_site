FROM python:3.11.6-slim

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /app


# Dependencies
RUN apt-get update && \
    apt-get install -y libsndfile1 ffmpeg && \
    rm -rf /var/lib/apt/lists/*

COPY . /app

EXPOSE 5678

ENV NAME World


CMD ["python", "app.py"]
