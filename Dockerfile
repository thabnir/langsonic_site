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
ENV VIRTUAL_HOST langsonic.com
ENV LETSENCRYPT_HOST langsonic.com


CMD ["python", "app.py"]
# CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5678", "app:app"]

