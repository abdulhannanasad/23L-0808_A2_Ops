FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /srv
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY app.py preprocess.py ./
COPY model ./model
RUN useradd -m appuser && chown -R appuser /srv
USER appuser
# single worker: the model is loaded once and RAM is small on a free instance
CMD ["sh", "-c", "gunicorn -w 1 --threads 2 -b 0.0.0.0:${PORT:-10000} --timeout 120 app:app"]
