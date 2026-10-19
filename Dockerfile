FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /srv
# dependencies first: this layer is cached until requirements change
COPY requirements-app.txt .
RUN pip install -r requirements-app.txt
COPY src ./src
COPY app ./app
COPY api ./api
RUN useradd -m appuser && chown -R appuser /srv
USER appuser
EXPOSE 5000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD python -c "import urllib.request as u;u.urlopen('http://localhost:5000/health')"
CMD ["python", "-m", "app.app"]
