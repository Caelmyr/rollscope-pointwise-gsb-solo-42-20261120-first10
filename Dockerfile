FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 ROLLSCOPE_DATA_DIR=/var/lib/rollscope HOST=0.0.0.0 PORT=8000
WORKDIR /opt/rollscope
COPY requirements.txt .
RUN python -m pip install --no-cache-dir --only-binary=:all: -r requirements.txt
COPY rollscope ./rollscope
COPY static ./static
COPY templates ./templates
COPY app.py .
RUN useradd --create-home --uid 11450 rollscope && install -d -o rollscope -g rollscope /var/lib/rollscope
USER rollscope
EXPOSE 8000
CMD ["python", "app.py"]
