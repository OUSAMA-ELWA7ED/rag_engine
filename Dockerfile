FROM python:3.11-slim
RUN pip install --no-cache-dir uv
WORKDIR /app
COPY pyproject.toml uv.lock requirements-models.txt ./
RUN uv pip install --system -r pyproject.toml && \
    uv pip install --system --index-url https://download.pytorch.org/whl/cpu torch torchvision && \
    uv pip install --system timm opencv-python-headless numpy pillow
COPY rag_engine ./rag_engine
COPY model_engine ./model_engine
ENV PYTHONPATH=/app:/app/rag_engine MODEL_PACKAGE_DIR=/models
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
