FROM python:3.14-slim

WORKDIR /app

COPY requirements.txt .
# sentence-transformers (the reranker) pulls in torch; installing the
# CPU-only build first (this service never uses a GPU) avoids pip also
# downloading torch's multi-gigabyte CUDA/cuDNN dependencies.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
