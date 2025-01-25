# docker/model_server/Dockerfile

FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV MODEL_NAME=""
ENV GPU_CONFIG=""
ENV MEMORY_LIMIT=""

CMD ["python", "-m", "model_server.main"]