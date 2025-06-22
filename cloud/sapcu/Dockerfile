FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-setuptools python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV AIP_HEALTH_ROUTE=/health
ENV AIP_PREDICT_ROUTE=/predict

COPY requirements.txt .

RUN pip3 install -r requirements.txt

COPY main.py .
COPY fd_coder.py .
COPY fd_config.py .
COPY fn_coder.py .
COPY fn_config.py .
COPY layers.py .
COPY utils.py .
COPY dense .

RUN chmod +x dense

RUN mkdir -p model-store
COPY combined_model.pt ./model-store/

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]