FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl wget ca-certificates build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY . .

RUN pip install --no-cache-dir vllm==0.15.1 rapidfuzz
RUN pip install --no-cache-dir -e ".[qwen,ocrbench,table_parsing]"

# NLTK data required by OCRBench v2
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('wordnet')"

ENV HF_HOME=/data/cache_dir/huggingface/

ENTRYPOINT ["python", "-m", "lmms_eval"]
