FROM nvidia/cuda:12.1.0-base-ubuntu22.04 

RUN apt-get update -y \
    && apt-get install -y python3-pip

RUN ldconfig /usr/local/cuda-12.1/compat/

# Install Python dependencies
COPY builder/requirements.txt /requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --upgrade pip && \
    python3 -m pip install --upgrade -r /requirements.txt

# Install vLLM (switching back to pip installs since issues that required building fork are fixed and space optimization is not as important since caching) and FlashInfer 
RUN python3 -m pip install vllm==0.9.1 && \
    python3 -m pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.3

# Setup for Option 2: Building the Image with the Model included
ARG MODEL_NAME=""
ARG TOKENIZER_NAME=""
ARG BASE_PATH="/runpod-volume"
ARG QUANTIZATION=""
ARG MODEL_REVISION=""
ARG TOKENIZER_REVISION=""

ENV MODEL_NAME=$MODEL_NAME \
    MODEL_REVISION=$MODEL_REVISION \
    TOKENIZER_NAME=$TOKENIZER_NAME \
    TOKENIZER_REVISION=$TOKENIZER_REVISION \
    BASE_PATH=$BASE_PATH \
    QUANTIZATION=$QUANTIZATION \
    HF_DATASETS_CACHE="${BASE_PATH}/huggingface-cache/datasets" \
    HUGGINGFACE_HUB_CACHE="${BASE_PATH}/huggingface-cache/hub" \
    HF_HOME="${BASE_PATH}/huggingface-cache/hub" \
    HF_HUB_ENABLE_HF_TRANSFER=0 

ENV PYTHONPATH="/:/vllm-workspace"

# Copy only src/download_model.py and src/utils.py to avoid copying the entire src directory
COPY src/download_model.py /src/download_model.py
COPY src/utils.py /src/utils.py
RUN --mount=type=secret,id=HF_TOKEN,required=false \
    if [ -f /run/secrets/HF_TOKEN ]; then \
    export HF_TOKEN=$(cat /run/secrets/HF_TOKEN); \
    fi && \
    if [ -n "$MODEL_NAME" ]; then \
    python3 /src/download_model.py; \
    fi

# Install FlashAttention
# Download from "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.0.post2/flash_attn-2.8.0.post2+cu12torch2.7cxx11abiTRUE-cp310-cp310-linux_x86_64.whl"
RUN apt-get update -y \
    && apt-get install -y python3-pip curl
RUN curl -L "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.0.post2/flash_attn-2.8.0.post2+cu12torch2.7cxx11abiTRUE-cp310-cp310-linux_x86_64.whl" -o /tmp/flash_attn-2.8.0.post2+cu12torch2.7cxx11abiTRUE-cp310-cp310-linux_x86_64.whl
RUN python3 -m pip install "/tmp/flash_attn-2.8.0.post2+cu12torch2.7cxx11abiTRUE-cp310-cp310-linux_x86_64.whl" --no-build-isolation

# Patch vllm entrypoint to import custom DotsOCR modeling code
RUN sed -i '/^from vllm\.entrypoints\.cli\.main import main/a from src.dotsocr import modeling_dots_ocr_vllm' $(which vllm)
RUN sed -i '/^from vllm\.entrypoints\.cli\.main import main/a from src.dotsocr.modeling_dots_ocr_vllm import DotsOCRForCausalLM' $(which vllm)

# Copy the rest of the src directory
COPY src /src

# Start the handler
# For testing
# COPY test_input.json /test_input.json
# CMD ["python3", "/src/handler.py", "--rp_serve_api", "--rp_api_host",  "0.0.0.0", "--rp_log_level", "INFO"]

# For production
CMD ["python3", "/src/handler.py"]
