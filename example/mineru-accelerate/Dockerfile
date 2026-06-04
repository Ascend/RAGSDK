FROM quay.io/ascend/vllm-ascend:v0.13.0

RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir \
    doclayout_yolo \
    ultralytics \
    mineru_vl_utils==0.1.22 \
    mineru==2.7.6 \
    numpy==2.3.0 \
    -i https://mirrors.aliyun.com/pypi/simple/ \
    --trusted-host mirrors.aliyun.com

COPY patch/vllm_adapt.patch /tmp/vllm_adapt.patch
COPY patch/mineru_adapt.patch /tmp/mineru_adapt.patch

RUN cd /vllm-workspace/vllm && \
    patch -p1 < /tmp/vllm_adapt.patch && \
    rm /tmp/vllm_adapt.patch

RUN MINERU_VL_UTILS_DIR=$(python3 -c "import mineru_vl_utils, os; print(os.path.dirname(mineru_vl_utils.__file__))") && \
    cd "${MINERU_VL_UTILS_DIR}" && \
    patch -p0 < /tmp/mineru_adapt.patch && \
    rm /tmp/mineru_adapt.patch
