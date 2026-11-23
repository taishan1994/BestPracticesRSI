FROM nvcr.io/nvidia/pytorch@sha256:3c90e38f5ec51e51d1c73bd7eb3d83674a254f451147c5cadc4344314258a112
COPY wheels /wheels
COPY configs/requirements-product.txt /tmp/fa4-requirements.txt
RUN python -m pip install --no-index --find-links=/wheels --no-cache-dir -r /tmp/fa4-requirements.txt && \
    python -m pip install --no-index --find-links=/wheels --no-cache-dir --force-reinstall --no-deps nvidia-cutlass-dsl-libs-base==4.8.1 nvidia-cutlass-dsl-libs-core==4.8.1 && \
    python -c "import cutlass, importlib.metadata as m; print(m.version('flash-attn-4'),m.version('nvidia-cutlass-dsl'))"
COPY . /workspace
WORKDIR /workspace
ENV PYTHONDONTWRITEBYTECODE=1 \
    FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=0 \
    CUTE_DSL_CACHE_DIR=/cache/cutlass \
    CUDA_CACHE_PATH=/cache/cuda \
    XDG_CACHE_HOME=/cache \
    TORCHINDUCTOR_CACHE_DIR=/cache/inductor
CMD ["python", "examples/prefill.py"]
