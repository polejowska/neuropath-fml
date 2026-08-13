# Template for BraTS Path-style challenge Docker submissions.
# Interface: mount input data at /input and write all required outputs to /output.
# ADAPT: For a different inference algorithm, update the base image,
# requirements, source files, checkpoint folders, and optional environment
# variables here. Keep /input and /output stable for challenge compatibility.

# ADAPT: Change this image if your code requires a different PyTorch/CUDA stack.
# Kit default is pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime; this project
# uses 2.3.1 instead -- confirmed to build and run correctly end-to-end
# (matched a local reference prediction), where the kit's own default was
# never actually tested against this ensemble.
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

WORKDIR /workspace

# ADAPT: Add offline/runtime environment variables here, for example flags for
# Hugging Face local-only loading, thread limits, or framework-specific caches.
# No HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE needed: none of the three
# foundations (virchow2, hoptimus1, genbiopathfm) touch transformers or the
# HF Hub client at runtime any more -- all three build a local model skeleton
# (timm or vendored) and load a flat local weight file directly.
ENV PYTHONUNBUFFERED=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# ADAPT: Put algorithm-specific pip dependencies in requirements.txt.
COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r /workspace/requirements.txt

# ADAPT: Keep run.py stable unless your evaluator changes the container
# interface. Replace src/ contents with your model code, checkpoints, and utils.
COPY run.py /workspace/run.py
COPY src/ /workspace/src/

# NOTE: src/foundation_model_weights/ must contain, per foundation:
#   virchow2/model.safetensors + virchow2/config.json      loaded via timm's
#   hoptimus1/model.safetensors + hoptimus1/config.json      "local-dir:" scheme
#   genbiopathfm/model.pth                                 loaded into the
#                                                            vendored
#                                                            src/genbio_pathfm_model.py
#                                                            ViT skeleton (no
#                                                            config.json needed)
# No HF snapshot directories, no transformers/AutoModel, no trust_remote_code,
# for any foundation. virchow2/hoptimus1 DO need a small local config.json
# (timm architecture metadata, not weights) -- see
# src/foundation_model_weights/README.md for why.

# Keep this interface stable. Challenge runners usually call the image without
# extra arguments and only provide mounted /input and /output directories.
# ADAPT: Change only if the challenge requires different CLI arguments.
CMD ["python", "run.py", "--input", "/input", "--output", "/output"]
