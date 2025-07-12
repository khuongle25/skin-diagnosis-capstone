# FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
FROM python:3.10-bullseye
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV DEBIAN_FRONTEND=noninteractive

# Tạo thư mục làm việc
WORKDIR /app

# download https://storage.googleapis.com/model-khuonglele/checkpoints.zip
ADD "https://storage.googleapis.com/model-khuonglele/checkpoints.zip" /app
ADD "https://storage.googleapis.com/model-khuonglele/meta_classifier.zip" /app

RUN apt-get update 
RUN apt-get install -y \
    unzip bzip2

RUN unzip checkpoints.zip && rm checkpoints.zip
RUN unzip meta_classifier.zip && rm meta_classifier.zip

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-editable

# Copy toàn bộ mã nguồn sau (thường xuyên thay đổi)

COPY . .

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked
RUN pip install git+https://github.com/MarioVilas/googlesearch.git
ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--log-level", "debug"]