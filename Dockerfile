FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Minimal runtime deps for opencv-python on Debian slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first for better layer caching
COPY COD_BC/requirements.txt /app/COD_BC/requirements.txt
RUN python -m pip install --upgrade pip && \
    pip install -r /app/COD_BC/requirements.txt

# Copy project code
COPY COD_BC /app/COD_BC

WORKDIR /app/COD_BC

# Note:
# - `infer` / `collect` require a GUI + OS-level input access; inside Docker (Linux) this is typically not usable
#   to control Windows COD directly. Container is intended for train/eval/data-info and offline utilities.
ENTRYPOINT ["python", "main.py"]
CMD ["--help"]

