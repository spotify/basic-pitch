FROM apache/beam_python3.10_sdk:2.51.0

RUN --mount=type=cache,target=/var/cache/apt \
  apt-get update \
  && apt-get install --no-install-recommends -y --fix-missing \
    sox \
    libsndfile1 \
    libsox-fmt-all \
    ffmpeg \
    libhdf5-dev \
  && rm -rf /var/lib/apt/lists/*

COPY . /basic-pitch
WORKDIR basic-pitch
RUN --mount=type=cache,target=/root/.cache \
  pip3 install --upgrade pip && \
  pip3 install --upgrade setuptools wheel && \
  pip3 install -e '.[train]' 

