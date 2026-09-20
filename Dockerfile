# syntax=docker/dockerfile:1
FROM python:3.12-slim

WORKDIR /srv/doctrace

# git is only needed if scripts/build_index.py runs inside the container (it
# sparse-clones the docs). build-essential covers any wheel that has to compile.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# --- PyTorch nightly (cu128), installed first and pinned ---
#
# The target GPU is an RTX 5060 (Blackwell, sm_120), which stable PyTorch builds
# do not support; only nightly cu128 wheels do (background in
# doctrace/answering/local_model.py and ENGINEERING_NOTES.md). Left to requirements.txt,
# pip would pick a stable release that quietly runs on CPU or crashes at kernel
# launch. Pinning the exact nightly that was verified makes a mismatch fail at
# install time instead.
#
# Nightly wheels get pruned from the index. When this pin was set, 2.12.0.dev20260408+cu128
# was the only cu128 nightly listed for cp312/manylinux. If it 404s later, look at
# https://download.pytorch.org/whl/nightly/cu128/torch/, choose a build, and confirm
# `torch.cuda.is_available()` is True before trusting it. Do not fall back to a
# stable release and do not leave it unpinned.
RUN pip install --no-cache-dir --pre \
    torch==2.12.0.dev20260408+cu128 \
    --index-url https://download.pytorch.org/whl/nightly/cu128

# Remaining packages. torch already satisfies its requirement, so pip leaves it alone.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Package code plus the runtime data: passages.json feeds the keyword index and the
# saved reports back /v1/scorecard. data/raw/ stays out of the image; only the
# offline indexing script uses it.
COPY doctrace/ doctrace/
COPY data/processed/ data/processed/
COPY dashboard.py .
# Tuned cross-encoder. doctrace/search/fusion.py falls back to the stock model when
# the weights are missing, so a build without Git LFS objects still runs, just
# without the gain from tuning on this corpus.
COPY models/docs-reranker-minilm/ models/docs-reranker-minilm/

EXPOSE 8000
EXPOSE 8501

# Default command starts the Streamlit dashboard on 8501. That suits single-container
# hosts such as Hugging Face Spaces (Docker SDK), which run the image's default CMD and
# route traffic to the app_port set in README.md. compose.yaml overrides `command:`
# for both services: the HTTP service (`api`, port 8000) and the dashboard (`ui`, port 8501).
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=3)" || exit 1

CMD ["python", "-m", "streamlit", "run", "dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
