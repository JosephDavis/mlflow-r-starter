FROM rocker/r-ver:latest

# Install system dependencies for Python and R packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3.12-venv \
    && rm -rf /var/lib/apt/lists/*

# Create and activate a Python virtual environment and install mlflow
RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir --upgrade pip mlflow

# Add virtualenv binaries to PATH
ENV PATH="/opt/venv/bin:$PATH"

# Install required R packages
RUN Rscript -e "install.packages(c('logger', 'mlflow', 'carrier', 'randomForest', 'caret', 'ggplot2', 'jsonlite'), dependencies=TRUE)"

