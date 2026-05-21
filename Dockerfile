# Use a Python 3.11+ base image
FROM python:3.11-slim-bookworm

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv (fast Python package installer)
RUN pip install uv

# Copy project configuration files
COPY pyproject.toml .
COPY src/retail_forecasting/__init__.py src/retail_forecasting/__init__.py

# Install dependencies using uv (we install the ML group to get lightgbm/xgboost if needed)
RUN uv pip install --system -e ".[ml,dev]"

# Copy the rest of the project
COPY . .

# Expose ports for FastAPI (8000), MLflow (5000), and Streamlit (8501)
EXPOSE 8000 5000 8501

# The default command will just keep the container alive or can be overridden by docker-compose
CMD ["bash"]
