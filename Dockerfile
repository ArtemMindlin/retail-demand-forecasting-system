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

# Copy project configuration files and package dynamic metadata sources
COPY pyproject.toml README.md LICENSE ./
COPY src/retail_forecasting/__init__.py src/retail_forecasting/__init__.py

# Install dependencies using uv (we install the ML group to get lightgbm/xgboost if needed)
RUN uv pip install --system -e ".[ml,dev]"

# Copy the rest of the project
COPY . .

# Collect the dashboard's static assets into STATIC_ROOT so they can be served
# with the hashed filenames ManifestStaticFilesStorage expects in production.
RUN DJANGO_DEBUG=false DJANGO_SECRET_KEY=build-time-only \
    python manage.py collectstatic --noinput

# Expose the dashboard port
EXPOSE 8000

# Serve the Django app over ASGI. PORT is injected by the platform when present.
CMD ["sh", "-c", "uvicorn retail_forecasting.api.asgi:application --host 0.0.0.0 --port ${PORT:-8000}"]
