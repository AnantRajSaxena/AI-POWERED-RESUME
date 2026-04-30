FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (cached layer)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create instance folder for SQLite DB
RUN mkdir -p instance

# Expose port
EXPOSE 5000

# Use gunicorn for production (single worker — CPU-bound ML training)
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--timeout", "300", "--access-logfile", "-", "app:app"]
