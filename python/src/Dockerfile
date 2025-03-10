FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements from src directory
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files from src directory to container
COPY . .

# Set up AWS configuration directory
RUN mkdir -p /root/.aws

# Create an entry point script for credential handling
RUN echo '#!/bin/bash\n\
if [ -n "$AWS_ACCESS_KEY_ID" ] && [ -n "$AWS_SECRET_ACCESS_KEY" ]; then\n\
  mkdir -p /root/.aws\n\
  echo "[default]" > /root/.aws/credentials\n\
  echo "aws_access_key_id = $AWS_ACCESS_KEY_ID" >> /root/.aws/credentials\n\
  echo "aws_secret_access_key = $AWS_SECRET_ACCESS_KEY" >> /root/.aws/credentials\n\
  if [ -n "$AWS_SESSION_TOKEN" ]; then\n\
    echo "aws_session_token = $AWS_SESSION_TOKEN" >> /root/.aws/credentials\n\
  fi\n\
  echo "[default]" > /root/.aws/config\n\
  echo "region = $AWS_REGION" >> /root/.aws/config\n\
fi\n\
\n\
exec "$@"' > /entrypoint.sh && chmod +x /entrypoint.sh

# Set environment variables
ENV PYTHONPATH=/app \
    API_MODE=true \
    PORT=8000

# Add healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Expose port for API
EXPOSE 8000

# Use the entrypoint script
ENTRYPOINT ["/entrypoint.sh"]

# We're already in the src directory, so just use main.py directly
CMD ["python", "main.py"]