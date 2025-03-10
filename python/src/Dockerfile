FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Set environment variables
ENV PYTHONPATH=/app

# Expose port for API
EXPOSE 8000

# Start API server
CMD ["python", "src/main.py"]