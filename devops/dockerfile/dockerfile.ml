# Use official Python image with additional ML dependencies
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV APP_HOME /app

# Create and set working directory
WORKDIR $APP_HOME

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install additional ML dependencies
RUN pip install --no-cache-dir \
    scikit-learn \
    pandas \
    numpy \
    joblib \
    pyyaml

# Copy project
COPY . .

# Command to run training (example)
CMD ["python", "ml/training/train_model.py"]