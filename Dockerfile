# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# We install gcc and libpq-dev temporarily to build psycopg
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get purge -y --auto-remove gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy the rest of the application code
COPY . .

# Set a default port if not provided by the cloud platform
ENV PORT=8080
EXPOSE $PORT

# Run the application using Gunicorn (Production WSGI Server)
# We use the shell form so the $PORT environment variable resolves correctly
CMD gunicorn --bind 0.0.0.0:$PORT --workers 2 --timeout 120 main:app