# Use an official Python runtime as a parent image (choose a version compatible with your code)
FROM python:3.10-slim

# Set environment variables to prevent Python from writing pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /app

# Install system dependencies (Faiss might need libomp)
RUN apt-get update && apt-get install -y --no-install-recommends libomp5 && \
    rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container at /app
# Make sure you have a requirements.txt file in your project root!
COPY requirements.txt .

# Install Python dependencies from requirements.txt
# Use --no-cache-dir to reduce image size
# Consider adding --index-url https://download.pytorch.org/whl/cpu for torch if you only need CPU support
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code (app.py, profiles.json, api_docs.md, etc.)
# Assumes your main python file is named 'app.py'
COPY . .

# Make port 8080 available to the world outside this container
# This is the port Uvicorn will listen on inside the container
EXPOSE 8080

# Define environment variable placeholder for OpenAI API Key
# --- IMPORTANT: DO NOT PUT YOUR ACTUAL KEY HERE ---
# Render will inject the real key from its secrets management
ENV OPENAI_API_KEY=""

# Define the command to run your application using Uvicorn
# - "app:app" means: run the object named 'app' from the file 'app.py'
# - "--host 0.0.0.0" makes the server accessible from outside the container
# - "--port 8080" matches the EXPOSE instruction
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
