FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy the repository files
COPY . .

# Install dependencies from requirements.txt (including local .whl)
RUN pip3 install --no-cache-dir -r requirements.txt

# Expose Hugging Face default port
EXPOSE 7860

# Config streamlit and execute
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
