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

# Install dependencies from requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# Safely install local enterprise wheel (fall back gracefully to simulation if OS/platform mismatch occurs)
RUN pip3 install kalpana_sdk_enterprise-1.0.0-cp312-cp312-linux_x86_64.whl || echo "Warning: Native wheel skipped. App will run in high-fidelity simulation mode."

# Expose Hugging Face default port
EXPOSE 7860

# Config streamlit and execute
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
