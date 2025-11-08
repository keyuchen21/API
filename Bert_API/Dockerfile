# Step 1: Base image — use an official Python image
FROM python:3.10-slim

# Step 2: Set working directory inside container
WORKDIR /app

# Step 3: Copy your project files into the container
COPY . /app

# Copy a custom resolv.conf to override DNS just for build
# (This avoids the read-only file system issue)
COPY resolv.conf /tmp/resolv.conf
RUN cp /etc/resolv.conf /tmp/resolv.conf.bak && \
    cat /tmp/resolv.conf > /etc/resolv.conf || true

# Step 4: Install dependencies
RUN PIP_DEFAULT_TIMEOUT=100 pip install --no-cache-dir fastapi uvicorn torch transformers
# (Optional) RUN PIP_DEFAULT_TIMEOUT=100 pip install --no-cache-dir fastapi uvicorn gunicorn torch transformers

# ✅ Step 5: Copy pre-downloaded local model into the image
COPY bert_model /app/bert_model

# Step 6: Expose port and start the app
EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]

# (Optional) Production variant:
# CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "main:app", "-w", "2", "-b", "0.0.0.0:8080", "--timeout", "120"]
