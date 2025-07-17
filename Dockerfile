# official Python base image
FROM python:3.10-slim

# working dir in the container
WORKDIR /app

# Copy everything into the container
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set PYTHONPATH so src/ is importable from app/
ENV PYTHONPATH=/app

# Expose port for Streamlit
EXPOSE 8501

# Set Streamlit to run the app
CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
