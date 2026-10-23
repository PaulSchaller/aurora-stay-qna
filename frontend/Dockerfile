
FROM python:3.13-slim

WORKDIR /app

COPY frontend_files/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY frontend_files/app.py .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableXsrfProtection=false"]
