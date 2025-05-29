FROM python:3.10-slim

WORKDIR /app

COPY ./streamlit_app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --upgrade streamlit
COPY ./streamlit_app .

# TẮT file watcher để tránh tạo quá nhiều thread
ENV STREAMLIT_SERVER_FILEWATCHERTYPE=none

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true", \
     "--server.enableCORS=false", \
     "--server.enableXsrfProtection=false", \
     "--server.fileWatcherType=none"]
