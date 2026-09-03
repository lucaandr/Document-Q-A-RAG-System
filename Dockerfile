FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY chroma_db_medium ./chroma_db_medium
COPY app.py .

EXPOSE 7860

CMD ["python", "app.py"]