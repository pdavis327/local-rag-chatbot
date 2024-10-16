# Dockerfile, Image, Container
FROM python:3.11.10
LABEL maintainer="pete.davis@steampunk.com"

WORKDIR /app

COPY requirements.txt ./
COPY . . 

RUN pip install -r requirements.txt

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

