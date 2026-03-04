FROM python:3.11-slim

WORKDIR /app

COPY . .

EXPOSE 80

CMD ["python", "05_launch_server.py"]