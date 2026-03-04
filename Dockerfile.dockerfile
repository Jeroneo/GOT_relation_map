FROM python:3.11-slim

WORKDIR /app

COPY . .

# Verify required files are present at build time
RUN test -f index.html || (echo "ERROR: index.html is missing" && exit 1)
RUN test -f data/graph.json || (echo "ERROR: data/graph.json is missing" && exit 1)

EXPOSE 80

CMD ["python", "05_launch_server.py"]