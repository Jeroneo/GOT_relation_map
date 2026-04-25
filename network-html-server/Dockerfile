# Use a lightweight version of Python
FROM python:3.11-alpine

# Set the working directory inside the container
WORKDIR /app

# Copy all your website files into the /app directory
COPY ./ /app/

# Expose port 80 (so we don't have to change your docker-compose.yml)
EXPOSE 80

# Run the same Python server command you used locally, but on port 80
CMD ["python", "-m", "http.server", "80"]