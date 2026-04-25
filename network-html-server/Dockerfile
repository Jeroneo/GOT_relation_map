# Use the lightweight Alpine version of Nginx
FROM nginx:alpine

# Copy EVERYTHING in the current directory to the Nginx public folder
COPY ./ /usr/share/nginx/html/

# Nginx listens on port 80 by default inside the container
EXPOSE 80