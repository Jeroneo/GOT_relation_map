# Use the lightweight Alpine version of Nginx
FROM nginx:alpine

# Copy your HTML file into the Nginx public directory
# (Assuming your file is named index.html. If not, change the first part!)
COPY ./index.html /usr/share/nginx/html/index.html

# Nginx listens on port 80 by default inside the container
EXPOSE 80