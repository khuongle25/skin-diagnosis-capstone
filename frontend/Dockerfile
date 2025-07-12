# Build stage
FROM node:23 AS build
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
COPY .env .env          
RUN npm run build

# Serve static với nginx
FROM nginx:alpine
COPY --from=build /app/build /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 8080
CMD ["nginx", "-g", "daemon off;"]