FROM python:3.12-slim AS api
WORKDIR /app
COPY pyproject.toml README.md ./
COPY velacl velacl
COPY apps apps
COPY experiments experiments
RUN pip install --no-cache-dir 'torch>=2.2,<3' --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir .
ENV PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["uvicorn", "apps.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

FROM node:20-alpine AS dashboard-build
WORKDIR /app
COPY apps/dashboard/package*.json ./
RUN npm install
COPY apps/dashboard ./
RUN npm run build

FROM node:20-alpine AS dashboard
WORKDIR /app
COPY --from=dashboard-build /app ./
EXPOSE 3000
CMD ["npm", "start"]
