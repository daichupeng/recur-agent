FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml .python-version README.md ./
RUN uv sync

COPY . .

CMD ["uv", "run", "--", "python", "main.py", "--web"]
