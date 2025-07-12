FROM python:3.10-bullseye

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN python -m venv .venv
ENV PATH="/app/.venv/bin:$PATH"
RUN pip install uv
RUN uv sync --no-install-project --no-editable

COPY chatbot ./chatbot

EXPOSE 8080

CMD ["python", "-m", "uvicorn", "chatbot.sse_server:app", "--host", "0.0.0.0", "--port", "8080"]