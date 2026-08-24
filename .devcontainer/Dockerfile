ARG VARIANT=3.12-slim
FROM python:${VARIANT}

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

WORKDIR /app

COPY . .

# pyproject.toml is the single source of truth for dependencies. Installing
# after COPY means any source change reinstalls them, which is the accepted
# cost of not hand-maintaining a second requirements list that drifts.
RUN pip install --no-cache-dir -e ".[dev]"

EXPOSE 8501
EXPOSE 4213

# ENTRYPOINT
ENTRYPOINT ["bash"]