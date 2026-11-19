# syntax=docker/dockerfile:1
# NovusPipeline as a shared MCP server over HTTP. See docs/enterprise.md.
#
#   docker build -t novuspipeline .                          # core install
#   docker build --build-arg EXTRAS=gnn -t novuspipeline .   # + structural GNN (CPU torch)
#
#   docker run -p 8000:8000 -v /srv/repos/billing:/workspace \
#     -v /etc/novuspipeline:/etc/xdg/novuspipeline:ro \
#     -e NOVUS_AUTH_TOKENS_FILE=/etc/xdg/novuspipeline/tokens.toml novuspipeline
#
# The server binds all interfaces inside the container, so it refuses to start without
# authentication (API tokens, NOVUS_AUTH_TOKEN or OIDC).

FROM python:3.15-slim AS build
WORKDIR /src
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir build && python -m build --wheel --outdir /dist

FROM python:3.15-slim
ARG EXTRAS=""
LABEL org.opencontainers.image.title="NovusPipeline" \
      org.opencontainers.image.description="Codebase-aware legacy code modernization MCP server, test-gated" \
      org.opencontainers.image.source="https://github.com/Almogod/NovusPipeline" \
      org.opencontainers.image.licenses="MIT"

RUN apt-get update \
 && apt-get install -y --no-install-recommends git \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --create-home --uid 10001 --shell /usr/sbin/nologin novus \
 # The mounted repository is usually owned by another uid; trust exactly that path, nothing else.
 && git config --system --add safe.directory /workspace \
 && git config --system user.name "NovusPipeline" \
 && git config --system user.email "novuspipeline@localhost"

COPY --from=build /dist /dist
RUN if [ -n "$EXTRAS" ]; then pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu; fi \
 && wheel="$(ls /dist/*.whl)" \
 && pip install --no-cache-dir "${wheel}${EXTRAS:+[$EXTRAS]}" \
 && rm -rf /dist

USER novus
ENV PYTHONUNBUFFERED=1 \
    NOVUS_WORKSPACE_ROOT=/workspace \
    NOVUS_LOG_FORMAT=json \
    NOVUS_LOG_LEVEL=INFO \
    NOVUS_LLM_OFFLINE=1
WORKDIR /workspace
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD ["python", "-c", "import sys, urllib.request; sys.exit(urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=4).status != 200)"]
ENTRYPOINT ["novuspipeline", "serve", "--transport", "http", "--host", "0.0.0.0", "--port", "8000"]
