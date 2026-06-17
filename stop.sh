#!/bin/bash
cd "$(dirname "$0")"

echo "=== Stopping LiteLLM Proxy ==="
docker compose down
echo "=== LiteLLM Proxy is stopped ==="
