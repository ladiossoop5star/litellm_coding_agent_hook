#!/bin/bash
set -e
cd "$(dirname "$0")"

echo "=== Starting LiteLLM Container ==="
docker compose up -d
echo "=== LiteLLM Proxy is successfully started! ==="
