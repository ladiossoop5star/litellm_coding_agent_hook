#!/bin/bash
cd "$(dirname "$0")"

echo "=== Restarting LiteLLM Proxy ==="
./stop.sh
./start.sh
echo "=== LiteLLM Proxy restarted successfully ==="
