#!/bin/bash
# Build must exist (npm run build). Starts the production server, runs the mocked-API UI smoke, stops the server.
cd "$(dirname "$0")/.."
npx next start -p "${PORT:-3100}" > /tmp/neuromas-next.log 2>&1 &
PID=$!
trap 'kill $PID 2>/dev/null' EXIT
sleep 6
python3 e2e/smoke.py "http://localhost:${PORT:-3100}"
