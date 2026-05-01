#!/bin/bash
# Serve OpenCAS documentation locally (bound to localhost to avoid public exposure)

DOCS_DIR="$(dirname "$0")/docs/release/website"
PORT=8234

echo "Serving documentation securely on local loopback..."
echo "Open your browser to: http://127.0.0.1:$PORT"
echo "Press Ctrl+C to stop the server."

cd "$DOCS_DIR" && python3 -m http.server $PORT --bind 127.0.0.1
