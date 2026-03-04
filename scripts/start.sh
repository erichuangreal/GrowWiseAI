#!/bin/bash
# Quick start script for GrowWiseAI
# Usage: ./scripts/start.sh [--backend-only]
#   --backend-only  Start only backend (for nginx; frontend served by nginx)

PROJECT_DIR="/home/sean/GrowWiseAI"
VENV_DIR="$PROJECT_DIR/venv"
FRONTEND_DIR="$PROJECT_DIR/frontend"
LOGS_DIR="$PROJECT_DIR/logs"
BACKEND_ONLY=""
[ "$1" = "--backend-only" ] || [ "$1" = "-b" ] && BACKEND_ONLY="y"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}Starting GrowWiseAI...${NC}"

# Create logs directory
mkdir -p "$LOGS_DIR"

# Check if already running
if pgrep -f "uvicorn backend.main:app" > /dev/null; then
    echo "Backend already running!"
else
    echo "Starting backend..."
    cd "$PROJECT_DIR"
    source "$VENV_DIR/bin/activate"
    nohup uvicorn backend.main:app --host 127.0.0.1 --port 8001 > "$LOGS_DIR/backend.log" 2>&1 &
    echo -e "${GREEN}✓${NC} Backend started (PID: $!)"
fi

if [ -z "$BACKEND_ONLY" ]; then
    if lsof -i :5173 > /dev/null 2>&1; then
        echo "Frontend port 5173 already in use"
        FRONTEND_PID=$(lsof -t -i :5173)
        echo "  PID: $FRONTEND_PID"
    else
        echo "Starting frontend..."
        cd "$FRONTEND_DIR"
        nohup npx serve -s dist -l 5173 -L > "$LOGS_DIR/frontend.log" 2>&1 &
        echo -e "${GREEN}✓${NC} Frontend started (PID: $!)"
    fi
fi

sleep 2

echo ""
echo -e "${GREEN}✓ Services started!${NC}"
echo ""
echo "Access your app:"
if [ -n "$BACKEND_ONLY" ]; then
    echo "  Backend only (for nginx). Serve frontend via nginx: http://localhost (port 80)"
else
    echo "  Local:   http://localhost:5173"
    echo "  Network: http://$(hostname -I | awk '{print $1}'):5173"
fi
echo ""
echo "View logs:"
echo "  tail -f $LOGS_DIR/backend.log"
[ -z "$BACKEND_ONLY" ] && echo "  tail -f $LOGS_DIR/frontend.log"
echo ""
echo "Stop services:"
echo "  ./scripts/stop.sh"
