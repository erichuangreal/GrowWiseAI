#!/bin/bash
# Stop GrowWiseAI services

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

echo "Stopping GrowWiseAI services..."

# Stop backend
BACKEND_PIDS=$(pgrep -f "uvicorn backend.main:app")
if [ -n "$BACKEND_PIDS" ]; then
    echo "Stopping backend (PIDs: $BACKEND_PIDS)..."
    kill $BACKEND_PIDS
    echo -e "${GREEN}✓${NC} Backend stopped"
else
    echo "Backend not running"
fi

# Stop frontend
FRONTEND_PIDS=$(pgrep -f "serve.*dist")
if [ -n "$FRONTEND_PIDS" ]; then
    echo "Stopping frontend (PIDs: $FRONTEND_PIDS)..."
    kill $FRONTEND_PIDS
    echo -e "${GREEN}✓${NC} Frontend stopped"
else
    echo "Frontend not running"
fi

echo ""
echo "All services stopped."
