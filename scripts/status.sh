#!/bin/bash
# Check GrowWiseAI service status

GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

echo "GrowWiseAI Service Status"
echo "========================="
echo ""

# Check backend
BACKEND_PIDS=$(pgrep -f "uvicorn backend.main:app")
if [ -n "$BACKEND_PIDS" ]; then
    echo -e "${GREEN}✓${NC} Backend: Running (PIDs: $BACKEND_PIDS)"
    if lsof -i :8001 > /dev/null 2>&1; then
        echo "  Port 8001: Listening"
    fi
else
    echo -e "${RED}✗${NC} Backend: Not running"
fi

# Check frontend
FRONTEND_PIDS=$(pgrep -f "serve.*dist")
if [ -n "$FRONTEND_PIDS" ]; then
    echo -e "${GREEN}✓${NC} Frontend: Running (PIDs: $FRONTEND_PIDS)"
    if lsof -i :5173 > /dev/null 2>&1; then
        echo "  Port 5173: Listening"
    fi
else
    echo -e "${RED}✗${NC} Frontend: Not running"
fi

echo ""

# Health check
if [ -n "$BACKEND_PIDS" ]; then
    echo "Testing backend health..."
    if curl -s http://localhost:8001/health > /dev/null 2>&1; then
        echo -e "${GREEN}✓${NC} Backend API responding"
    else
        echo -e "${RED}✗${NC} Backend API not responding"
    fi
fi

if [ -n "$FRONTEND_PIDS" ]; then
    echo "Testing frontend..."
    if curl -s http://localhost:5173 > /dev/null 2>&1; then
        echo -e "${GREEN}✓${NC} Frontend responding"
    else
        echo -e "${RED}✗${NC} Frontend not responding"
    fi
fi

echo ""
echo "Access your app:"
echo "  With nginx:  http://localhost or your domain (port 80)"
echo "  With serve:  http://localhost:5173 or http://$(hostname -I | awk '{print $1}'):5173"
