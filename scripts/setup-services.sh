#!/bin/bash
# GrowWiseAI Service Setup Script

set -e

PROJECT_DIR="/home/sean/GrowWiseAI"
SCRIPTS_DIR="$PROJECT_DIR/scripts"
USER="sean"

echo "🚀 Setting up GrowWiseAI as systemd services..."
echo ""

# Check if running as correct user
if [ "$USER" != "sean" ]; then
    echo "⚠️  This script should be run as user 'sean'"
    exit 1
fi

# Ask which mode
echo "Select mode:"
echo "1) Production (optimized build)"
echo "2) Development (with hot reload)"
read -p "Enter choice [1-2]: " mode

USE_NGINX=""
if [ "$mode" = "1" ]; then
    SERVICE_PREFIX=""
    echo "📦 Building frontend..."
    cd "$PROJECT_DIR/frontend"
    npm run build
    
    echo ""
    read -p "Use nginx to serve frontend? (Y/n): " nginx_choice
    if [[ ! "$nginx_choice" =~ ^[Nn]$ ]]; then
        USE_NGINX="y"
        echo "   Backend-only setup (nginx serves frontend from frontend/dist)"
    else
        if ! command -v serve &> /dev/null; then
            echo "📥 Installing 'serve' package..."
            sudo npm install -g serve
        fi
    fi
elif [ "$mode" = "2" ]; then
    SERVICE_PREFIX="-dev"
    echo "🔧 Using development mode..."
else
    echo "❌ Invalid choice"
    exit 1
fi

# Create log files
echo "📝 Creating log files..."
sudo touch /var/log/growwiseai-backend${SERVICE_PREFIX}.log
sudo touch /var/log/growwiseai-backend${SERVICE_PREFIX}-error.log
if [ -z "$USE_NGINX" ]; then
    sudo touch /var/log/growwiseai-frontend${SERVICE_PREFIX}.log
    sudo touch /var/log/growwiseai-frontend${SERVICE_PREFIX}-error.log
fi
sudo chown $USER:$USER /var/log/growwiseai-*.log

# Install service files
echo "📋 Installing service files..."
sudo cp "$SCRIPTS_DIR/growwiseai-backend${SERVICE_PREFIX}.service" /etc/systemd/system/
if [ -z "$USE_NGINX" ]; then
    sudo cp "$SCRIPTS_DIR/growwiseai-frontend${SERVICE_PREFIX}.service" /etc/systemd/system/
fi

# Reload systemd
echo "🔄 Reloading systemd..."
sudo systemctl daemon-reload

# Enable and start services
echo "✅ Enabling services..."
sudo systemctl enable growwiseai-backend${SERVICE_PREFIX}
if [ -z "$USE_NGINX" ]; then
    sudo systemctl enable growwiseai-frontend${SERVICE_PREFIX}
fi

echo "▶️  Starting services..."
sudo systemctl start growwiseai-backend${SERVICE_PREFIX}
if [ -z "$USE_NGINX" ]; then
    sudo systemctl start growwiseai-frontend${SERVICE_PREFIX}
fi

# Wait a moment for services to start
sleep 2

# Check status
echo ""
echo "📊 Service Status:"
echo "=================="
sudo systemctl status growwiseai-backend${SERVICE_PREFIX} --no-pager | head -n 10
if [ -z "$USE_NGINX" ]; then
    echo ""
    sudo systemctl status growwiseai-frontend${SERVICE_PREFIX} --no-pager | head -n 10
fi

echo ""
echo "✨ Setup complete!"
echo ""
if [ -n "$USE_NGINX" ]; then
    echo "🌐 Access your app: http://localhost (port 80) — configure nginx; see scripts/nginx-growwiseai.conf.example"
else
    echo "🌐 Access your app at: http://localhost:5173"
    echo "   Or from network: http://$(hostname -I | awk '{print $1}'):5173"
fi
echo ""
echo "📝 Useful commands:"
echo "   sudo systemctl status growwiseai-backend${SERVICE_PREFIX}"
if [ -z "$USE_NGINX" ]; then
    echo "   sudo systemctl status growwiseai-frontend${SERVICE_PREFIX}"
    echo "   sudo systemctl restart growwiseai-backend${SERVICE_PREFIX}"
    echo "   sudo systemctl restart growwiseai-frontend${SERVICE_PREFIX}"
else
    echo "   sudo systemctl restart growwiseai-backend${SERVICE_PREFIX}"
fi
echo "   sudo journalctl -u growwiseai-backend${SERVICE_PREFIX} -f"
if [ -z "$USE_NGINX" ]; then
    echo "   sudo journalctl -u growwiseai-frontend${SERVICE_PREFIX} -f"
fi
