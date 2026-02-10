# Scripts Directory

This directory contains systemd service files and utility scripts for running GrowWiseAI as a system service on Ubuntu.

## 📁 Contents

### Setup Script
- **`setup-services.sh`** - Automated service installation script
  - Interactive setup for production or development mode
  - Creates log files and installs systemd services
  - Enables auto-start on boot

### Service Files

#### Production Mode
- **`growwiseai-backend.service`** - Backend API service (optimized)
- **`growwiseai-frontend.service`** - Frontend service (serves built static files)

#### Development Mode
- **`growwiseai-backend-dev.service`** - Backend API service (with hot reload)
- **`growwiseai-frontend-dev.service`** - Frontend service (Vite dev server)

### Documentation
- **`SERVICE-MANAGEMENT.md`** - Complete guide for managing the services

## 🚀 Quick Start

```bash
cd /home/sean/mlproject
./scripts/setup-services.sh
```

Follow the prompts to choose production or development mode.

## 📝 Service File Details

### Backend Service Features:
- Runs uvicorn server
- Loads environment variables from `googlies.env`
- Automatic restart on failure
- Logging to `/var/log/growwiseai-backend*.log`

### Frontend Service Features:
- Production: Serves optimized build with `serve`
- Development: Runs Vite dev server with hot reload
- Binds to `0.0.0.0:5173` for network access
- Proxies API requests to backend
- Automatic restart on failure
- Logging to `/var/log/growwiseai-frontend*.log`

## 🔧 Manual Service Management

See [SERVICE-MANAGEMENT.md](SERVICE-MANAGEMENT.md) for detailed commands including:
- Starting/stopping services
- Viewing logs
- Troubleshooting
- Updating after code changes
- Removing services

## 📋 Requirements

- Ubuntu/Debian Linux with systemd
- Node.js and npm installed
- Python virtual environment at `../venv/`
- For production mode: `serve` npm package (`sudo npm install -g serve`)

## 🔐 Security Notes

- Backend runs on `127.0.0.1:8001` (localhost only)
- Frontend runs on `0.0.0.0:5173` (accessible from network)
- Only port 5173 needs to be open in firewall
- Environment variables loaded from `googlies.env` (not committed to git)
