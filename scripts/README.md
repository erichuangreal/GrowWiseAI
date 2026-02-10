# Scripts Directory

This directory contains systemd service files and utility scripts for running GrowWiseAI as a system service on Ubuntu.

## 📁 Contents

### Deployment & Setup Scripts

- **`deploy.sh`** - Complete deployment script (run this first!)
  - Sets up Python virtual environment
  - Installs all backend dependencies
  - Installs all frontend dependencies
  - Builds production bundle (optional)
  - Interactive and guided setup process

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
- **`WORKFLOWS.md`** - Common workflows and quick reference guide

## 🚀 Quick Start

### First Time Setup

Run the deployment script to set up everything from scratch:

```bash
cd /home/sean/mlproject
./scripts/deploy.sh
```

This will:
1. ✅ Check prerequisites (Python, Node.js, npm)
2. ✅ Set up environment variables
3. ✅ Create Python virtual environment
4. ✅ Install backend dependencies
5. ✅ Install frontend dependencies
6. ✅ Optionally build for production

### Install as System Service (Optional)

After deployment, you can install as a system service:

```bash
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
