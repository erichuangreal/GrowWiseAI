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
  - Interactive setup for production (nginx or serve) or development mode
  - Production with nginx: backend service only; production with serve: backend + frontend
  - Creates log files and installs systemd services; enables auto-start on boot

- **`start.sh`** - Quick start (backend + frontend with serve). Use `./scripts/start.sh --backend-only` for nginx (backend only).
- **`stop.sh`** - Stop backend and/or frontend processes.
- **`status.sh`** - Show whether backend and frontend are running and how to access the app.

### Service Files

#### Production Mode
- **`growwiseai-backend.service`** - Backend API service (used with nginx or with serve)
- **`growwiseai-frontend.service`** - Frontend service (only when using serve on 5173; not used with nginx)

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
cd /home/sean/GrowWiseAI
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
- For production with serve (not nginx): `serve` npm package (`sudo npm install -g serve`)

## 🌐 Production with nginx

For production on your own server, you can run **only the backend** (port 8001) and serve the frontend with **nginx** (no Node on 5173). Build once (`cd frontend && npm run build`), then use the example config `scripts/nginx-growwiseai.conf.example`. Nginx serves the built files and proxies `/api` to the backend. See main README → "Production with Nginx".

## 🔐 Security Notes

- Backend runs on `127.0.0.1:8001` (localhost only)
- With systemd + serve: frontend runs on `0.0.0.0:5173`; with nginx, only port 80/443 is exposed
- Environment variables loaded from `googlies.env` (not committed to git)
