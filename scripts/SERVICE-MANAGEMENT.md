# GrowWiseAI Service Management Guide

## 🚀 Quick Setup

Run the automated setup script:

```bash
cd /home/sean/GrowWiseAI
./scripts/setup-services.sh
```

Choose:
- **Option 1** - Production (then choose: nginx for frontend = backend only, or serve on 5173 = backend + frontend services)
- **Option 2** - Development mode (hot reload, backend + frontend services)

---

## 📋 Manual Setup (if needed)

### Production with Nginx (recommended)

Run only the backend; nginx serves the built frontend and proxies `/api`.

```bash
# Build frontend
cd /home/sean/GrowWiseAI/frontend
npm run build

# Install backend service only
sudo cp scripts/growwiseai-backend.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable growwiseai-backend
sudo systemctl start growwiseai-backend

# Configure nginx (see scripts/nginx-growwiseai.conf.example)
# Then access at http://localhost (port 80)
```

### Production with serve (frontend on 5173)

```bash
# Build frontend
cd /home/sean/GrowWiseAI/frontend
npm run build
sudo npm install -g serve  # if not installed

# Install both services
sudo cp scripts/growwiseai-backend.service /etc/systemd/system/
sudo cp scripts/growwiseai-frontend.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable growwiseai-backend growwiseai-frontend
sudo systemctl start growwiseai-backend growwiseai-frontend
# Access at http://localhost:5173
```

### Development Mode

```bash
# Install services
sudo cp scripts/growwiseai-backend-dev.service /etc/systemd/system/
sudo cp scripts/growwiseai-frontend-dev.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable growwiseai-backend-dev growwiseai-frontend-dev
sudo systemctl start growwiseai-backend-dev growwiseai-frontend-dev
```

---

## 🎮 Service Control Commands

### Check Status

```bash
# Backend
sudo systemctl status growwiseai-backend
# or for dev mode:
sudo systemctl status growwiseai-backend-dev

# Frontend
sudo systemctl status growwiseai-frontend
# or for dev mode:
sudo systemctl status growwiseai-frontend-dev
```

### Start/Stop/Restart

```bash
# Start
sudo systemctl start growwiseai-backend
sudo systemctl start growwiseai-frontend

# Stop
sudo systemctl stop growwiseai-backend
sudo systemctl stop growwiseai-frontend

# Restart (after code changes in production)
sudo systemctl restart growwiseai-backend
sudo systemctl restart growwiseai-frontend

# Restart both at once
sudo systemctl restart growwiseai-backend growwiseai-frontend
```

### Enable/Disable Auto-Start

```bash
# Enable (start on boot)
sudo systemctl enable growwiseai-backend
sudo systemctl enable growwiseai-frontend

# Disable (don't start on boot)
sudo systemctl disable growwiseai-backend
sudo systemctl disable growwiseai-frontend
```

---

## 📊 Monitoring & Logs

### View Logs

```bash
# Backend logs (follow/tail)
sudo journalctl -u growwiseai-backend -f

# Frontend logs (follow/tail)
sudo journalctl -u growwiseai-frontend -f

# Last 50 lines
sudo journalctl -u growwiseai-backend -n 50

# Logs from last hour
sudo journalctl -u growwiseai-backend --since "1 hour ago"

# Logs from specific time
sudo journalctl -u growwiseai-backend --since "2024-02-09 10:00:00"
```

### Log Files

Direct log files are also available:

```bash
# Backend
tail -f /var/log/growwiseai-backend.log
tail -f /var/log/growwiseai-backend-error.log

# Frontend
tail -f /var/log/growwiseai-frontend.log
tail -f /var/log/growwiseai-frontend-error.log
```

### Check Resource Usage

```bash
# CPU and memory usage
systemctl status growwiseai-backend
systemctl status growwiseai-frontend

# Detailed info
systemd-cgtop
```

---

## 🔧 Troubleshooting

### Service Won't Start

1. **Check the status:**
   ```bash
   sudo systemctl status growwiseai-backend
   ```

2. **View detailed logs:**
   ```bash
   sudo journalctl -u growwiseai-backend -xe
   ```

3. **Common issues:**
   - Virtual environment path incorrect
   - API keys missing in `googlies.env`
   - Port already in use
   - Permissions issue

### Check if Ports are in Use

```bash
# Check if backend port is occupied
sudo lsof -i :8001

# Check if frontend port is occupied
sudo lsof -i :5173

# Kill process on port (if needed)
sudo kill -9 $(sudo lsof -t -i:8001)
```

### Service Fails After Code Changes

**Development mode:** Should auto-reload, but you can restart:
```bash
sudo systemctl restart growwiseai-backend-dev
```

**Production mode:** Must rebuild and restart.
- With nginx: rebuild frontend (`cd frontend && npm run build`); nginx serves new files. Restart backend if backend changed: `sudo systemctl restart growwiseai-backend`
- With serve: rebuild frontend and restart both: `cd frontend && npm run build && sudo systemctl restart growwiseai-frontend growwiseai-backend`

### Environment Variables Not Loading

Check the service file points to correct env file:
```bash
sudo systemctl cat growwiseai-backend
# Should show: EnvironmentFile=/home/sean/GrowWiseAI/googlies.env
```

Verify the file exists and has correct format:
```bash
cat /home/sean/GrowWiseAI/googlies.env
# Should show KEY=value format (no quotes, no spaces around =)
```

---

## 🔄 Updating Services

### After Changing Service Files

```bash
# Reload systemd configuration
sudo systemctl daemon-reload

# Restart the service
sudo systemctl restart growwiseai-backend
sudo systemctl restart growwiseai-frontend
```

### After Pulling New Code

**Production (nginx):** Only backend service is running; rebuild frontend so nginx serves new files.
```bash
cd /home/sean/GrowWiseAI
source venv/bin/activate
pip install -r requirements.txt
sudo systemctl restart growwiseai-backend

cd frontend
npm install
npm run build
# No frontend service to restart; nginx serves updated dist/
```

**Production (serve):** Backend + frontend services.
```bash
cd /home/sean/GrowWiseAI
source venv/bin/activate
pip install -r requirements.txt
sudo systemctl restart growwiseai-backend

cd frontend
npm install
npm run build
sudo systemctl restart growwiseai-frontend
```

**Development:**
```bash
cd /home/sean/GrowWiseAI

# Update backend (service will auto-reload)
source venv/bin/activate
pip install -r requirements.txt

# Update frontend (service will auto-reload)
cd frontend
npm install
# No restart needed in dev mode!
```

---

## 🗑️ Removing Services

```bash
# Stop and disable (if you have both backend and frontend services)
sudo systemctl stop growwiseai-backend growwiseai-frontend
sudo systemctl disable growwiseai-backend growwiseai-frontend

# With nginx you may only have backend:
# sudo systemctl stop growwiseai-backend
# sudo systemctl disable growwiseai-backend

# Remove service files (only the ones you installed)
sudo rm /etc/systemd/system/growwiseai-backend.service
sudo rm /etc/systemd/system/growwiseai-frontend.service   # omit if you never installed it

# Reload systemd
sudo systemctl daemon-reload
sudo systemctl reset-failed

# Remove log files (optional)
sudo rm /var/log/growwiseai-*.log
```

---

## 📱 Access Your App

Once services are running:

- **With nginx (production):** `http://localhost` or your domain (port 80). Nginx serves frontend and proxies `/api` to backend.
- **With serve / dev:** `http://localhost:5173` (or `http://[your-ip]:5173`)
- **Backend API (direct):** `http://localhost:8001` (e.g. `/health`)

Check your IP:
```bash
hostname -I
```

---

## ✅ Health Checks

Quick test to see if everything is running:

```bash
# Check backend (always)
curl http://localhost:8001/health

# Check frontend (if using serve on 5173)
curl http://localhost:5173

# With nginx (production)
curl http://localhost/
```

Expected responses:
- Backend: `{"status":"ok"}`
- Frontend (serve or nginx): HTML content of the app

---

## 🔐 Security Notes

- Backend runs on `127.0.0.1:8001` (not exposed directly; nginx or Vite proxy to it)
- With nginx: only port 80 (or 443) is exposed; nginx serves frontend and proxies `/api` to backend
- With serve: frontend runs on `0.0.0.0:5173`; only port 5173 needs to be open in firewall
- Keep `googlies.env` secure (already in `.gitignore`)

---

**Need help?** Check logs with `journalctl -u <service-name> -f`
