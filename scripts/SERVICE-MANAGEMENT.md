# GrowWiseAI Service Management Guide

## 🚀 Quick Setup

Run the automated setup script:

```bash
cd /home/sean/mlproject
./scripts/setup-services.sh
```

Choose:
- **Option 1** - Production mode (optimized, no hot reload)
- **Option 2** - Development mode (hot reload enabled)

---

## 📋 Manual Setup (if needed)

### Production Mode

```bash
# Build frontend
cd /home/sean/mlproject/frontend
npm run build

# Install serve (if not installed)
sudo npm install -g serve

# Install services
sudo cp scripts/growwiseai-backend.service /etc/systemd/system/
sudo cp scripts/growwiseai-frontend.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable growwiseai-backend growwiseai-frontend
sudo systemctl start growwiseai-backend growwiseai-frontend
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

**Production mode:** Must rebuild and restart:
```bash
# If you changed frontend code
cd /home/sean/mlproject/frontend
npm run build
sudo systemctl restart growwiseai-frontend

# If you changed backend code
sudo systemctl restart growwiseai-backend
```

### Environment Variables Not Loading

Check the service file points to correct env file:
```bash
sudo systemctl cat growwiseai-backend
# Should show: EnvironmentFile=/home/sean/mlproject/googlies.env
```

Verify the file exists and has correct format:
```bash
cat /home/sean/mlproject/googlies.env
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

**Production:**
```bash
cd /home/sean/mlproject

# Update backend
source venv/bin/activate
pip install -r requirements.txt
sudo systemctl restart growwiseai-backend

# Update frontend
cd frontend
npm install
npm run build
sudo systemctl restart growwiseai-frontend
```

**Development:**
```bash
cd /home/sean/mlproject

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
# Stop services
sudo systemctl stop growwiseai-backend growwiseai-frontend
sudo systemctl disable growwiseai-backend growwiseai-frontend

# Remove service files
sudo rm /etc/systemd/system/growwiseai-backend.service
sudo rm /etc/systemd/system/growwiseai-frontend.service

# Reload systemd
sudo systemctl daemon-reload
sudo systemctl reset-failed

# Remove log files (optional)
sudo rm /var/log/growwiseai-*.log
```

---

## 📱 Access Your App

Once services are running:

- **Local:** `http://localhost:5173`
- **Network:** `http://[your-ip]:5173`
- **Backend API:** `http://localhost:8001` (via Vite proxy at `/api`)

Check your IP:
```bash
hostname -I
```

---

## ✅ Health Checks

Quick test to see if everything is running:

```bash
# Check backend
curl http://localhost:8001/health

# Check frontend
curl http://localhost:5173

# Check from another device
curl http://[your-ip]:5173
```

Expected responses:
- Backend: `{"status":"ok"}`
- Frontend: HTML content of the app

---

## 🔐 Security Notes

- Backend runs on `127.0.0.1:8001` (not exposed to network)
- Frontend runs on `0.0.0.0:5173` (accessible from network)
- API calls proxied through Vite from frontend to backend
- Only port 5173 needs to be open in firewall
- Keep `googlies.env` secure (already in `.gitignore`)

---

**Need help?** Check logs with `journalctl -u <service-name> -f`
