# GrowWiseAI Common Workflows

Quick reference for common deployment and development tasks.

---

## 🚀 Initial Deployment

### First Time Setup (Recommended)

```bash
cd /home/sean/mlproject
./scripts/deploy.sh
```

Follow the interactive prompts. Choose:
- **Development mode** for local development with hot reload
- **Production mode** for optimized deployment

---

## 🔧 Development Workflow

### Start Development Servers

```bash
# Terminal 1 - Backend with auto-reload
source venv/bin/activate
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8001

# Terminal 2 - Frontend with hot reload
cd frontend
npm run dev
```

### Making Changes

**Backend changes:**
- Edit Python files
- Server auto-reloads (no restart needed)

**Frontend changes:**
- Edit React/JSX files
- Browser auto-refreshes (no restart needed)

**Dependency changes:**
```bash
# Backend
source venv/bin/activate
pip install <new-package>
pip freeze > requirements.txt

# Frontend
cd frontend
npm install <new-package>
```

---

## 📦 Production Workflow

### Build for Production

```bash
# Build frontend
cd frontend
npm run build

# Verify dist/ folder created
ls -la dist/
```

### Deploy Production Build

**Option 1: Manual (Simple)**
```bash
# Terminal 1 - Backend
source venv/bin/activate
uvicorn backend.main:app --host 127.0.0.1 --port 8001

# Terminal 2 - Frontend (serve static files)
cd frontend
npx serve -s dist -l 5173 --host 0.0.0.0
```

**Option 2: System Service (Persistent)**
```bash
# Set up as systemd service
./scripts/setup-services.sh
# Choose option 1 (Production)

# Services will start automatically on boot
```

---

## 🔄 Update Workflow

### After Pulling New Code

**Development mode:**
```bash
# Update backend
source venv/bin/activate
pip install -r requirements.txt
# Server auto-reloads

# Update frontend
cd frontend
npm install
# Browser auto-refreshes
```

**Production mode (manual):**
```bash
# Update backend
source venv/bin/activate
pip install -r requirements.txt
# Restart backend server

# Update frontend
cd frontend
npm install
npm run build
# Restart frontend server
```

**Production mode (systemd service):**
```bash
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

---

## 🐛 Troubleshooting Workflow

### Check Service Status

```bash
# Manual processes
ps aux | grep -E "(uvicorn|vite|serve)"

# Systemd services
sudo systemctl status growwiseai-backend
sudo systemctl status growwiseai-frontend
```

### View Logs

**Manual mode:**
```bash
# Logs appear in terminal where services are running
```

**Systemd service mode:**
```bash
# Live logs
sudo journalctl -u growwiseai-backend -f
sudo journalctl -u growwiseai-frontend -f

# Last 50 lines
sudo journalctl -u growwiseai-backend -n 50
```

### Common Issues

**Port already in use:**
```bash
# Find process on port
sudo lsof -i :8001  # Backend
sudo lsof -i :5173  # Frontend

# Kill process
kill -9 <PID>
```

**Module not found:**
```bash
# Ensure venv is activated
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

**API keys not working:**
```bash
# Check environment file exists
cat googlies.env

# Verify format (no quotes, no spaces around =)
# Correct: API_KEY=value
# Wrong: API_KEY = "value"
```

---

## 🔐 Environment Setup

### Create Environment File

```bash
# Copy template
cp googlies.env.example googlies.env

# Edit with your keys
nano googlies.env
# or
vim googlies.env
```

### Required Keys

```env
GOOGLE_MAPS_API_KEY=your_actual_key_here
GEMINI_API_KEY=your_actual_key_here
```

Get keys from:
- Google Maps API: https://console.cloud.google.com/apis/credentials
- Gemini API: https://aistudio.google.com/app/apikey

---

## 🌐 Network Access Setup

### Allow Network Access (Ubuntu)

```bash
# Allow frontend port through firewall
sudo ufw allow 5173/tcp

# Enable firewall
sudo ufw enable

# Check status
sudo ufw status
```

### Access from Other Devices

1. Find your IP:
   ```bash
   hostname -I
   ```

2. Access from browser:
   ```
   http://[your-ip]:5173
   ```

---

## 🧹 Cleanup Workflow

### Clean Build Artifacts

```bash
# Frontend
cd frontend
rm -rf dist/ node_modules/
npm install

# Backend
rm -rf venv/
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Remove Services

```bash
# Stop services
sudo systemctl stop growwiseai-backend growwiseai-frontend

# Disable auto-start
sudo systemctl disable growwiseai-backend growwiseai-frontend

# Remove service files
sudo rm /etc/systemd/system/growwiseai-*.service
sudo systemctl daemon-reload
```

---

## 📊 Testing Workflow

### Quick Health Check

```bash
# Backend API
curl http://localhost:8001/health
# Expected: {"status":"ok"}

# Frontend
curl http://localhost:5173
# Expected: HTML content

# From another device
curl http://[your-ip]:5173
```

### Test Prediction

```bash
curl -X POST http://localhost:8001/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "elevation": 100,
      "temperature": 15,
      "humidity": 70,
      "soil_tn": 0.2,
      "soil_tp": 0.05,
      "soil_ap": 0.03,
      "soil_an": 0.1
    }
  }'
```

---

## 🎯 Quick Commands Reference

```bash
# Deployment
./scripts/deploy.sh                    # Initial setup

# Development
source venv/bin/activate               # Activate Python env
uvicorn backend.main:app --reload      # Start backend (dev)
cd frontend && npm run dev             # Start frontend (dev)

# Production
npm run build                          # Build frontend
./scripts/setup-services.sh            # Install services

# Service Management
sudo systemctl status growwiseai-*     # Check status
sudo systemctl restart growwiseai-*    # Restart services
sudo journalctl -u growwiseai-* -f     # View logs

# Troubleshooting
sudo lsof -i :8001                     # Check backend port
sudo lsof -i :5173                     # Check frontend port
pip install -r requirements.txt        # Reinstall dependencies
npm install                            # Reinstall node modules
```

---

**For more details, see:**
- [scripts/README.md](README.md) - Scripts documentation
- [scripts/SERVICE-MANAGEMENT.md](SERVICE-MANAGEMENT.md) - Service management
- [../README.md](../README.md) - Main project documentation
