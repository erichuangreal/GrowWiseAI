#!/bin/bash
# GrowWiseAI Deployment Script
# Sets up backend and frontend from scratch

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project configuration
PROJECT_DIR="/home/sean/GrowWiseAI"
VENV_DIR="$PROJECT_DIR/venv"
FRONTEND_DIR="$PROJECT_DIR/frontend"
BACKEND_DIR="$PROJECT_DIR/backend"
ENV_FILE="$PROJECT_DIR/googlies.env"
ENV_EXAMPLE="$PROJECT_DIR/googlies.env.example"
# Nginx web root for production static files
WEB_ROOT="/var/www/growwiseai"

echo -e "${BLUE}╔════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   GrowWiseAI Deployment Script        ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════╝${NC}"
echo ""

# Function to print status
print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

# Check if running from project directory
if [ ! -f "$PROJECT_DIR/README.md" ]; then
    print_error "Please run this script from the project root or update PROJECT_DIR variable"
    exit 1
fi

cd "$PROJECT_DIR"
print_status "Working directory: $PROJECT_DIR"
echo ""

# ============================================================================
# PREREQUISITES CHECK
# ============================================================================

echo -e "${BLUE}[1/6] Checking Prerequisites...${NC}"

# Check Python
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed"
    exit 1
fi
PYTHON_VERSION=$(python3 --version)
print_status "Found $PYTHON_VERSION"

# Check Node.js
if ! command -v node &> /dev/null; then
    print_error "Node.js is not installed"
    exit 1
fi
NODE_VERSION=$(node --version)
print_status "Found Node.js $NODE_VERSION"

# Check npm
if ! command -v npm &> /dev/null; then
    print_error "npm is not installed"
    exit 1
fi
NPM_VERSION=$(npm --version)
print_status "Found npm $NPM_VERSION"

echo ""

# ============================================================================
# ENVIRONMENT FILE SETUP
# ============================================================================

echo -e "${BLUE}[2/6] Setting up Environment Variables...${NC}"

if [ ! -f "$ENV_FILE" ]; then
    print_warning "googlies.env not found"
    
    if [ -f "$ENV_EXAMPLE" ]; then
        echo "Creating googlies.env from template..."
        cp "$ENV_EXAMPLE" "$ENV_FILE"
        print_info "Please edit googlies.env and add your API keys:"
        print_info "  - GOOGLE_MAPS_API_KEY"
        print_info "  - GEMINI_API_KEY"
        echo ""
        read -p "Press Enter after you've added your API keys, or Ctrl+C to exit..."
    else
        print_error "googlies.env.example not found. Cannot proceed."
        exit 1
    fi
else
    print_status "Environment file exists: $ENV_FILE"
fi

# Validate environment file has keys
if ! grep -q "GOOGLE_MAPS_API_KEY=" "$ENV_FILE" || ! grep -q "GEMINI_API_KEY=" "$ENV_FILE"; then
    print_warning "Environment file seems incomplete"
    print_info "Make sure googlies.env contains both API keys"
fi

echo ""

# ============================================================================
# BACKEND SETUP
# ============================================================================

echo -e "${BLUE}[3/6] Setting up Backend...${NC}"

# Create virtual environment
if [ -d "$VENV_DIR" ]; then
    print_info "Virtual environment already exists at $VENV_DIR"
    read -p "Recreate it? (y/N): " recreate
    if [[ $recreate =~ ^[Yy]$ ]]; then
        print_info "Removing existing virtual environment..."
        rm -rf "$VENV_DIR"
        print_status "Creating new virtual environment..."
        python3 -m venv "$VENV_DIR"
    fi
else
    print_status "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Upgrade pip
print_info "Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1

# Install Python dependencies
print_status "Installing Python dependencies from requirements.txt..."
if [ -f "$PROJECT_DIR/requirements.txt" ]; then
    pip install -r "$PROJECT_DIR/requirements.txt"
    print_status "Backend dependencies installed successfully"
else
    print_error "requirements.txt not found"
    exit 1
fi

echo ""

# ============================================================================
# FRONTEND SETUP
# ============================================================================

echo -e "${BLUE}[4/6] Setting up Frontend...${NC}"

if [ ! -d "$FRONTEND_DIR" ]; then
    print_error "Frontend directory not found: $FRONTEND_DIR"
    exit 1
fi

cd "$FRONTEND_DIR"

# Clean install
if [ -d "node_modules" ]; then
    print_info "node_modules already exists"
    read -p "Reinstall dependencies? (y/N): " reinstall
    if [[ $reinstall =~ ^[Yy]$ ]]; then
        print_info "Removing node_modules..."
        rm -rf node_modules package-lock.json
        print_status "Installing npm dependencies..."
        npm install
    fi
else
    print_status "Installing npm dependencies..."
    npm install
fi

print_status "Frontend dependencies installed successfully"

echo ""

# ============================================================================
# BUILD OPTION
# ============================================================================

echo -e "${BLUE}[5/6] Build Configuration...${NC}"
echo "Choose deployment mode:"
echo "  1) Development (npm run dev - with hot reload)"
echo "  2) Production (npm run build - optimized build)"
echo ""
read -p "Enter choice [1-2]: " build_choice

cd "$FRONTEND_DIR"

if [ "$build_choice" = "2" ]; then
    print_status "Building production bundle..."
    npm run build
    
    # Check if build was successful
    if [ -d "dist" ]; then
        print_status "Production build created successfully in frontend/dist/"
        print_info "Syncing dist/ to nginx web root at $WEB_ROOT (sudo)..."
        sudo mkdir -p "$WEB_ROOT"
        sudo rsync -a --delete "dist/" "$WEB_ROOT/"
        print_status "Frontend deployed to $WEB_ROOT"
        print_info "Nginx should be configured with: root $WEB_ROOT;"
    else
        print_error "Build failed - dist directory not created"
        exit 1
    fi
else
    print_status "Development mode selected - build step skipped"
fi

cd "$PROJECT_DIR"
echo ""

# ============================================================================
# SUMMARY & NEXT STEPS
# ============================================================================

echo -e "${BLUE}[6/6] Deployment Summary${NC}"
echo ""
echo -e "${GREEN}✓ Backend Setup Complete${NC}"
echo "  - Virtual environment: $VENV_DIR"
echo "  - Python dependencies: Installed"
echo "  - Environment file: $ENV_FILE"
echo ""
echo -e "${GREEN}✓ Frontend Setup Complete${NC}"
echo "  - Frontend directory: $FRONTEND_DIR"
echo "  - Node dependencies: Installed"

if [ "$build_choice" = "2" ]; then
    echo "  - Production build: Ready in frontend/dist/ and synced to $WEB_ROOT"
else
    echo "  - Mode: Development"
fi

echo ""
echo -e "${BLUE}═══════════════════════════════════════${NC}"
echo -e "${GREEN}  Deployment Complete! 🎉${NC}"
echo -e "${BLUE}═══════════════════════════════════════${NC}"
echo ""

# ============================================================================
# NEXT STEPS
# ============================================================================

echo -e "${YELLOW}Next Steps:${NC}"
echo ""

if [ "$build_choice" = "2" ]; then
    echo "📦 Production Mode:"
    echo ""
    echo "  Option A (recommended): Nginx reverse proxy"
    echo "    Run only the backend; nginx serves frontend and proxies /api."
    echo "    Backend:  source venv/bin/activate && uvicorn backend.main:app --host 127.0.0.1 --port 8001"
    echo "    Nginx:    see scripts/nginx-growwiseai.conf.example and main README."
    echo "    Access:   http://localhost or your domain (port 80)."
    echo ""
    echo "  Option B: Run manually (backend + serve on 5173)"
    echo "    Terminal 1 - Backend:  source venv/bin/activate && uvicorn backend.main:app --host 127.0.0.1 --port 8001"
    echo "    Terminal 2 - Frontend: cd frontend && npx serve -s dist -l 5173 --host 0.0.0.0"
    echo "    (Install serve if needed: npm install -g serve)"
    echo ""
    echo "  Option C: Install as system services"
    echo "    ./scripts/setup-services.sh (choose Production; with nginx, enable only backend)"
else
    echo "🔧 Development Mode:"
    echo ""
    echo "  Option A: Run manually"
    echo "    Terminal 1 - Backend:"
    echo "      source venv/bin/activate"
    echo "      uvicorn backend.main:app --reload --host 127.0.0.1 --port 8001"
    echo ""
    echo "    Terminal 2 - Frontend:"
    echo "      cd frontend"
    echo "      npm run dev"
    echo ""
    echo "  Option B: Install as system services"
    echo "    ./scripts/setup-services.sh"
    echo "    (Choose option 2 for Development)"
fi

echo ""
echo "🌐 Access your application:"
if [ "$build_choice" = "2" ]; then
    echo "   With nginx: http://localhost or your domain (port 80)"
    echo "   With serve: http://localhost:5173 or http://$(hostname -I | awk '{print $1}'):5173"
else
    echo "   Local:   http://localhost:5173"
    echo "   Network: http://$(hostname -I | awk '{print $1}'):5173"
fi
echo ""
echo "📚 Documentation:"
echo "   Service management: scripts/SERVICE-MANAGEMENT.md"
echo "   Main README: README.md"
echo ""

# Ask if user wants to start services now
echo ""
read -p "Would you like to start the services now? (y/N): " start_now

if [[ $start_now =~ ^[Yy]$ ]]; then
    if [ "$build_choice" = "2" ]; then
        echo ""
        print_info "Starting backend (production: use nginx to serve frontend)..."
        
        LOGS_DIR="$PROJECT_DIR/logs"
        mkdir -p "$LOGS_DIR"
        
        echo "Backend starting in background..."
        cd "$PROJECT_DIR"
        source "$VENV_DIR/bin/activate"
        nohup uvicorn backend.main:app --host 127.0.0.1 --port 8001 > "$LOGS_DIR/backend.log" 2>&1 &
        BACKEND_PID=$!
        echo "Backend PID: $BACKEND_PID"
        
        sleep 2
        echo ""
        print_status "Backend started!"
        echo "Backend logs: $LOGS_DIR/backend.log"
        echo ""
        echo "Serve the frontend with nginx (see scripts/nginx-growwiseai.conf.example),"
        echo "or run manually: cd frontend && npx serve -s dist -l 5173 --host 0.0.0.0"
        echo ""
        echo "To stop backend: kill $BACKEND_PID"
    else
        echo ""
        print_info "Development mode requires two terminals. Please start manually:"
        echo ""
        echo "Terminal 1: source venv/bin/activate && uvicorn backend.main:app --reload --host 127.0.0.1 --port 8001"
        echo "Terminal 2: cd frontend && npm run dev"
    fi
fi

echo ""
print_status "Done! Happy coding! 🚀"
