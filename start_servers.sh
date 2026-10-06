#!/usr/bin/env bash
# ==============================================================================
# StepWise PRO — Unified Server Launcher
# Launches FastAPI Backend (Port 8000) & Next.js Frontend (Port 3000)
# ==============================================================================

set -eo pipefail

# Text Colors & Formatting
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Workspace Root
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# Print StepWise PRO Banner
echo -e "${BLUE}${BOLD}"
echo "=============================================================================="
echo "    ____  _              _    _ _            ____  ____   ___ "
echo "   / ___|| |_ ___ _ __  | |  | (_)___  ___  |  _ \|  _ \ / _ \ "
echo "   \___ \| __/ _ \ '_ \ | |/\| | / __|/ _ \ | |_) | |_) | | | |"
echo "    ___) | ||  __/ |_) ||  /\  / \__ \  __/ |  __/|  _ <| |_| |"
echo "   |____/ \__\___| .__/  \/  \/|_|___/\___| |_|   |_| \_\\___/ "
echo "                 |_|                                          "
echo "   Precision Dementia Clinical Decision Support System — GE Healthcare 2026   "
echo "=============================================================================="
echo -e "${NC}"

# Cleanup Handler for Graceful Shutdown
cleanup() {
    echo ""
    echo -e "${YELLOW}>> Shutting down all StepWise PRO servers gracefully...${NC}"
    if [ -n "$BACKEND_PID" ] && kill -0 "$BACKEND_PID" 2>/dev/null; then
        echo -e "${RED}[Backend] Stopping FastAPI process (PID: $BACKEND_PID)...${NC}"
        kill -SIGTERM "$BACKEND_PID" 2>/dev/null || true
    fi
    if [ -n "$FRONTEND_PID" ] && kill -0 "$FRONTEND_PID" 2>/dev/null; then
        echo -e "${RED}[Frontend] Stopping Next.js process (PID: $FRONTEND_PID)...${NC}"
        kill -SIGTERM "$FRONTEND_PID" 2>/dev/null || true
    fi
    wait 2>/dev/null || true
    echo -e "${GREEN}✓ All StepWise PRO services stopped cleanly. Goodbye!${NC}"
    exit 0
}

trap cleanup SIGINT SIGTERM EXIT

# Kill existing processes on port 8000 or 3000 if occupied
echo -e "${CYAN}[1/4] Checking ports 8000 and 3000...${NC}"
for PORT in 8000 3000; do
    PID=$(lsof -ti :$PORT || true)
    if [ -n "$PID" ]; then
        echo -e "${YELLOW}  Freeing port $PORT (killing existing PID: $PID)...${NC}"
        kill -9 $PID 2>/dev/null || true
        sleep 1
    fi
done

# Check Python virtual environment
echo -e "${CYAN}[2/4] Verifying Python virtual environment...${NC}"
if [ -f "$ROOT_DIR/.venv/bin/python3" ]; then
    PYTHON_EXEC="$ROOT_DIR/.venv/bin/python3"
elif command -v python3 &>/dev/null; then
    PYTHON_EXEC="python3"
else
    echo -e "${RED}ERROR: Python 3 not found. Please install Python 3 or create .venv.${NC}"
    exit 1
fi
echo -e "${GREEN}  ✓ Python executable: $PYTHON_EXEC${NC}"

# Reseed / verify database integrity
echo -e "${CYAN}[3/4] Checking SQLite database & multi-visit integrity...${NC}"
$PYTHON_EXEC -c "
from backend.database import init_db, reseed_all_patient_visits_db
init_db()
reseed_all_patient_visits_db()
" 2>/dev/null || true
echo -e "${GREEN}  ✓ SQLite database verified at backend/data/stepwise.db${NC}"

# Check Node / npm for frontend
if ! command -v npm &>/dev/null; then
    echo -e "${RED}ERROR: npm not found. Please install Node.js (v18+).${NC}"
    exit 1
fi

echo -e "${CYAN}[4/4] Launching StepWise PRO Services...${NC}"
echo ""

# 1. Start FastAPI Backend in background with prefixed logging
echo -e "${GREEN}${BOLD}>> Starting FastAPI Backend on http://0.0.0.0:8000...${NC}"
$PYTHON_EXEC -m uvicorn backend.app:app --host 0.0.0.0 --port 8000 2>&1 | sed -e "s/^/$(echo -e "${GREEN}[Backend]${NC} ")/" &
BACKEND_PID=$!

# Wait for backend to initialize
sleep 2

# 2. Start Next.js Frontend in background with prefixed logging
echo -e "${BLUE}${BOLD}>> Starting Next.js Production Frontend on http://0.0.0.0:3000...${NC}"
if [ -d "$ROOT_DIR/frontend/.next" ]; then
    npm run start --prefix "$ROOT_DIR/frontend" -- -p 3000 2>&1 | sed -e "s/^/$(echo -e "${BLUE}[Frontend]${NC} ")/" &
else
    echo -e "${YELLOW}  Building frontend for optimized production...${NC}"
    npm run build --prefix "$ROOT_DIR/frontend"
    npm run start --prefix "$ROOT_DIR/frontend" -- -p 3000 2>&1 | sed -e "s/^/$(echo -e "${BLUE}[Frontend]${NC} ")/" &
fi
FRONTEND_PID=$!

sleep 2

echo ""
echo -e "${GREEN}${BOLD}==============================================================================${NC}"
echo -e "${GREEN}${BOLD} ✓ StepWise PRO Platform is LIVE and Operational!${NC}"
echo -e "${GREEN}${BOLD}==============================================================================${NC}"
echo -e "  🌐 ${BOLD}Patient Dossier & UI:${NC}    ${CYAN}http://localhost:3000${NC}"
echo -e "  ⚙️  ${BOLD}FastAPI REST Engine:${NC}     ${CYAN}http://localhost:8000${NC}"
echo -e "  📖 ${BOLD}Interactive API Docs:${NC}    ${CYAN}http://localhost:8000/docs${NC}"
echo -e "  🏥 ${BOLD}Analytics & ROI Hub:${NC}     ${CYAN}http://localhost:3000 (Sidebar: Analytics)${NC}"
echo -e "  🧠 ${BOLD}Imaging AI & Grad-CAM:${NC}   ${CYAN}http://localhost:3000 (Sidebar: Imaging AI)${NC}"
echo -e "${GREEN}${BOLD}==============================================================================${NC}"
echo -e "${YELLOW}>> Live logs streaming below. Press ${BOLD}Ctrl+C${NC}${YELLOW} to stop all servers.${NC}"
echo ""

# Wait for background tasks
wait $BACKEND_PID $FRONTEND_PID
