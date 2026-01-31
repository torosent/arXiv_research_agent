#!/bin/bash
# Launch script for arXiv Research Agent

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🔬 Starting arXiv Research Agent${NC}"

# Check if .env exists
if [ ! -f .env ]; then
    echo -e "${YELLOW}⚠️  No .env file found. Copying from .env.example${NC}"
    cp .env.example .env
    echo -e "${RED}❌ Please edit .env and add your AZURE_OPENAI_ENDPOINT${NC}"
    exit 1
fi

# Check if AZURE_OPENAI_ENDPOINT is set
source .env
if [ -z "$AZURE_OPENAI_ENDPOINT" ]; then
    echo -e "${RED}❌ Please set AZURE_OPENAI_ENDPOINT in .env file${NC}"
    exit 1
fi

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}❌ Docker is not running. Please start Docker.${NC}"
    exit 1
fi

# Check if emulator is running
if ! docker ps | grep -q dtsemulator; then
    echo -e "${YELLOW}📦 Starting Durable Task Scheduler emulator...${NC}"
    
    # Remove existing container if it exists
    docker rm -f dtsemulator 2>/dev/null || true
    
    # Start emulator
    docker run --name dtsemulator -d -p 8080:8080 -p 8082:8082 mcr.microsoft.com/dts/dts-emulator:latest
    
    echo -e "${GREEN}✅ Emulator started. Dashboard: http://localhost:8082${NC}"
    
    # Wait for emulator to be ready
    echo -e "${YELLOW}⏳ Waiting for emulator to be ready...${NC}"
    sleep 5
else
    echo -e "${GREEN}✅ Emulator is already running${NC}"
fi

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Start worker in background
echo -e "${YELLOW}🔧 Starting worker...${NC}"
python -m arxiv_research_agent.worker &
WORKER_PID=$!

# Wait a moment for worker to initialize
sleep 2

# Start API server
echo -e "${YELLOW}🌐 Starting API server...${NC}"
python -m arxiv_research_agent.client &
API_PID=$!

# Wait for API to be ready
sleep 2

echo -e "${GREEN}✅ Services started!${NC}"
echo -e "   📊 Dashboard: http://localhost:8082"
echo -e "   🔌 API: http://localhost:8000"
echo -e "   📚 API Docs: http://localhost:8000/docs"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop all services${NC}"

# Handle cleanup
cleanup() {
    echo -e "\n${YELLOW}🛑 Stopping services...${NC}"
    kill $WORKER_PID 2>/dev/null || true
    kill $API_PID 2>/dev/null || true
    echo -e "${GREEN}✅ Services stopped${NC}"
    exit 0
}

trap cleanup SIGINT SIGTERM

# Wait for processes
wait
