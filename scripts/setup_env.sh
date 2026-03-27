#!/usr/bin/env bash
# ============================================================
# setup_env.sh — Bootstrap local development environment
#
# Usage:
#   chmod +x scripts/setup_env.sh
#   ./scripts/setup_env.sh
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "🔧  ToolRef — Development Environment Setup"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ── 1. Create .env from .env.example if not present ──────────────────────────
if [ ! -f "$PROJECT_ROOT/.env" ]; then
  echo "📋  Creating .env from .env.example ..."
  cp "$PROJECT_ROOT/.env.example" "$PROJECT_ROOT/.env"
  echo "    ✓  .env created — review & update secrets before proceeding"
else
  echo "    ✓  .env already exists — skipping"
fi

# ── 2. Check Docker / Docker Compose ─────────────────────────────────────────
if ! command -v docker &>/dev/null; then
  echo "❌  Docker not found. Install Docker Desktop: https://www.docker.com/products/docker-desktop"
  exit 1
fi

DOCKER_COMPOSE_VERSION=$(docker compose version 2>/dev/null || echo "not found")
echo "    ✓  Docker Compose: $DOCKER_COMPOSE_VERSION"

# ── 3. Pull images ────────────────────────────────────────────────────────────
echo "📦  Pulling Docker images (this may take a few minutes) ..."
cd "$PROJECT_ROOT"
docker compose pull --quiet

# ── 4. Build application images ───────────────────────────────────────────────
echo "🏗️   Building application images ..."
docker compose build --quiet

# ── 5. Frontend dependencies ──────────────────────────────────────────────────
if command -v node &>/dev/null; then
  echo "📦  Installing frontend dependencies ..."
  cd "$PROJECT_ROOT/frontend"
  npm install --silent
  echo "    ✓  Frontend deps installed"
else
  echo "    ⚠️   Node.js not found — skipping frontend install. Install from https://nodejs.org"
fi

echo ""
echo "✅  Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Start all services:    docker compose up -d"
echo "  2. Run migrations:        make migrate"
echo "  3. Start frontend dev:    cd frontend && npm run dev"
echo "  4. Open Chat UI:          http://localhost:5173"
echo "  5. API docs:              http://localhost:8000/docs"
echo ""
