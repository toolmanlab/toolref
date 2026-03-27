#!/usr/bin/env bash
# ============================================================
# check_services.sh — Verify all services are healthy
#
# Usage:
#   ./scripts/check_services.sh
# ============================================================
set -euo pipefail

OK=0
FAIL=0

check() {
  local name="$1"
  local cmd="$2"
  if eval "$cmd" &>/dev/null; then
    echo "  ✓  $name"
    OK=$((OK + 1))
  else
    echo "  ✗  $name (not reachable)"
    FAIL=$((FAIL + 1))
  fi
}

echo "🔍  ToolRef — Service Health Check"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

check "Backend   (http://localhost:8000/health)" \
  "curl -sf http://localhost:8000/health"
check "Frontend  (http://localhost:5173)" \
  "curl -sf http://localhost:5173 -o /dev/null"
check "PostgreSQL (localhost:5432)" \
  "docker compose exec -T postgres pg_isready -U toolref"
check "Redis     (localhost:6379)" \
  "docker compose exec -T redis redis-cli ping"
check "Milvus    (http://localhost:9091/healthz)" \
  "curl -sf http://localhost:9091/healthz"
check "MinIO     (http://localhost:9000/minio/health/live)" \
  "curl -sf http://localhost:9000/minio/health/live"

echo ""
echo "Summary: $OK healthy, $FAIL unhealthy"
[ "$FAIL" -eq 0 ] && exit 0 || exit 1
