.PHONY: up down logs migrate lint test build clean

# ── Docker Compose ───────────────────────────────────────────────────────────

up:  ## Start all services
	docker compose up -d --build

down:  ## Stop all services
	docker compose down

logs:  ## Tail logs for all services
	docker compose logs -f

logs-backend:  ## Tail backend logs
	docker compose logs -f backend

restart:  ## Restart all services
	docker compose restart

# ── Database ─────────────────────────────────────────────────────────────────

migrate:  ## Run Alembic migrations inside the backend container
	docker compose exec backend alembic upgrade head

migrate-down:  ## Rollback one migration
	docker compose exec backend alembic downgrade -1

# ── Development ──────────────────────────────────────────────────────────────

lint:  ## Run ruff linter
	cd backend && ruff check .

format:  ## Run ruff formatter
	cd backend && ruff format .

typecheck:  ## Run mypy
	cd backend && mypy app

test:  ## Run pytest
	cd backend && pytest -v

# ── Build ────────────────────────────────────────────────────────────────────

build:  ## Build Docker images without starting
	docker compose build

clean:  ## Remove volumes and orphan containers
	docker compose down -v --remove-orphans

# ── Help ─────────────────────────────────────────────────────────────────────

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
