#!/usr/bin/env bash
#
# validate-submission.sh — OpenEnv Submission Validator for OfficeAgentEnv
#
# Checks that your HF Space is live, Docker image builds, and openenv validate passes.
#
# Prerequisites:
#   - Docker:       https://docs.docker.com/get-docker/
#   - openenv-core: pip install openenv-core
#   - curl (usually pre-installed)
#
# Run:
#   curl -fsSL https://raw.githubusercontent.com/<owner>/<repo>/main/scripts/validate-submission.sh | bash -s -- <ping_url> [repo_dir]
#
#   Or download and run locally:
#     chmod +x validate-submission.sh
#     ./validate-submission.sh <ping_url> [repo_dir]
#
# Arguments:
#   ping_url   Your HuggingFace Space URL (e.g. https://your-space.hf.space)
#   repo_dir   Path to your repo (default: current directory)
#
# Examples:
#   ./validate-submission.sh https://my-team.hf.space
#   ./validate-submission.sh https://my-team.hf.space ./my-repo
#

set -uo pipefail

DOCKER_BUILD_TIMEOUT=600
OPENENV_VALIDATE_TIMEOUT=300

if [ -t 1 ]; then
  RED='\033[0;31m'
  GREEN='\033[0;32m'
  YELLOW='\033[1;33m'
  BOLD='\033[1m'
  NC='\033[0m'
else
  RED='' GREEN='' YELLOW='' BOLD='' NC=''
fi

info() {
  echo -e "${BOLD}[*]${NC} $*"
}

ok() {
  echo -e "${GREEN}[OK]${NC} $*"
}

warn() {
  echo -e "${YELLOW}[WARN]${NC} $*"
}

fail() {
  echo -e "${RED}[FAIL]${NC} $*" >&2
}

run_with_timeout() {
  local secs="$1"; shift
  if command -v timeout >/dev/null 2>&1; then
    timeout "$secs" "$@"
  elif command -v gtimeout >/dev/null 2>&1; then
    gtimeout "$secs" "$@"
  else
    "$@" &
    local pid=$!
    ( sleep "$secs" && kill "$pid" 2>/dev/null ) &
    local watcher=$!
    wait "$pid" 2>/dev/null
    local rc=$?
    kill "$watcher" 2>/dev/null
    wait "$watcher" 2>/dev/null || true
    return "$rc"
  fi
}

portable_mktemp() {
  local prefix="${1:-validate}"
  if command -v mktemp >/dev/null 2>&1; then
    mktemp "${TMPDIR:-/tmp}/${prefix}.XXXXXX" 2>/dev/null || mktemp
  else
    echo "${TMPDIR:-/tmp}/${prefix}.$$"
  fi
}

usage() {
  cat <<EOF
Usage: $(basename "$0") <ping_url> [repo_dir]

  ping_url   Your HuggingFace Space URL (e.g. https://your-space.hf.space)
  repo_dir   Path to your repo (default: current directory)
EOF
}

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
  usage
  exit 1
fi

PING_URL="$1"
REPO_DIR="${2:-.}"

if ! command -v curl >/dev/null 2>&1; then
  fail "curl is required but was not found in PATH."
  exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
  fail "Docker is required but was not found in PATH."
  exit 1
fi

# ---------------------------------------------------------------------------
# Step 1 — Ping HuggingFace Space
# ---------------------------------------------------------------------------

info "Pinging HF Space: $PING_URL"
if run_with_timeout 20 curl -fsS "$PING_URL" >/dev/null; then
  ok "HF Space is reachable."
else
  fail "HF Space is not reachable or returned an error."
  exit 1
fi

# ---------------------------------------------------------------------------
# Step 2 — Docker build
# ---------------------------------------------------------------------------

info "Changing directory to repo: $REPO_DIR"
cd "$REPO_DIR" 2>/dev/null || { fail "Repo directory not found: $REPO_DIR"; exit 1; }

if [ ! -f Dockerfile ]; then
  fail "No Dockerfile found in repo directory ($REPO_DIR)."
  exit 1
fi

IMAGE_TAG="officeagentenv-validate"
info "Building Docker image ($IMAGE_TAG) with timeout ${DOCKER_BUILD_TIMEOUT}s..."
if run_with_timeout "$DOCKER_BUILD_TIMEOUT" docker build -t "$IMAGE_TAG" .; then
  ok "Docker image built successfully."
else
  fail "Docker build failed or timed out."
  exit 1
fi

# ---------------------------------------------------------------------------
# Step 3 — openenv validation
# ---------------------------------------------------------------------------

info "Running openenv validation..."

validate_cmd=""
if command -v openenv >/dev/null 2>&1; then
  validate_cmd="openenv validate ."
elif python - <<'PY' 2>/dev/null
import importlib
import sys
try:
    importlib.import_module("openenv_core")
except ImportError:
    sys.exit(1)
PY
then
  validate_cmd="python -m openenv_core validate ."
fi

if [ -z "$validate_cmd" ]; then
  fail "Neither 'openenv' CLI nor 'openenv_core' module is available. Install openenv-core first."
  exit 1
fi

info "Using validation command: $validate_cmd"
# shellcheck disable=SC2086
if run_with_timeout "$OPENENV_VALIDATE_TIMEOUT" bash -lc "$validate_cmd"; then
  ok "openenv validation passed."
else
  fail "openenv validation failed or timed out."
  exit 1
fi

ok "All checks passed. Your OfficeAgentEnv submission looks ready."
exit 0
