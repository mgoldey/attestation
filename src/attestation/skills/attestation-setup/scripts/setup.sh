#!/usr/bin/env bash
# Thin delegator: resolves a local checkout (or the uvx-from-git fallback)
# and hands off to `attest install --yes`, the idempotent installer/doctor.
# Model checks, .env creation, first-data ingest, warmup, MCP/skill/cron
# wiring — all of that is now the installer's job (src/attestation/install.py),
# not this script's. See SKILL.md's "Configuration contract" table.
set -uo pipefail

# Default to the checkout this script lives in, found by walking up to a
# pyproject.toml marker (same test install.py:_checkout_root() uses). Walking
# beats a fixed "../.." count: the script ships inside the package, so it runs
# from src/attestation/skills/<name>/scripts/ in a checkout, from site-packages
# under uvx, and from ~/.hermes/skills/<name>/scripts/ once copied out. Only
# the first is a checkout; the other two fall through to uvx-from-git below.
_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_find_checkout() {
  local d="$1"
  while [ "$d" != "/" ]; do
    if [ -f "$d/pyproject.toml" ]; then echo "$d"; return 0; fi
    d="$(dirname "$d")"
  done
  return 1
}
_default_project_dir="$(_find_checkout "${_script_dir}" || echo "")"
PROJECT_DIR="${HERMES_RSS_PROJECT_DIR:-${_default_project_dir}}"
HERMES_CONFIG="${HOME}/.hermes/config.yaml"

fail() {
  echo "SETUP FAILED: $1" >&2
  exit 1
}

# Read skill config key `science_recommendations.repo_url` from ~/.hermes/config.yaml,
# if present -- an override for forks. Empty string if the file, key, or `yq`
# itself is missing, in which case the published repo is used.
read_repo_url() {
  if [ -f "${HERMES_CONFIG}" ] && command -v yq >/dev/null 2>&1; then
    yq -r '.science_recommendations.repo_url // ""' "${HERMES_CONFIG}" 2>/dev/null
  else
    echo ""
  fi
}

# uv on PATH
command -v uv >/dev/null 2>&1 || fail "uv is not on PATH — install uv (https://docs.astral.sh/uv/) and retry"

# Project dir exists locally; if not, fall back to uvx-from-git.
if [ -n "${PROJECT_DIR}" ] && [ -d "${PROJECT_DIR}" ]; then
  exec uv run --project "${PROJECT_DIR}" attest install --yes
fi

# PROJECT_DIR is empty when no checkout was found above (running from a
# packaged install) -- describe that rather than printing a blank path.
_where="${PROJECT_DIR:-no local checkout found}"

REPO_URL="$(read_repo_url)"
REPO_URL="${REPO_URL:-https://github.com/mgoldey/attestation}"
echo "${_where} — using uvx --from git+${REPO_URL} instead." >&2
# NOTE: the pyproject package name is `attestation` and its console script is
# `attest` ([project.scripts] attest = "attestation.cli:main") — uvx takes
# the package via --from and the executable name as the command, so this is
# `uvx --from git+<repo_url> attest ...`.
exec uvx --from "git+${REPO_URL}" attest install --yes
