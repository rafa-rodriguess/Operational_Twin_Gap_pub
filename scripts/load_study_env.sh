#!/usr/bin/env bash
# Source from the study root or any subdirectory.
#   source scripts/load_study_env.sh
set -euo pipefail

_find_root() {
    if git rev-parse --show-toplevel >/dev/null 2>&1; then
        git rev-parse --show-toplevel
        return
    fi
    echo "load_study_env.sh: not inside a git working tree" >&2
    return 1
}

STUDY_ROOT="$(_find_root)"
ENV_FILE="${ENV_FILE:-$STUDY_ROOT/.env}"
if [[ ! -f "$ENV_FILE" ]]; then
    echo "Missing $ENV_FILE (copy from .env.example)" >&2
    return 1 2>/dev/null || exit 1
fi
set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a
export STUDY_ROOT
