#!/usr/bin/env bash
# Python formatting, linting, and typing helper
#
# Usage:
#   bash scripts/format.sh [--all | --files <file1> <file2> ...]

set -eo pipefail

# Ensure script runs from repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
cd "$ROOT"

usage() {
    echo "Usage: bash scripts/format.sh [--all | --staged | --files <file1> <file2> ...]"
    exit 1
}

ensure_command() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "❓❓ $1 is not installed. Install formatting deps with 'uv pip install .[format]'."
        exit 1
    fi
}

for cmd in black ruff mypy codespell isort; do
    ensure_command "$cmd"
done

sort_imports() { isort "$@"; }
format_files() { black "$@"; }
lint_files() { ruff check "$@"; }
type_check_files() { mypy "$@"; }
spell_check_files() { codespell "$@"; }

process_all() {
    echo "Sorting imports with isort..."
    isort .
    echo "Formatting all Python files with Black..."
    black .
    echo "Linting all Python files with Ruff..."
    ruff check .
    echo "Type-checking with MyPy..."
    mypy
    echo "Checking spelling with Codespell..."
    codespell .
}

process_files() {
    echo "Processing specified files..."
    echo "Sorting imports with isort..."
    sort_imports "$@"
    echo "Formatting all Python files with Black..."
    format_files "$@"
    echo "Linting all Python files with Ruff..."
    lint_files "$@"
    echo "Type-checking with MyPy..."
    type_check_files "$@"
    echo "Checking spelling with Codespell..."
    spell_check_files "$@"
}

process_changed() {
    MERGEBASE="$(git merge-base origin/main HEAD 2>/dev/null || git merge-base main HEAD 2>/dev/null || git merge-base origin/master HEAD 2>/dev/null || git merge-base master HEAD 2>/dev/null || true)"

    if [[ -z "$MERGEBASE" ]]; then
        echo "Could not determine merge-base with main; run with --all, --staged, or specify files."
        exit 1
    fi

    changed_files=$(git diff --name-only "$@" --diff-filter=ACM "$MERGEBASE" -- '*.py')

    if [[ -n "$changed_files" ]]; then
        echo "Processing changed Python files..."
        echo "$changed_files" | xargs -P 5 -n 1 isort
        echo "$changed_files" | xargs -P 5 -n 1 black
        echo "$changed_files" | xargs -P 5 -n 1 ruff check
        echo "$changed_files" | xargs -P 5 -n 1 mypy
        echo "$changed_files" | xargs -P 5 -n 1 codespell
    else
        echo "No changed Python files to process."
    fi
}

case "$1" in
    --all)
        process_all
        ;;
    --staged)
        process_changed --staged
        ;;
    --files)
        shift
        [[ $# -gt 0 ]] || usage
        process_files "$@"
        ;;
    -h|--help)
        usage
        ;;
    *)
        if [[ $# -gt 0 ]]; then
            process_files "$@"
        else
            process_changed
        fi
        ;;
esac

echo "✨🎉 All checks passed! 🎉✨"