#!/bin/sh
# Point git at the tracked hooks directory so the secret guard runs on commit.
set -e
root=$(git rev-parse --show-toplevel)
ln -sf ../../scripts/pre-commit "$root/.git/hooks/pre-commit"
echo "pre-commit hook installed"
