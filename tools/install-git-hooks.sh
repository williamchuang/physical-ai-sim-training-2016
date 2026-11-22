#!/usr/bin/env bash
# Installs a pre-commit hook that refreshes the repository tree in README.md
# before every commit, so the tree always matches what the commit contains.
#   bash tools/install-git-hooks.sh
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
hook=.git/hooks/pre-commit
cat > "$hook" <<'HOOK'
#!/usr/bin/env bash
# Keeps the repository tree in README.md current (installed by tools/install-git-hooks.sh).
set -e
python3 tools/update_readme_tree.py >/dev/null
git add README.md
HOOK
chmod +x "$hook"
echo "installed $hook"
