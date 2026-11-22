# `tools/` — Repository maintenance scripts

| File | |
|---|---|
| `update_readme_tree.py` | rewrites the repository tree in the main README from the files git tracks, each folder annotated with the tagline of its own README; `--check` only reports |
| `install-git-hooks.sh` | installs a pre-commit hook that runs the script above before every commit |

```sh
bash tools/install-git-hooks.sh                # once per clone
python3 tools/update_readme_tree.py --check    # what the GitHub workflow runs on every push
```

The workflow, `.github/workflows/checks.yml`, also builds the simulation core and runs its tests.
