#!/usr/bin/env bash
set -euo pipefail
# Clone the three model repos into third_party/ and pin their commits (the commit is
# recorded on first run and stays fixed afterwards).
#
# Idempotent: existing repos are not re-cloned; an existing *.commit lock file keeps
# the recorded commit. Repo contents are gitignored (third_party/*/), while the lock
# files third_party/<repo>.commit are committed.
cd "$(dirname "$0")/.."
mkdir -p third_party
# Use <repo>/.git as the existence check: a leftover non-git directory must not count
# as cloned (otherwise `git -C ... rev-parse` would blow up).
[ -d third_party/vipe/.git ] || git clone https://github.com/nv-tlabs/vipe.git third_party/vipe
[ -d third_party/fast-foundationstereo/.git ] || git clone https://github.com/nvlabs/fast-foundationstereo.git third_party/fast-foundationstereo
[ -d third_party/hamer/.git ] || git clone --recursive https://github.com/geopavlakos/hamer.git third_party/hamer
# First run: record the current HEAD as the pinned commit
for repo in vipe fast-foundationstereo hamer; do
  file="third_party/$repo.commit"
  if [ ! -f "$file" ]; then git -C "third_party/$repo" rev-parse HEAD > "$file"; fi
  git -C "third_party/$repo" checkout "$(cat "$file")"
done
echo "Vendored repos pinned to the recorded commits."
