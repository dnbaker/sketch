#!/usr/bin/env bash
# Release gate: a sketch release must not break its consumers. Builds and
# tests sketch, then bonsai and dashing2 with THIS sketch commit in place of
# the one their submodule pins, and fails on the first error. Run it on the
# release commit before fast-forwarding main or pushing:
#
#     scripts/release_check.sh
#
# Everything is built from committed trees exported with `git archive` (sketch
# HEAD; the consumers at their release branches; every submodule at its
# pinned commit, taken from the local checkouts, no network), never from a
# working checkout, so files left untracked in a development tree cannot
# hide a file the release dropped.
#
# Environment:
#   BONSAI_REPO   [$DASHING2_REPO/bonsai]  BONSAI_REF   [maint]
#   DASHING2_REPO [~/Desktop/code/dashing2]  DASHING2_REF [maint]
#   RELEASE_CC [gcc-15], RELEASE_CXX [g++-15]; JOBS [2] (the dashing2 build is memory-hungry: keep it low);
#   SKIP_SKETCH=1 / SKIP_BONSAI=1 / SKIP_DASHING2=1 skip a stage;
#   KEEP=1 keeps the work directory (it is always kept on failure).
#
# Consumer checks (each repository owns its tests):
#   bonsai:   make check   (unit tests and the classify/profile CLI smokes)
#   dashing2: test/regress_modes.sh ./dashing2   (sketch/compare goldens)
set -euo pipefail

SKETCH=$(cd "$(dirname "$0")/.." && pwd)
DASHING2_REPO=${DASHING2_REPO:-$HOME/Desktop/code/dashing2}
BONSAI_REPO=${BONSAI_REPO:-$DASHING2_REPO/bonsai}
BONSAI_REF=${BONSAI_REF:-maint}
DASHING2_REF=${DASHING2_REF:-maint}
# Compilers come from RELEASE_CC / RELEASE_CXX only, never the shell's CC/CXX
# (a broken system clang there must not decide a release).
export CC=${RELEASE_CC:-gcc-15} CXX=${RELEASE_CXX:-g++-15}
JOBS=${JOBS:-2}
WORK=$(mktemp -d "${TMPDIR:-/tmp}/sketch_release_XXXXXX")

log() { printf '[release_check] %s\n' "$*"; }
fail() { log "FAIL: $*"; log "work directory kept: $WORK"; KEEP=1; exit 1; }
trap '[ -n "${KEEP:-}" ] || rm -rf "$WORK"' EXIT

SKETCH_REV=$(git -C "$SKETCH" rev-parse HEAD)
git -C "$SKETCH" diff --quiet HEAD -- . ':!pybind11' || log "note: uncommitted changes in $SKETCH are NOT tested (HEAD $SKETCH_REV is)"

# export_tree <repo> <rev> <dest> <role>: the committed tree of <repo> at
# <rev>, then each submodule at its pinned commit from <repo>/<path>,
# recursively; role "bonsai" swaps the hll submodule for the sketch candidate,
# role "dashing2" swaps the bonsai submodule for bonsai at BONSAI_REF.
export_tree() {
    local repo=$1 rev=$2 dst=$3 role=$4 mode type sha path
    mkdir -p "$dst"
    git -C "$repo" archive "$rev" | tar -x -C "$dst"
    while read -r mode type sha path; do
        [ "$type" = commit ] || continue
        rm -rf "${dst:?}/$path"
        if [ "$role" = sketch ] && [ "$path" = pybind11 ]; then
            continue   # Python bindings only; no consumer builds them
        elif [ "$role" = bonsai ] && [ "$path" = hll ]; then
            export_tree "$SKETCH" "$SKETCH_REV" "$dst/$path" sketch
        elif [ "$role" = dashing2 ] && [ "$path" = bonsai ]; then
            export_tree "$BONSAI_REPO" "$BONSAI_REV" "$dst/$path" bonsai
        else
            [ -e "$repo/$path/.git" ] || fail "$repo/$path: submodule not checked out (git -C $repo submodule update --init $path)"
            git -C "$repo/$path" cat-file -e "$sha^{commit}" 2>/dev/null \
                || fail "$repo/$path lacks pinned commit $sha (fetch it there)"
            export_tree "$repo/$path" "$sha" "$dst/$path" other
        fi
    done < <(git -C "$repo" ls-tree -r "$rev")
}

run() {   # run <name> <dir> <cmd...>: log to $WORK/<name>.log
    local name=$1 dir=$2; shift 2
    log "$name: $*"
    (cd "$dir" && "$@") > "$WORK/$name.log" 2>&1 || { tail -n 30 "$WORK/$name.log"; fail "$name (log: $WORK/$name.log)"; }
}

log "sketch $SKETCH_REV, CC=$CC CXX=$CXX, JOBS=$JOBS, work $WORK"
export_tree "$SKETCH" "$SKETCH_REV" "$WORK/sketch" sketch

if [ -z "${SKIP_SKETCH:-}" ]; then
    run sketch-tests "$WORK/sketch" env CXX="$CXX" bash ./run_tests.sh
fi

BONSAI_REV=$(git -C "$BONSAI_REPO" rev-parse "$BONSAI_REF")
if [ -z "${SKIP_BONSAI:-}" ]; then
    log "bonsai $BONSAI_REF = $BONSAI_REV"
    export_tree "$BONSAI_REPO" "$BONSAI_REV" "$WORK/bonsai" bonsai
    grep -q '^check:' "$WORK/bonsai/Makefile" || fail "bonsai $BONSAI_REF has no 'make check' target"
    run bonsai-check "$WORK/bonsai" make check CXX="$CXX" JOBS="$JOBS"
fi

if [ -z "${SKIP_DASHING2:-}" ]; then
    DASHING2_REV=$(git -C "$DASHING2_REPO" rev-parse "$DASHING2_REF")
    log "dashing2 $DASHING2_REF = $DASHING2_REV (bonsai $BONSAI_REV)"
    export_tree "$DASHING2_REPO" "$DASHING2_REV" "$WORK/dashing2" dashing2
    [ -x "$WORK/dashing2/test/regress_modes.sh" ] || fail "dashing2 $DASHING2_REF has no test/regress_modes.sh"
    run dashing2-build "$WORK/dashing2" make dashing2 CC="$CC" CXX="$CXX" -j"$JOBS"
    run dashing2-regress "$WORK/dashing2" bash test/regress_modes.sh ./dashing2
fi

log "PASS: sketch $SKETCH_REV is safe for bonsai $BONSAI_REF and dashing2 $DASHING2_REF"
