#!/usr/bin/env bash
# tools/sha256_matrix.sh — regression safety net for refactors.
#
# Runs fe_run_file across the supported tiers on this host and the 3
# testaudio clips, producing a {clip,tier} → output SHA256 matrix.
#
# Use case:
#   1. Before refactor:   ./tools/sha256_matrix.sh  > /tmp/pre.txt
#   2. After refactor:    ./tools/sha256_matrix.sh  > /tmp/post.txt
#   3. Diff:              diff /tmp/pre.txt /tmp/post.txt
#                         (empty diff = no regression at byte level)
#
# Cross-tier byte-id within the same matrix run is also a useful sanity
# check (avx2 vs avxvnni on i5; neon/dotprod/i8mm on aarch64 Macs).
#
# Frames default to 500 (fast; covers warmup + steady-state). Override
# with FRAMES env var for deeper coverage.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if [ -z "${RUNNER:-}" ]; then
    for candidate in \
        "${ROOT}/build/fe_run_file" \
        "${ROOT}/build_android/fe_run_file" \
        "${ROOT}/build/Release/fe_run_file"; do
        if [ -x "$candidate" ]; then
            RUNNER="$candidate"
            break
        fi
    done
fi
RUNNER="${RUNNER:-${ROOT}/build/fe_run_file}"
WEIGHTS="${WEIGHTS:-${ROOT}/weights/fe.q8}"
FRAMES="${FRAMES:-500}"
TMPDIR="${TMPDIR:-${ROOT}/build/sha256_tmp}"

mkdir -p "$TMPDIR"

if [ ! -x "$RUNNER" ]; then
    printf 'runner not found/executable: %s\n' "$RUNNER" >&2
    printf 'build first, or pass RUNNER=/path/to/fe_run_file\n' >&2
    exit 2
fi
if [ ! -f "$WEIGHTS" ]; then
    printf 'weights not found: %s\n' "$WEIGHTS" >&2
    exit 2
fi

hash_file() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | cut -c1-16
    else
        shasum -a 256 "$1" | cut -c1-16
    fi
}

CLIPS=(babble car street)
# Try every tier; if not available on this CPU it'll just print "(unavailable)".
TIERS=(neon dotprod i8mm avx2 avxvnni avx512vnni)

printf '# fe_run_file SHA256 matrix\n'
printf '# weights=%s  frames=%s\n' "$(basename "$WEIGHTS")" "$FRAMES"
printf '# %-8s %-12s sha256\n' clip tier
for clip in "${CLIPS[@]}"; do
    INPUT="${ROOT}/testaudio/${clip}_5dB.f32"
    if [ ! -f "$INPUT" ]; then
        printf '# skip %s (no .f32)\n' "$clip" >&2
        continue
    fi
    for tier in "${TIERS[@]}"; do
        OUT="${TMPDIR}/${clip}_${tier}.wav"
        rm -f "$OUT"   # never hash a stale WAV (crash != pass)
        # fe_run_file will return non-zero with "tier 'X' not available"
        # message if the host CPU lacks it.
        if "$RUNNER" "$WEIGHTS" "$INPUT" "$OUT" "$FRAMES" "$tier" \
                > /dev/null 2>/dev/null && [ -s "$OUT" ]; then
            HASH=$(hash_file "$OUT")
            printf '  %-8s %-12s %s\n' "$clip" "$tier" "$HASH"
        else
            printf '  %-8s %-12s (unavailable)\n' "$clip" "$tier"
        fi
    done
done
