#!/usr/bin/env bash
#
# Builds everything Chessbook needs: the bundled Stockfish engine and the
# Python dependencies.
#
#   ./build.sh                  engine + Python packages for the current interpreter
#   ./build.sh --venv           additionally create ./.venv and install into it
#   ./build.sh --engine-only    only compile Stockfish
#   ./build.sh --python-only    only install the Python packages
#   ./build.sh --clean          recompile the engine from scratch
#   ./build.sh --jobs 4         limit parallel compile jobs (default: all cores)
#
# The Stockfish architecture is detected from /proc/cpuinfo and can be
# overridden:  ARCH=x86-64-avx2 ./build.sh
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENGINE_SRC="$ROOT/stockfish/src"
ENGINE_BIN="$ROOT/stockfish/stockfish"   # path gui.py expects
VENV="$ROOT/.venv"

do_engine=1
do_python=1
do_venv=0
do_clean=0
JOBS="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 2)"

while [ $# -gt 0 ]; do
    case "$1" in
        --venv)        do_venv=1 ;;
        --engine-only) do_python=0 ;;
        --python-only) do_engine=0 ;;
        --clean)       do_clean=1 ;;
        --jobs)        JOBS="$2"; shift ;;
        -h|--help)     awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "$0"
                       exit 0 ;;
        *)             echo "unknown option: $1 (try --help)" >&2; exit 2 ;;
    esac
    shift
done

say()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[33mwarning: %s\033[0m\n' "$*" >&2; }
die()  { printf '\033[31merror: %s\033[0m\n' "$*" >&2; exit 1; }

# --------------------------------------------------------------------------
# Stockfish architecture: pick the best instruction set this CPU supports.
# --------------------------------------------------------------------------
detect_arch() {
    local machine
    machine="$(uname -m)"

    case "$machine" in
        x86_64|amd64)
            local flags=""
            [ -r /proc/cpuinfo ] && flags="$(grep -m1 '^flags' /proc/cpuinfo || true)"
            case " $flags " in
                *" avx2 "*)   echo "x86-64-avx2" ;;
                *" sse4_1 "*) echo "x86-64-sse41-popcnt" ;;
                *" ssse3 "*)  echo "x86-64-ssse3" ;;
                *)            echo "x86-64" ;;
            esac
            ;;
        aarch64|arm64)
            if [ "$(uname -s)" = "Darwin" ]; then echo "apple-silicon"; else echo "armv8"; fi
            ;;
        *)
            echo "general-64"
            ;;
    esac
}

build_engine() {
    say "Building Stockfish"

    command -v make >/dev/null || die "make not found - install it (Debian/Ubuntu: sudo apt install build-essential)"
    command -v g++  >/dev/null || command -v clang++ >/dev/null || \
        die "no C++ compiler found (Debian/Ubuntu: sudo apt install build-essential)"
    [ -f "$ENGINE_SRC/Makefile" ] || die "$ENGINE_SRC/Makefile is missing - is the stockfish/ directory complete?"

    local arch="${ARCH:-$(detect_arch)}"
    echo "architecture: $arch    jobs: $JOBS"

    if [ "$do_clean" = 1 ]; then
        make -C "$ENGINE_SRC" clean >/dev/null
    fi

    # 'build' also pulls in the NNUE network ('net' target); it ships with this
    # repository, so no download is needed.
    make -C "$ENGINE_SRC" -j"$JOBS" build ARCH="$arch"

    cp -f "$ENGINE_SRC/stockfish" "$ENGINE_BIN"

    # Say hello to the engine.  Note the missing 'head' in the pipe: with
    # 'set -o pipefail' it would close the pipe early, kill the engine with
    # SIGPIPE and take the whole script down with it.
    local banner
    banner="$(printf 'uci\nquit\n' | "$ENGINE_BIN" 2>/dev/null)" || true
    printf '%s -> %s\n' "$ENGINE_BIN" "$(printf '%s\n' "$banner" | head -n1)"
}

# --------------------------------------------------------------------------
# Python side
# --------------------------------------------------------------------------
install_python() {
    say "Installing Python packages"

    local py="python3"
    command -v "$py" >/dev/null || die "python3 not found"

    if [ "$do_venv" = 1 ]; then
        if [ ! -d "$VENV" ]; then
            # --system-site-packages so the distribution's python3-gi (used for
            # silent screenshots) stays visible inside the virtualenv.
            "$py" -m venv --system-site-packages "$VENV" \
                || die "could not create $VENV (Debian/Ubuntu: sudo apt install python3-venv)"
        fi
        py="$VENV/bin/python"
        echo "virtualenv: $VENV"
        "$py" -m pip install --upgrade pip >/dev/null
    fi

    if ! "$py" -m pip install -r "$ROOT/requirements.txt"; then
        # Recent distributions refuse to install into the system interpreter
        # (PEP 668).  A virtualenv is the way out.
        die "pip install failed - retry with:  ./build.sh --venv"
    fi

    "$py" - <<'EOF' || true
try:
    import gi  # noqa: F401
except ImportError:
    print("\nnote: PyGObject (python3-gi) is not available.")
    print("      Screenshots fall back to gnome-screenshot, which flashes the")
    print("      screen and plays the shutter sound on every capture.")
    print("      Fix with:  sudo apt install python3-gi\n")
EOF
}

verify() {
    say "Verifying"

    local py="python3"
    [ "$do_venv" = 1 ] && py="$VENV/bin/python"

    if [ "$do_python" = 1 ]; then
        "$py" - <<'EOF'
import importlib, sys
missing = [m for m in ("PyQt5", "cv2", "numpy", "PIL", "chess", "stockfish", "anytree")
           if importlib.util.find_spec(m) is None]
if missing:
    sys.exit("missing Python modules: " + ", ".join(missing))
print("Python packages OK")
EOF
    fi

    if [ "$do_engine" = 1 ] || [ -x "$ENGINE_BIN" ]; then
        [ -x "$ENGINE_BIN" ] || die "engine binary $ENGINE_BIN is missing - run ./build.sh --engine-only"
        echo "Engine OK"
    fi

    say "Done"
    if [ "$do_venv" = 1 ]; then
        echo "Start the app with:  $VENV/bin/python gui.py"
    else
        echo "Start the app with:  python3 gui.py"
    fi
}

[ "$do_engine" = 1 ] && build_engine
[ "$do_python" = 1 ] && install_python
verify
