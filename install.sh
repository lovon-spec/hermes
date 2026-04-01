#!/bin/bash
set -e

cd "$(dirname "$0")"
PROJECT_DIR="$(pwd)"

echo "=== Hermes Installer ==="

# 1. Check macOS version
SW_VER=$(sw_vers -productVersion | cut -d. -f1)
if [ "$SW_VER" -lt 15 ]; then
    echo "Error: macOS 15+ required (you have $(sw_vers -productVersion))"
    exit 1
fi

# 2. Check Python 3
PYTHON=""
for p in /opt/homebrew/bin/python3 /usr/local/bin/python3 /usr/bin/python3; do
    if [ -x "$p" ]; then PYTHON="$p"; break; fi
done
if [ -z "$PYTHON" ]; then
    echo "Error: python3 not found. Install it with: brew install python3"
    exit 1
fi
echo "Using Python: $PYTHON"

# 3. Install Python dependencies
echo "Installing Python dependencies..."
"$PYTHON" -m pip install -q -r "$PROJECT_DIR/translator_service/requirements.txt"

# 4. Detect deployment target
MACOS_VER=$(sw_vers -productVersion | cut -d. -f1).0
echo "Building for macOS $MACOS_VER..."

# 5. Build Swift binary
echo "Compiling Hermes..."
swiftc -o "$PROJECT_DIR/hermes-app" \
    -target arm64-apple-macos${MACOS_VER} \
    -framework ScreenCaptureKit \
    -framework AVFoundation \
    -framework Cocoa \
    -framework QuartzCore \
    "$PROJECT_DIR"/Hermes/*.swift

echo ""
echo "=== Done ==="
echo "Run with: $PROJECT_DIR/hermes-app"
