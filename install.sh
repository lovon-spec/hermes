#!/bin/bash
set -e

cd "$(dirname "$0")"
PROJECT_DIR="$(pwd)"
APP_BUNDLE="$PROJECT_DIR/Hermes.app"

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
"$PYTHON" -m pip install -q Cython
"$PYTHON" -m pip install -q -r "$PROJECT_DIR/translator_service/requirements.txt"

# 4. Detect deployment target
MACOS_VER=$(sw_vers -productVersion | cut -d. -f1).0
echo "Building for macOS $MACOS_VER..."

# 5. Compile Swift binary
echo "Compiling Hermes..."
swiftc -o /tmp/hermes-build \
    -target arm64-apple-macos${MACOS_VER} \
    -framework ScreenCaptureKit \
    -framework AVFoundation \
    -framework Cocoa \
    -framework QuartzCore \
    -framework CoreAudio \
    "$PROJECT_DIR"/Hermes/*.swift

# 6. Build .app bundle
echo "Creating Hermes.app bundle..."
rm -rf "$APP_BUNDLE"
mkdir -p "$APP_BUNDLE/Contents/MacOS"
mkdir -p "$APP_BUNDLE/Contents/Resources"

mv /tmp/hermes-build "$APP_BUNDLE/Contents/MacOS/Hermes"
cp "$PROJECT_DIR/Hermes/Info.plist" "$APP_BUNDLE/Contents/"
cp -r "$PROJECT_DIR/translator_service" "$APP_BUNDLE/Contents/Resources/translator_service"

# 7. Convert SVG icon to icns (if sips available)
if [ -f "$PROJECT_DIR/static/icon.svg" ]; then
    ICONSET_DIR=$(mktemp -d)/Hermes.iconset
    mkdir -p "$ICONSET_DIR"

    # Render SVG to PNG at required sizes using sips via a temp PNG
    # sips can't read SVG directly, so we use qlmanage for the conversion
    qlmanage -t -s 1024 -o /tmp "$PROJECT_DIR/static/icon.svg" 2>/dev/null || true
    if [ -f "/tmp/icon.svg.png" ]; then
        for size in 16 32 64 128 256 512; do
            sips -z $size $size "/tmp/icon.svg.png" --out "$ICONSET_DIR/icon_${size}x${size}.png" 2>/dev/null
            double=$((size * 2))
            sips -z $double $double "/tmp/icon.svg.png" --out "$ICONSET_DIR/icon_${size}x${size}@2x.png" 2>/dev/null
        done
        iconutil -c icns "$ICONSET_DIR" -o "$APP_BUNDLE/Contents/Resources/AppIcon.icns" 2>/dev/null && \
            echo "App icon set." || echo "Warning: icon conversion failed, using default icon."
        rm -f /tmp/icon.svg.png
    else
        echo "Warning: could not render SVG, using default icon."
    fi
    rm -rf "$(dirname "$ICONSET_DIR")"
fi

# 8. Sign with entitlements
SIGN_IDENTITY=$(security find-identity -v -p codesigning | grep "Hermes Dev" | head -1 | sed 's/.*"\(.*\)".*/\1/')
if [ -n "$SIGN_IDENTITY" ]; then
    codesign --force --sign "$SIGN_IDENTITY" --entitlements "$PROJECT_DIR/Hermes/Hermes.entitlements" "$APP_BUNDLE" && \
        echo "Code signed with '$SIGN_IDENTITY'."
else
    codesign --force --sign - --entitlements "$PROJECT_DIR/Hermes/Hermes.entitlements" "$APP_BUNDLE" 2>/dev/null && \
        echo "Code signed (ad-hoc)." || echo "Warning: codesign failed, ScreenCaptureKit may not work."
fi

echo ""
echo "=== Done ==="
echo "Hermes.app is ready. Open it with:"
echo "  open $APP_BUNDLE"
