#!/usr/bin/env bash
# ==============================================================================
# Maidere CLI Setup & Global PATH Installer
# ==============================================================================
set -e

SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do
  DIR="$(cd -P "$(dirname "$SOURCE")" >/dev/null 2>&1 && pwd)"
  SOURCE="$(readlink "$SOURCE")"
  [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE"
done
SCRIPTS_DIR="$(cd -P "$(dirname "$SOURCE")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -P "$SCRIPTS_DIR/.." >/dev/null 2>&1 && pwd)"

echo "==> Setting up Maidere CLI launcher..."

# 1. Ensure launch.sh is executable
chmod +x "$REPO_ROOT/launch.sh"
echo "  ✓ Made $REPO_ROOT/launch.sh executable"

# 2. Ensure ~/.local/bin directory exists
BIN_DIR="$HOME/.local/bin"
mkdir -p "$BIN_DIR"
echo "  ✓ Verified local bin directory at $BIN_DIR"

# 3. Create / update symlink to ~/.local/bin/maidere
TARGET_LINK="$BIN_DIR/maidere"
ln -sf "$REPO_ROOT/launch.sh" "$TARGET_LINK"
echo "  ✓ Created symlink: $TARGET_LINK -> $REPO_ROOT/launch.sh"

# 4. Check and configure PATH in user shell rc files
PATH_CONFIGURED=false

# Check if ~/.local/bin is already in current PATH
if [[ ":$PATH:" == *":$BIN_DIR:"* ]]; then
  PATH_CONFIGURED=true
fi

add_to_shell_rc() {
  local rc_file="$1"
  if [ -f "$rc_file" ]; then
    if ! grep -q 'export PATH="$HOME/.local/bin:$PATH"' "$rc_file" && ! grep -q 'export PATH=$HOME/.local/bin:$PATH' "$rc_file"; then
      echo "" >> "$rc_file"
      echo '# Added by Maidere CLI installer' >> "$rc_file"
      echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$rc_file"
      echo "  ✓ Added ~/.local/bin to $rc_file"
    else
      echo "  ✓ ~/.local/bin already configured in $rc_file"
    fi
  fi
}

add_to_shell_rc "$HOME/.bashrc"
add_to_shell_rc "$HOME/.zshrc"
add_to_shell_rc "$HOME/.profile"

echo ""
echo "================================================================="
echo "  🎉 Maidere CLI successfully installed!"
echo "================================================================="
echo ""
echo "  You can now launch Maidere from ANY directory in your terminal by typing:"
echo "    maidere"
echo ""
if [ "$PATH_CONFIGURED" = false ]; then
  echo "  Note: To apply PATH changes in your current terminal session, run:"
  echo "    source ~/.bashrc   (or restart your terminal)"
  echo ""
fi
