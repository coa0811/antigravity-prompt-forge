#!/bin/sh
PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"
export PATH
if command -v python3 >/dev/null 2>&1; then
  exec python3 "$@"
fi
echo "prompt-forge: python3 not found in PATH" >&2
exit 127
